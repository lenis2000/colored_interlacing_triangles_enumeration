// Unified Metal GPU + Disk-Chunked Enumerator for T_N(n)
// - Metal GPU for fast interlacing checks
// - Disk-based chunking when state space exceeds memory threshold
// - Works for any N, n

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdint>
#include <sys/stat.h>
#include "MetalController.h"

using State = std::vector<uint8_t>;

// Configuration
constexpr size_t MAX_GPU_BATCH = 500000000;      // 500MB max GPU batch
constexpr uint64_t DISK_CHUNK_SIZE = 50000000;   // 50M states per disk chunk
constexpr uint64_t DISK_THRESHOLD = 100000000;   // Use disk chunking if > 100M states

// ============================================================================
// Utilities
// ============================================================================

uint64_t factorial(int n) {
    uint64_t result = 1;
    for (int i = 2; i <= n; ++i) result *= i;
    return result;
}

void create_directory(const std::string& path) { mkdir(path.c_str(), 0755); }

void save_progress(const std::string& dir, int level, uint64_t chunk, uint64_t sum) {
    std::ofstream f(dir + "/progress.txt");
    f << level << " " << chunk << " " << sum << std::endl;
}

bool load_progress(const std::string& dir, int& level, uint64_t& chunk, uint64_t& sum) {
    std::ifstream f(dir + "/progress.txt");
    if (!f) return false;
    f >> level >> chunk >> sum;
    return f.good();
}

// ============================================================================
// State generation - canonical (first=1, last=n) for S_n symmetry
// ============================================================================

uint64_t calculate_canonical_count(int n, int k) {
    if (k == 0) return 0;
    if (k == 1) return 1;
    if (n == 1) return 1;
    // States with first=1, last=n: permutations of middle (n*k - 2 elements)
    // Middle has (k-1) ones, k of each 2..(n-1), (k-1) n's
    uint64_t num = factorial(n * k - 2);
    uint64_t den = factorial(k - 1);
    for (int c = 2; c < n; ++c) den *= factorial(k);
    den *= factorial(k - 1);
    return num / den;
}

// Check if state has boundary ordering: state[i*k - 1] < state[i*k] for i = 1..n-1
bool has_boundary_ordering(const uint8_t* state, int n, int k) {
    for (int i = 1; i < n; ++i) {
        if (state[i * k - 1] >= state[i * k]) {
            return false;
        }
    }
    return true;
}

// Generate all canonical states into flat buffer
uint64_t generate_canonical_states(int n, int k, std::vector<uint8_t>& buffer) {
    uint64_t state_size = n * k;
    if (k == 1) {
        buffer.resize(state_size);
        for (int i = 0; i < n; ++i) buffer[i] = i + 1;
        return 1;
    }

    State middle;
    for (int i = 0; i < k - 1; ++i) middle.push_back(1);
    for (int c = 2; c < n; ++c)
        for (int i = 0; i < k; ++i) middle.push_back(c);
    for (int i = 0; i < k - 1; ++i) middle.push_back(n);
    std::sort(middle.begin(), middle.end());

    uint64_t expected = calculate_canonical_count(n, k);
    buffer.resize(expected * state_size);

    uint64_t count = 0;
    do {
        uint8_t* dest = buffer.data() + count * state_size;
        dest[0] = 1;
        std::copy(middle.begin(), middle.end(), dest + 1);
        dest[state_size - 1] = n;
        count++;
    } while (std::next_permutation(middle.begin(), middle.end()));

    return count;
}

// Generate range of canonical states [start_idx, start_idx + max_count)
uint64_t generate_canonical_range(int n, int k, std::vector<uint8_t>& buffer,
                                   uint64_t start_idx, uint64_t max_count) {
    uint64_t state_size = n * k;
    if (k == 1) {
        if (start_idx > 0) return 0;
        buffer.resize(state_size);
        for (int i = 0; i < n; ++i) buffer[i] = i + 1;
        return 1;
    }

    State middle;
    for (int i = 0; i < k - 1; ++i) middle.push_back(1);
    for (int c = 2; c < n; ++c)
        for (int i = 0; i < k; ++i) middle.push_back(c);
    for (int i = 0; i < k - 1; ++i) middle.push_back(n);
    std::sort(middle.begin(), middle.end());

    // Skip to start_idx
    for (uint64_t i = 0; i < start_idx; ++i) {
        if (!std::next_permutation(middle.begin(), middle.end())) return 0;
    }

    buffer.resize(max_count * state_size);
    uint64_t count = 0;
    do {
        uint8_t* dest = buffer.data() + count * state_size;
        dest[0] = 1;
        std::copy(middle.begin(), middle.end(), dest + 1);
        dest[state_size - 1] = n;
        count++;
        if (count >= max_count) break;
    } while (std::next_permutation(middle.begin(), middle.end()));

    buffer.resize(count * state_size);
    return count;
}

// Generate range of boundary-ordered states [start_idx, start_idx + max_count)
// where start_idx is the index among boundary-ordered states only
uint64_t generate_boundary_ordered_range(int n, int k, std::vector<uint8_t>& buffer,
                                          uint64_t start_idx, uint64_t max_count) {
    uint64_t state_size = n * k;
    if (k == 1) {
        if (start_idx > 0) return 0;
        buffer.resize(state_size);
        for (int i = 0; i < n; ++i) buffer[i] = i + 1;
        return 1;
    }

    State middle;
    for (int i = 0; i < k - 1; ++i) middle.push_back(1);
    for (int c = 2; c < n; ++c)
        for (int i = 0; i < k; ++i) middle.push_back(c);
    for (int i = 0; i < k - 1; ++i) middle.push_back(n);
    std::sort(middle.begin(), middle.end());

    buffer.resize(max_count * state_size);
    uint64_t boundary_idx = 0;  // Count of boundary-ordered states seen
    uint64_t count = 0;         // Count of states added to buffer
    uint64_t total = calculate_canonical_count(n, k);
    uint64_t checked = 0;
    uint64_t report_interval = std::max(total / 20, (uint64_t)10000000);
    auto gen_start = std::chrono::steady_clock::now();

    do {
        // Build state temporarily to check boundary ordering
        State temp(state_size);
        temp[0] = 1;
        std::copy(middle.begin(), middle.end(), temp.begin() + 1);
        temp[state_size - 1] = n;

        if (has_boundary_ordering(temp.data(), n, k)) {
            if (boundary_idx >= start_idx) {
                uint8_t* dest = buffer.data() + count * state_size;
                std::copy(temp.begin(), temp.end(), dest);
                count++;
                if (count >= max_count) break;
            }
            boundary_idx++;
        }
        checked++;
        if (checked % report_interval == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - gen_start).count();
            std::cout << "\r      Scanning: " << (checked * 100 / total) << "% ("
                      << count << "/" << max_count << " found)    " << std::flush;
        }
    } while (std::next_permutation(middle.begin(), middle.end()));

    buffer.resize(count * state_size);
    return count;
}

// Count boundary-ordered states (without generating) - with progress
uint64_t count_boundary_ordered_states(int n, int k) {
    if (k == 1) return 1;

    State middle;
    for (int i = 0; i < k - 1; ++i) middle.push_back(1);
    for (int c = 2; c < n; ++c)
        for (int i = 0; i < k; ++i) middle.push_back(c);
    for (int i = 0; i < k - 1; ++i) middle.push_back(n);
    std::sort(middle.begin(), middle.end());

    uint64_t total = calculate_canonical_count(n, k);
    uint64_t state_size = n * k;
    State temp(state_size);
    uint64_t count = 0;
    uint64_t checked = 0;
    uint64_t report_interval = std::max(total / 100, (uint64_t)1000000);

    std::cout << "    Counting boundary-ordered states (from " << total << ")..." << std::flush;
    auto start = std::chrono::steady_clock::now();

    do {
        temp[0] = 1;
        std::copy(middle.begin(), middle.end(), temp.begin() + 1);
        temp[state_size - 1] = n;

        if (has_boundary_ordering(temp.data(), n, k)) {
            count++;
        }
        checked++;
        if (checked % report_interval == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double pct = 100.0 * checked / total;
            double eta = (elapsed / pct) * (100 - pct);
            std::cout << "\r    Counting: " << (int)pct << "% (" << count << " found, ETA " << (int)eta << "s)    " << std::flush;
        }
    } while (std::next_permutation(middle.begin(), middle.end()));

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "\r    Found " << count << " boundary-ordered states in " << (int)elapsed << "s" << std::endl;

    return count;
}

// Generate only boundary-ordered canonical states
uint64_t generate_boundary_ordered_states(int n, int k, std::vector<uint8_t>& buffer) {
    uint64_t state_size = n * k;
    if (k == 1) {
        buffer.resize(state_size);
        for (int i = 0; i < n; ++i) buffer[i] = i + 1;
        return 1;
    }

    uint64_t bo_count = count_boundary_ordered_states(n, k);

    State middle;
    for (int i = 0; i < k - 1; ++i) middle.push_back(1);
    for (int c = 2; c < n; ++c)
        for (int i = 0; i < k; ++i) middle.push_back(c);
    for (int i = 0; i < k - 1; ++i) middle.push_back(n);
    std::sort(middle.begin(), middle.end());

    buffer.resize(bo_count * state_size);

    uint64_t count = 0;
    do {
        uint8_t* dest = buffer.data() + count * state_size;
        dest[0] = 1;
        std::copy(middle.begin(), middle.end(), dest + 1);
        dest[state_size - 1] = n;

        if (has_boundary_ordering(dest, n, k)) {
            count++;
        }
    } while (std::next_permutation(middle.begin(), middle.end()));

    return count;
}

// ============================================================================
// CPU interlacing check (fallback)
// ============================================================================

bool check_interlacing_cpu(const uint8_t* s_k, const uint8_t* s_k1, int n, int k) {
    for (int color = 1; color <= n; ++color) {
        int k_dot = 0, k1_dot = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                if (s_k1[i * (k + 1) + j] == color) {
                    if (k_dot != k1_dot) return false;
                    k1_dot++;
                }
                if (s_k[i * k + j] == color) {
                    if (k_dot != k1_dot - 1) return false;
                    k_dot++;
                }
            }
            if (s_k1[i * (k + 1) + k] == color) {
                if (k_dot != k1_dot) return false;
                k1_dot++;
            }
        }
        if (k_dot != k || k1_dot != k + 1) return false;
    }
    return true;
}

// ============================================================================
// Main enumeration
// ============================================================================

uint64_t count_triangles(int n, int N, MetalController* metal) {
    if (N < 1) return 0;

    auto total_start = std::chrono::steady_clock::now();
    std::string checkpoint_dir = "./checkpoints_T" + std::to_string(N) + "_" + std::to_string(n);

    // Calculate state counts for all levels
    std::vector<uint64_t> M(N + 1);
    std::cout << "Canonical state counts:" << std::endl;
    bool needs_disk = false;
    for (int k = 1; k <= N; ++k) {
        M[k] = calculate_canonical_count(n, k);
        std::cout << "  Level " << k << ": " << M[k] << std::endl;
        if (M[k] > DISK_THRESHOLD) needs_disk = true;
    }

    if (needs_disk) {
        create_directory(checkpoint_dir);
        std::cout << "Using disk-based chunking (checkpoints: " << checkpoint_dir << ")" << std::endl;
    }

    // Initialize: level 1 has single canonical state with count 1
    std::vector<uint8_t> states_curr;
    generate_canonical_states(n, 1, states_curr);
    std::vector<uint64_t> counts_curr = {1};
    uint64_t M_curr = 1;

    // Check for resume
    int resume_level = 0;
    uint64_t resume_chunk = 0, resume_sum = 0;
    if (needs_disk && load_progress(checkpoint_dir, resume_level, resume_chunk, resume_sum)) {
        std::cout << "Resume info found: level=" << resume_level << " chunk=" << resume_chunk << std::endl;
    }

    // Process level transitions
    for (int k = 1; k < N; ++k) {
        auto level_start = std::chrono::steady_clock::now();

        size_t state_k_size = n * k;
        size_t state_k1_size = n * (k + 1);

        std::cout << "\nLevel " << k << " -> " << (k + 1)
                  << " (formula: " << M[k] << " x " << M[k+1] << ")" << std::endl;

        // Collect valid source states (nonzero counts)
        std::vector<uint64_t> valid_idx, valid_cnt;
        for (uint64_t i = 0; i < M_curr; ++i) {
            if (counts_curr[i] > 0) {
                valid_idx.push_back(i);
                valid_cnt.push_back(counts_curr[i]);
            }
        }
        uint64_t num_valid = valid_idx.size();
        std::cout << "  Valid sources: " << num_valid << std::endl;

        // Extract valid source states
        std::vector<uint8_t> valid_states(num_valid * state_k_size);
        for (uint64_t i = 0; i < num_valid; ++i) {
            std::copy(states_curr.data() + valid_idx[i] * state_k_size,
                      states_curr.data() + (valid_idx[i] + 1) * state_k_size,
                      valid_states.data() + i * state_k_size);
        }

        // Check if this is the final level - use boundary ordering for 2^(n-1) speedup
        bool is_final_level = (k + 1 == N);

        // Get count for next level
        uint64_t actual_M_next;
        uint64_t formula_count = M[k + 1];

        // For final level, check if we need disk chunking BEFORE counting
        // Use estimate: boundary-ordered ≈ canonical / 2^(n-1) (actually better)
        bool use_disk;
        if (is_final_level) {
            uint64_t estimated_bo = formula_count / (1ULL << (n - 1));
            use_disk = (estimated_bo > DISK_THRESHOLD);
            if (!use_disk) {
                // Small enough to count exactly and fit in memory
                actual_M_next = count_boundary_ordered_states(n, k + 1);
            } else {
                // Too large - will generate in chunks, count during generation
                actual_M_next = estimated_bo;  // Use estimate for now
            }
            std::cout << "  Final level: using boundary-ordered states only" << std::endl;
        } else {
            actual_M_next = formula_count;
            use_disk = (actual_M_next > DISK_THRESHOLD);
        }

        std::cout << "  Actual states: " << num_valid << " x " << actual_M_next
                  << " = " << (num_valid * actual_M_next) << " pairs" << std::endl;

        // Determine disk chunking
        uint64_t num_disk_chunks = use_disk ? (actual_M_next + DISK_CHUNK_SIZE - 1) / DISK_CHUNK_SIZE : 1;

        if (use_disk) {
            std::cout << "  Disk chunks: " << num_disk_chunks << std::endl;
        }

        // Only generate states in memory for non-disk mode
        std::vector<uint8_t> states_next;
        std::vector<uint64_t> counts_next;
        if (!use_disk) {
            if (is_final_level) {
                actual_M_next = generate_boundary_ordered_states(n, k + 1, states_next);
            } else {
                actual_M_next = generate_canonical_states(n, k + 1, states_next);
            }
            counts_next.resize(actual_M_next, 0);
        }

        uint64_t running_sum = 0;
        uint64_t start_chunk = 0;

        // Handle resume
        if (use_disk && resume_level == k && resume_chunk > 0) {
            start_chunk = resume_chunk;
            running_sum = resume_sum;
            std::cout << "  Resuming from chunk " << start_chunk << ", sum=" << running_sum << std::endl;
        }

        for (uint64_t dc = start_chunk; dc < num_disk_chunks; ++dc) {
            uint64_t dc_start = dc * DISK_CHUNK_SIZE;
            uint64_t dc_size = std::min(DISK_CHUNK_SIZE, actual_M_next - dc_start);

            if (use_disk) {
                std::cout << "  Chunk " << (dc + 1) << "/" << num_disk_chunks
                          << " [" << dc_start << ".." << (dc_start + dc_size - 1) << "]" << std::endl;

                // Generate target states for this disk chunk
                std::cout << "    Generating states..." << std::flush;
                auto gen_start = std::chrono::steady_clock::now();
                if (is_final_level) {
                    generate_boundary_ordered_range(n, k + 1, states_next, dc_start, dc_size);
                } else {
                    generate_canonical_range(n, k + 1, states_next, dc_start, dc_size);
                }
                auto gen_end = std::chrono::steady_clock::now();
                auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
                std::cout << " done (" << gen_ms << "ms)" << std::endl;
            } else {
                dc_size = actual_M_next;
                // states_next already generated above
            }

            std::vector<uint64_t> chunk_counts(dc_size, 0);

            // GPU memory chunking
            size_t gpu_chunk = num_valid;
            if (num_valid * dc_size > MAX_GPU_BATCH) {
                gpu_chunk = MAX_GPU_BATCH / dc_size;
                if (gpu_chunk < 1) gpu_chunk = 1;
            }
            size_t num_gpu = (num_valid + gpu_chunk - 1) / gpu_chunk;

            auto compute_start = std::chrono::steady_clock::now();
            if (use_disk) {
                std::cout << "    Processing " << num_gpu << " GPU batches..." << std::flush;
            }

            for (size_t gc = 0; gc < num_gpu; ++gc) {
                size_t gc_start = gc * gpu_chunk;
                size_t gc_end = std::min(gc_start + gpu_chunk, (size_t)num_valid);
                uint32_t gc_size = gc_end - gc_start;

                std::vector<uint8_t> gc_states(gc_size * state_k_size);
                std::copy(valid_states.data() + gc_start * state_k_size,
                          valid_states.data() + gc_end * state_k_size,
                          gc_states.data());

                std::vector<uint8_t> trans;
                if (metal) {
                    trans = metal->checkInterlacingBatch(n, k, gc_states, states_next,
                                                          gc_size, (uint32_t)dc_size);
                } else {
                    // CPU fallback
                    trans.resize(gc_size * dc_size);
                    for (uint32_t i = 0; i < gc_size; ++i) {
                        for (uint64_t j = 0; j < dc_size; ++j) {
                            trans[i * dc_size + j] = check_interlacing_cpu(
                                gc_states.data() + i * state_k_size,
                                states_next.data() + j * state_k1_size, n, k) ? 1 : 0;
                        }
                    }
                }

                // Accumulate counts
                for (uint32_t i = 0; i < gc_size; ++i) {
                    uint64_t cnt = valid_cnt[gc_start + i];
                    for (uint64_t j = 0; j < dc_size; ++j) {
                        if (trans[i * dc_size + j]) {
                            chunk_counts[j] += cnt;
                        }
                    }
                }

                // Progress reporting
                if (num_gpu > 1) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(now - compute_start).count();
                    double progress = (double)(gc + 1) / num_gpu;
                    double eta = (elapsed / progress) - elapsed;
                    uint64_t pairs_done = (gc + 1) * gpu_chunk * dc_size;
                    double rate = pairs_done / elapsed / 1e9;

                    std::cout << "\r    GPU batch " << (gc + 1) << "/" << num_gpu
                              << " (" << (int)(progress * 100) << "%) "
                              << "[" << (int)elapsed << "s elapsed, ETA " << (int)eta << "s, "
                              << rate << " Gpairs/s]     " << std::flush;
                }
            }
            if (num_gpu > 1 && !use_disk) std::cout << std::endl;

            auto compute_end = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start).count();

            if (use_disk) {
                uint64_t pairs = (uint64_t)num_valid * dc_size;
                double rate = pairs / 1e9 / (ms / 1000.0 + 0.001);
                std::cout << " done (" << ms << "ms, " << rate << " Gpairs/s)" << std::endl;
            }

            // Sum this chunk
            uint64_t chunk_sum = 0;
            for (auto c : chunk_counts) chunk_sum += c;
            running_sum += chunk_sum;

            if (use_disk) {
                double chunk_progress = (double)(dc + 1) / num_disk_chunks;
                auto level_now = std::chrono::steady_clock::now();
                double level_elapsed = std::chrono::duration<double>(level_now - level_start).count();
                double level_eta = (level_elapsed / chunk_progress) - level_elapsed;

                std::cout << "  Chunk " << (dc + 1) << "/" << num_disk_chunks << " done"
                          << " | sum=" << chunk_sum << " | running=" << running_sum
                          << " | " << ms << "ms"
                          << " | level ETA: " << (int)(level_eta/60) << "m " << (int)level_eta%60 << "s"
                          << std::endl;
                save_progress(checkpoint_dir, k, dc + 1, running_sum);
            } else {
                counts_next = std::move(chunk_counts);
                std::cout << "  Computed " << (num_valid * dc_size) << " pairs in " << ms << "ms"
                          << " (" << (num_valid * dc_size / 1e6 / (ms/1000.0 + 0.001)) << " Mpairs/s)"
                          << std::endl;
            }
        }

        // Prepare for next level
        if (use_disk) {
            // For disk mode, we only need the sum going forward
            // If this is the final level, running_sum is our canonical count
            if (k == N - 1) {
                counts_curr.clear();
                counts_curr.push_back(running_sum);
                M_curr = 1;
            } else {
                // Need to regenerate states and counts for next level
                // This shouldn't happen often - disk mode is usually for final level
                generate_canonical_states(n, k + 1, states_curr);
                counts_curr.resize(actual_M_next);
                // Would need to reload from disk chunks - simplified: recalculate
                std::cerr << "Warning: Multi-level disk mode not fully implemented" << std::endl;
                M_curr = actual_M_next;
            }
        } else {
            states_curr = std::move(states_next);
            counts_curr = std::move(counts_next);
            M_curr = actual_M_next;
        }

        auto level_end = std::chrono::steady_clock::now();
        auto level_s = std::chrono::duration_cast<std::chrono::seconds>(level_end - level_start).count();
        std::cout << "  Level time: " << level_s << "s" << std::endl;

        // Reset resume after first level processed
        resume_level = 0;
        resume_chunk = 0;
    }

    // Sum final counts (already boundary-ordered from final level)
    uint64_t canonical_count = 0;
    for (auto c : counts_curr) canonical_count += c;

    auto total_end = std::chrono::steady_clock::now();
    auto total_s = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start).count();
    std::cout << "\nTotal time: " << total_s << "s" << std::endl;

    // Multiply by n! (S_n symmetry) and 2^(n-1) (boundary ordering symmetry)
    uint64_t symmetry_factor = factorial(n) * (1ULL << (n - 1));
    std::cout << "Symmetry factor: " << factorial(n) << " * " << (1ULL << (n-1))
              << " = " << symmetry_factor << std::endl;
    return canonical_count * symmetry_factor;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <n> <N> [--cpu]" << std::endl;
        std::cout << "  n = number of colors" << std::endl;
        std::cout << "  N = depth of triangle" << std::endl;
        std::cout << "  --cpu  Force CPU-only mode" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    bool force_cpu = false;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--cpu") force_cpu = true;
    }

    std::cout << "=== Metal GPU + Disk-Chunked Enumerator ===" << std::endl;
    std::cout << "Computing T_" << N << "(" << n << ")..." << std::endl;

    MetalController* metal = nullptr;
    if (!force_cpu) {
        metal = new MetalController();
        if (!metal->initialize()) {
            std::cerr << "Metal init failed, using CPU" << std::endl;
            delete metal;
            metal = nullptr;
        } else {
            std::cout << "GPU: " << metal->getDeviceName() << std::endl;
        }
    } else {
        std::cout << "CPU-only mode" << std::endl;
    }

    uint64_t count = count_triangles(n, N, metal);

    std::cout << "\n========================================" << std::endl;
    std::cout << "T_" << N << "(" << n << ") = " << count << std::endl;
    std::cout << "========================================" << std::endl;

    if (metal) delete metal;
    return 0;
}
