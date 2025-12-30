// CPU Enumerator for T_N(n) with progress reporting
// Uses level-by-level DP with canonical states (S_n symmetry)

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdint>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

using State = std::vector<uint8_t>;

uint64_t factorial(int n) {
    uint64_t result = 1;
    for (int i = 2; i <= n; ++i) result *= i;
    return result;
}

// Calculate canonical state count (first=1, last=n)
uint64_t calculate_canonical_count(int n, int k) {
    if (k == 0) return 0;
    if (k == 1) return 1;
    if (n == 1) return 1;
    uint64_t num = factorial(n * k - 2);
    uint64_t den = factorial(k - 1);
    for (int c = 2; c < n; ++c) den *= factorial(k);
    den *= factorial(k - 1);
    return num / den;
}

// Generate canonical states (first=1, last=n)
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

    uint64_t max_count = calculate_canonical_count(n, k);
    buffer.resize(max_count * state_size);

    uint64_t count = 0;
    do {
        uint8_t* dest = buffer.data() + count * state_size;
        dest[0] = 1;
        std::copy(middle.begin(), middle.end(), dest + 1);
        dest[state_size - 1] = n;
        count++;
    } while (std::next_permutation(middle.begin(), middle.end()));

    buffer.resize(count * state_size);
    return count;
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

// Count boundary-ordered states (without generating)
uint64_t count_boundary_ordered_states(int n, int k) {
    if (k == 1) return 1;

    State middle;
    for (int i = 0; i < k - 1; ++i) middle.push_back(1);
    for (int c = 2; c < n; ++c)
        for (int i = 0; i < k; ++i) middle.push_back(c);
    for (int i = 0; i < k - 1; ++i) middle.push_back(n);
    std::sort(middle.begin(), middle.end());

    uint64_t state_size = n * k;
    State temp(state_size);
    uint64_t count = 0;

    do {
        temp[0] = 1;
        std::copy(middle.begin(), middle.end(), temp.begin() + 1);
        temp[state_size - 1] = n;

        if (has_boundary_ordering(temp.data(), n, k)) {
            count++;
        }
    } while (std::next_permutation(middle.begin(), middle.end()));

    return count;
}

// Generate only boundary-ordered canonical states (for final level optimization)
uint64_t generate_boundary_ordered_states(int n, int k, std::vector<uint8_t>& buffer) {
    uint64_t state_size = n * k;
    if (k == 1) {
        buffer.resize(state_size);
        for (int i = 0; i < n; ++i) buffer[i] = i + 1;
        return 1;
    }

    // First count to allocate exact size
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

// Check interlacing between consecutive level states
bool check_interlacing(const uint8_t* s_k, const uint8_t* s_k1, int n, int k) {
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

uint64_t count_triangles(int n, int N) {
    if (N < 1) return 0;

    auto total_start = std::chrono::steady_clock::now();

    // Calculate and display state counts
    std::vector<uint64_t> M(N + 1);
    std::cout << "Canonical state counts:" << std::endl;
    for (int k = 1; k <= N; ++k) {
        M[k] = calculate_canonical_count(n, k);
        std::cout << "  Level " << k << ": " << M[k] << std::endl;
    }

    // Initialize level 1
    std::vector<uint8_t> states_curr;
    generate_canonical_states(n, 1, states_curr);
    std::vector<uint64_t> counts_curr = {1};
    uint64_t M_curr = 1;

    // Process level transitions
    for (int k = 1; k < N; ++k) {
        auto level_start = std::chrono::steady_clock::now();

        size_t state_k_size = n * k;
        size_t state_k1_size = n * (k + 1);

        std::cout << "\nLevel " << k << " -> " << (k + 1)
                  << " (formula: " << M[k] << " x " << M[k+1] << ")" << std::endl;

        // Collect valid source states
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

        // Generate next level states
        // Use boundary-ordered states only for the FINAL level (2^(n-1) speedup)
        std::vector<uint8_t> states_next;
        uint64_t actual_M_next;
        bool is_final_level = (k + 1 == N);

        if (is_final_level) {
            actual_M_next = generate_boundary_ordered_states(n, k + 1, states_next);
            std::cout << "  Final level: using boundary-ordered states only" << std::endl;
        } else {
            actual_M_next = generate_canonical_states(n, k + 1, states_next);
        }

        std::cout << "  Actual states: " << num_valid << " x " << actual_M_next
                  << " = " << (num_valid * actual_M_next) << " pairs" << std::endl;

        std::vector<uint64_t> counts_next(actual_M_next, 0);

        auto compute_start = std::chrono::steady_clock::now();

        #ifdef _OPENMP
        int num_threads = omp_get_max_threads();
        std::cout << "  Using " << num_threads << " OpenMP threads" << std::endl;
        #endif

        // Process all pairs with progress
        uint64_t total_pairs = num_valid * actual_M_next;
        std::atomic<uint64_t> sources_done{0};

        // Report interval: every 1% or every 1000 sources, whichever is larger
        uint64_t report_interval = std::max((uint64_t)1000, num_valid / 100);

        #pragma omp parallel for schedule(dynamic, 100)
        for (uint64_t i = 0; i < num_valid; ++i) {
            const uint8_t* s_k = valid_states.data() + i * state_k_size;
            uint64_t cnt = valid_cnt[i];

            for (uint64_t j = 0; j < actual_M_next; ++j) {
                const uint8_t* s_k1 = states_next.data() + j * state_k1_size;
                if (check_interlacing(s_k, s_k1, n, k)) {
                    #pragma omp atomic
                    counts_next[j] += cnt;
                }
            }

            uint64_t done = ++sources_done;

            // Progress reporting - only every report_interval to minimize overhead
            if (done % report_interval == 0 || done == num_valid) {
                #pragma omp critical
                {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(now - compute_start).count();
                    double progress = (double)done / num_valid;
                    double eta = elapsed > 0 ? (elapsed / progress) - elapsed : 0;
                    double rate = (done * actual_M_next) / elapsed / 1e6;

                    std::cout << "\r  Progress: " << done << "/" << num_valid
                              << " sources (" << (int)(progress * 100) << "%) "
                              << "[" << (int)elapsed << "s, ETA " << (int)eta << "s, "
                              << rate << " Mpairs/s]     " << std::flush;
                }
            }
        }

        auto compute_end = std::chrono::steady_clock::now();
        double compute_s = std::chrono::duration<double>(compute_end - compute_start).count();

        std::cout << "\n  Computed " << total_pairs << " pairs in " << compute_s << "s"
                  << " (" << (total_pairs / 1e6 / compute_s) << " Mpairs/s)" << std::endl;

        // Update for next level
        states_curr = std::move(states_next);
        counts_curr = std::move(counts_next);
        M_curr = actual_M_next;

        auto level_end = std::chrono::steady_clock::now();
        double level_s = std::chrono::duration<double>(level_end - level_start).count();
        std::cout << "  Level time: " << level_s << "s" << std::endl;
    }

    // Sum final counts
    uint64_t canonical_count = 0;
    for (auto c : counts_curr) canonical_count += c;

    auto total_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(total_end - total_start).count();
    std::cout << "\nTotal time: " << total_s << "s" << std::endl;

    // Multiply by n! (S_n symmetry) and 2^(n-1) (boundary ordering symmetry)
    uint64_t symmetry_factor = factorial(n) * (1ULL << (n - 1));
    std::cout << "Symmetry factor: " << factorial(n) << " * " << (1ULL << (n-1))
              << " = " << symmetry_factor << std::endl;
    return canonical_count * symmetry_factor;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <n> <N>" << std::endl;
        std::cout << "  n = number of colors" << std::endl;
        std::cout << "  N = depth of triangle" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);

    std::cout << "=== CPU Enumerator (OpenMP) ===" << std::endl;
    std::cout << "Computing T_" << N << "(" << n << ")..." << std::endl;

    uint64_t count = count_triangles(n, N);

    std::cout << "\n========================================" << std::endl;
    std::cout << "T_" << N << "(" << n << ") = " << count << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
