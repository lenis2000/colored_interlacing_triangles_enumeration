#include <metal_stdlib>
using namespace metal;

// Check if color c interlaces between level-k state and level-(k+1) state
// Returns true if the dots of color c interlace properly
inline bool check_color_interlacing(
    device const uint8_t* state_k,    // size n*k
    device const uint8_t* state_k1,   // size n*(k+1)
    int n, int k, int color
) {
    int k_dot_count = 0;
    int k1_dot_count = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            // Check level-(k+1) position (i, j)
            uint8_t c_k1_j = state_k1[i * (k + 1) + j];
            if (c_k1_j == color) {
                if (k_dot_count != k1_dot_count) {
                    return false;
                }
                k1_dot_count++;
            }

            // Check level-k position (i, j)
            uint8_t c_k_j = state_k[i * k + j];
            if (c_k_j == color) {
                if (k_dot_count != k1_dot_count - 1) {
                    return false;
                }
                k_dot_count++;
            }
        }

        // Check level-(k+1) position (i, k) - the extra column
        uint8_t c_k1_k = state_k1[i * (k + 1) + k];
        if (c_k1_k == color) {
            if (k_dot_count != k1_dot_count) {
                return false;
            }
            k1_dot_count++;
        }
    }

    // Verify counts
    if (k_dot_count != k || k1_dot_count != (k + 1)) {
        return false;
    }

    return true;
}

// Kernel: Check interlacing for all pairs (state_k[i], state_k1[j])
// Each thread handles one (i, j) pair
kernel void check_interlacing_kernel(
    device const uint8_t* states_k [[buffer(0)]],   // M_k states, each n*k bytes
    device const uint8_t* states_k1 [[buffer(1)]],  // M_k1 states, each n*(k+1) bytes
    device uint8_t* results [[buffer(2)]],          // M_k * M_k1 results
    constant uint32_t& n [[buffer(3)]],
    constant uint32_t& k [[buffer(4)]],
    constant uint32_t& M_k [[buffer(5)]],
    constant uint32_t& M_k1 [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint32_t i = gid.x;  // index into states_k
    uint32_t j = gid.y;  // index into states_k1

    if (i >= M_k || j >= M_k1) {
        return;
    }

    uint32_t state_k_size = n * k;
    uint32_t state_k1_size = n * (k + 1);

    device const uint8_t* state_k_ptr = states_k + i * state_k_size;
    device const uint8_t* state_k1_ptr = states_k1 + j * state_k1_size;

    // Check interlacing for all colors
    bool valid = true;
    for (uint32_t color = 1; color <= n && valid; ++color) {
        valid = check_color_interlacing(state_k_ptr, state_k1_ptr, n, k, color);
    }

    results[i * M_k1 + j] = valid ? 1 : 0;
}
