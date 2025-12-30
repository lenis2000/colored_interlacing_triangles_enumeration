#ifndef METAL_CONTROLLER_H
#define METAL_CONTROLLER_H

#include <cstdint>
#include <vector>
#include <string>

// Metal controller for general triangle enumeration
// Parallelizes interlacing checks between consecutive levels

class MetalController {
public:
    MetalController();
    ~MetalController();

    bool initialize();
    std::string getDeviceName() const;

    // Check interlacing between level-k states and level-(k+1) states
    // states_k: flattened array of M_k states, each of size n*k
    // states_k1: flattened array of M_{k+1} states, each of size n*(k+1)
    // Returns: vector of size M_k * M_{k+1}, where result[i * M_{k+1} + j] = 1 if valid
    std::vector<uint8_t> checkInterlacingBatch(
        int n, int k,
        const std::vector<uint8_t>& states_k,
        const std::vector<uint8_t>& states_k1,
        uint32_t M_k,
        uint32_t M_k1
    );

private:
    void* device;           // id<MTLDevice>
    void* commandQueue;     // id<MTLCommandQueue>
    void* computePipeline;  // id<MTLComputePipelineState>
};

#endif // METAL_CONTROLLER_H
