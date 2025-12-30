#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "MetalController.h"
#include <iostream>

// Metal shader source (embedded)
static NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

inline bool check_color_interlacing(
    device const uint8_t* state_k,
    device const uint8_t* state_k1,
    int n, int k, int color
) {
    int k_dot_count = 0;
    int k1_dot_count = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            uint8_t c_k1_j = state_k1[i * (k + 1) + j];
            if (c_k1_j == color) {
                if (k_dot_count != k1_dot_count) return false;
                k1_dot_count++;
            }
            uint8_t c_k_j = state_k[i * k + j];
            if (c_k_j == color) {
                if (k_dot_count != k1_dot_count - 1) return false;
                k_dot_count++;
            }
        }
        uint8_t c_k1_k = state_k1[i * (k + 1) + k];
        if (c_k1_k == color) {
            if (k_dot_count != k1_dot_count) return false;
            k1_dot_count++;
        }
    }
    return (k_dot_count == k && k1_dot_count == (k + 1));
}

kernel void check_interlacing_kernel(
    device const uint8_t* states_k [[buffer(0)]],
    device const uint8_t* states_k1 [[buffer(1)]],
    device uint8_t* results [[buffer(2)]],
    constant uint32_t& n [[buffer(3)]],
    constant uint32_t& k [[buffer(4)]],
    constant uint32_t& M_k [[buffer(5)]],
    constant uint32_t& M_k1 [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint32_t i = gid.x;
    uint32_t j = gid.y;
    if (i >= M_k || j >= M_k1) return;

    uint32_t state_k_size = n * k;
    uint32_t state_k1_size = n * (k + 1);
    device const uint8_t* state_k_ptr = states_k + i * state_k_size;
    device const uint8_t* state_k1_ptr = states_k1 + j * state_k1_size;

    bool valid = true;
    for (uint32_t color = 1; color <= n && valid; ++color) {
        valid = check_color_interlacing(state_k_ptr, state_k1_ptr, n, k, color);
    }
    results[i * M_k1 + j] = valid ? 1 : 0;
}
)";

MetalController::MetalController() : device(nullptr), commandQueue(nullptr), computePipeline(nullptr) {}

MetalController::~MetalController() {
    if (computePipeline) {
        CFRelease(computePipeline);
    }
    if (commandQueue) {
        CFRelease(commandQueue);
    }
    // device is autoreleased
}

bool MetalController::initialize() {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = MTLCreateSystemDefaultDevice();
        if (!mtlDevice) {
            std::cerr << "Metal is not supported on this device" << std::endl;
            return false;
        }
        device = (__bridge_retained void*)mtlDevice;

        id<MTLCommandQueue> mtlQueue = [mtlDevice newCommandQueue];
        if (!mtlQueue) {
            std::cerr << "Failed to create command queue" << std::endl;
            return false;
        }
        commandQueue = (__bridge_retained void*)mtlQueue;

        // Compile shader
        NSError* error = nil;
        id<MTLLibrary> library = [mtlDevice newLibraryWithSource:shaderSource
                                                         options:nil
                                                           error:&error];
        if (!library) {
            std::cerr << "Failed to compile Metal shader: "
                     << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"check_interlacing_kernel"];
        if (!kernelFunction) {
            std::cerr << "Failed to find kernel function" << std::endl;
            return false;
        }

        id<MTLComputePipelineState> pipeline = [mtlDevice newComputePipelineStateWithFunction:kernelFunction
                                                                                        error:&error];
        if (!pipeline) {
            std::cerr << "Failed to create compute pipeline: "
                     << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        computePipeline = (__bridge_retained void*)pipeline;

        return true;
    }
}

std::string MetalController::getDeviceName() const {
    if (!device) return "Not initialized";
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        return [[mtlDevice name] UTF8String];
    }
}

std::vector<uint8_t> MetalController::checkInterlacingBatch(
    int n, int k,
    const std::vector<uint8_t>& states_k,
    const std::vector<uint8_t>& states_k1,
    uint32_t M_k,
    uint32_t M_k1
) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> mtlQueue = (__bridge id<MTLCommandQueue>)commandQueue;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)computePipeline;

        // Create buffers
        id<MTLBuffer> bufferK = [mtlDevice newBufferWithBytes:states_k.data()
                                                       length:states_k.size()
                                                      options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferK1 = [mtlDevice newBufferWithBytes:states_k1.data()
                                                        length:states_k1.size()
                                                       options:MTLResourceStorageModeShared];

        size_t resultSize = (size_t)M_k * M_k1;
        id<MTLBuffer> bufferResults = [mtlDevice newBufferWithLength:resultSize
                                                             options:MTLResourceStorageModeShared];

        uint32_t params_n = n;
        uint32_t params_k = k;

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [mtlQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferK offset:0 atIndex:0];
        [encoder setBuffer:bufferK1 offset:0 atIndex:1];
        [encoder setBuffer:bufferResults offset:0 atIndex:2];
        [encoder setBytes:&params_n length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&params_k length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&M_k length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&M_k1 length:sizeof(uint32_t) atIndex:6];

        // Dispatch threads
        MTLSize gridSize = MTLSizeMake(M_k, M_k1, 1);
        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroupSize = MTLSizeMake(w, h, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy results
        std::vector<uint8_t> results(resultSize);
        memcpy(results.data(), [bufferResults contents], resultSize);

        return results;
    }
}
