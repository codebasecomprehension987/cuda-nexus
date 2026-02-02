#ifndef CUDA_NEXUS_H
#define CUDA_NEXUS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>

// Version information
#define CUDA_NEXUS_VERSION_MAJOR 1
#define CUDA_NEXUS_VERSION_MINOR 0
#define CUDA_NEXUS_VERSION_PATCH 0

// Common types
namespace cuda_nexus {

// Precision modes
enum class Precision {
    FP32,
    FP16,
    BF16,
    INT8,
    MIXED
};

// Kernel execution policies
enum class ExecutionPolicy {
    DEFAULT,
    PERSISTENT,
    COOPERATIVE,
    DYNAMIC_PARALLELISM
};

// Memory layout
enum class Layout {
    ROW_MAJOR,
    COLUMN_MAJOR,
    BLOCKED,
    STRIDED
};

// Error handling
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device properties cache
struct DeviceProperties {
    int sm_count;
    int warp_size;
    int max_threads_per_block;
    int max_shared_memory_per_block;
    int compute_capability_major;
    int compute_capability_minor;
    bool tensor_cores_available;
    
    static DeviceProperties get(int device = 0);
};

} // namespace cuda_nexus

// Include all kernel headers
#include "kernels/gemm.cuh"
#include "kernels/attention.cuh"
#include "kernels/reduction.cuh"
#include "kernels/normalization.cuh"
#include "kernels/activation.cuh"
#include "kernels/memory_ops.cuh"
#include "kernels/convolution.cuh"

// Include utility headers
#include "utils/memory_pool.h"
#include "utils/profiler.h"
#include "utils/async_ops.h"

#endif // CUDA_NEXUS_H
