#ifndef CUDA_NEXUS_REDUCTION_CUH
#define CUDA_NEXUS_REDUCTION_CUH

#include <cuda_runtime.h>
#include "cuda_nexus.h"

namespace cuda_nexus {
namespace kernels {

// Reduction operations
enum class ReductionOp {
    SUM,
    PROD,
    MAX,
    MIN,
    MEAN
};

// Reduce along a dimension
void reduce(
    const void* input,
    void* output,
    int* shape,
    int ndim,
    int reduce_dim,
    ReductionOp op,
    Precision precision,
    cudaStream_t stream = 0
);

// Fast sum reduction using warp shuffles
void warp_reduce_sum(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = 0
);

// Block-wide reduction with shared memory
void block_reduce_sum(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = 0
);

// Segmented reduction (reduce within segments)
void segmented_reduce(
    const void* input,
    const int* segment_ids,
    void* output,
    int num_segments,
    int size,
    ReductionOp op,
    Precision precision,
    cudaStream_t stream = 0
);

// Multi-dimensional reduction (reduce multiple axes)
void multi_reduce(
    const void* input,
    void* output,
    int* shape,
    int ndim,
    int* reduce_dims,
    int num_reduce_dims,
    ReductionOp op,
    Precision precision,
    cudaStream_t stream = 0
);

// Prefix sum (inclusive scan)
void prefix_sum_inclusive(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = 0
);

// Prefix sum (exclusive scan)
void prefix_sum_exclusive(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = 0
);

// Segmented prefix sum
void segmented_prefix_sum(
    const float* input,
    const int* segment_ids,
    float* output,
    int size,
    cudaStream_t stream = 0
);

// ArgMax reduction (returns indices)
void argmax(
    const void* input,
    int* output_indices,
    int* shape,
    int ndim,
    int reduce_dim,
    Precision precision,
    cudaStream_t stream = 0
);

// ArgMin reduction (returns indices)
void argmin(
    const void* input,
    int* output_indices,
    int* shape,
    int ndim,
    int reduce_dim,
    Precision precision,
    cudaStream_t stream = 0
);

// Top-K selection
void topk(
    const void* input,
    void* output_values,
    int* output_indices,
    int size,
    int k,
    bool largest,
    Precision precision,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace cuda_nexus

#endif // CUDA_NEXUS_REDUCTION_CUH
