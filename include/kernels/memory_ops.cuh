#ifndef CUDA_NEXUS_MEMORY_OPS_CUH
#define CUDA_NEXUS_MEMORY_OPS_CUH

#include <cuda_runtime.h>
#include "cuda_nexus.h"

namespace cuda_nexus {
namespace kernels {

// Vectorized memory copy
void vectorized_copy(
    const void* src,
    void* dst,
    size_t num_elements,
    Precision precision,
    cudaStream_t stream = 0
);

// Transpose 2D matrix
void transpose_2d(
    const void* input,
    void* output,
    int rows,
    int cols,
    Precision precision,
    cudaStream_t stream = 0
);

// Batched transpose
void transpose_batched(
    const void* input,
    void* output,
    int batch_size,
    int rows,
    int cols,
    Precision precision,
    cudaStream_t stream = 0
);

// Permute dimensions (generalized transpose)
void permute(
    const void* input,
    void* output,
    int* input_shape,
    int* output_strides,
    int* permutation,
    int ndim,
    Precision precision,
    cudaStream_t stream = 0
);

// Gather operation
void gather(
    const void* input,
    const int* indices,
    void* output,
    int num_indices,
    int inner_size,
    Precision precision,
    cudaStream_t stream = 0
);

// Scatter operation
void scatter(
    const void* input,
    const int* indices,
    void* output,
    int num_indices,
    int inner_size,
    Precision precision,
    cudaStream_t stream = 0
);

// Scatter-add operation
void scatter_add(
    const void* input,
    const int* indices,
    void* output,
    int num_indices,
    int inner_size,
    Precision precision,
    cudaStream_t stream = 0
);

// Index select
void index_select(
    const void* input,
    const int* indices,
    void* output,
    int input_size,
    int num_indices,
    int dim_size,
    Precision precision,
    cudaStream_t stream = 0
);

// Masked fill
void masked_fill(
    const void* input,
    const bool* mask,
    void* output,
    float fill_value,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// Masked select
void masked_select(
    const void* input,
    const bool* mask,
    void* output,
    int* output_size,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// Concatenate along dimension
void concatenate(
    const void** inputs,
    int num_inputs,
    void* output,
    int* input_sizes,
    int concat_dim,
    int outer_size,
    int inner_size,
    Precision precision,
    cudaStream_t stream = 0
);

// Split along dimension
void split(
    const void* input,
    void** outputs,
    int num_outputs,
    int* output_sizes,
    int split_dim,
    int outer_size,
    int inner_size,
    Precision precision,
    cudaStream_t stream = 0
);

// Repeat/tile operation
void repeat(
    const void* input,
    void* output,
    int* input_shape,
    int* repeats,
    int ndim,
    Precision precision,
    cudaStream_t stream = 0
);

// Embedding lookup
void embedding_lookup(
    const void* embeddings,
    const int* indices,
    void* output,
    int vocab_size,
    int embedding_dim,
    int num_indices,
    Precision precision,
    cudaStream_t stream = 0
);

// One-hot encoding
void one_hot(
    const int* indices,
    void* output,
    int num_indices,
    int num_classes,
    Precision precision,
    cudaStream_t stream = 0
);

// Reshape (view) - just metadata change, no kernel needed
// But we provide a utility function
void reshape_view(
    int* old_shape,
    int* new_shape,
    int old_ndim,
    int new_ndim,
    int* output_strides
);

} // namespace kernels
} // namespace cuda_nexus

#endif // CUDA_NEXUS_MEMORY_OPS_CUH
