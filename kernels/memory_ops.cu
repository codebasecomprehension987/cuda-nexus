#include "kernels/memory_ops.cuh"

namespace cuda_nexus {
namespace kernels {

// Vectorized copy kernel using float4
__global__ void vectorized_copy_kernel(
    const float* src,
    float* dst,
    size_t num_elements
) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < num_elements) {
        float4* src4 = (float4*)src;
        float4* dst4 = (float4*)dst;
        dst4[idx/4] = src4[idx/4];
    } else if (idx < num_elements) {
        // Handle remaining elements
        for (size_t i = idx; i < num_elements && i < idx + 4; ++i) {
            dst[i] = src[i];
        }
    }
}

// 2D transpose kernel with shared memory
template<int TILE_SIZE>
__global__ void transpose_2d_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load tile from input
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();
    
    // Write transposed tile to output
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Gather kernel
__global__ void gather_kernel(
    const float* input,
    const int* indices,
    float* output,
    int num_indices,
    int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices * inner_size) {
        int index_id = idx / inner_size;
        int inner_id = idx % inner_size;
        
        int src_index = indices[index_id];
        int src_offset = src_index * inner_size + inner_id;
        
        output[idx] = input[src_offset];
    }
}

// Scatter kernel
__global__ void scatter_kernel(
    const float* input,
    const int* indices,
    float* output,
    int num_indices,
    int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices * inner_size) {
        int index_id = idx / inner_size;
        int inner_id = idx % inner_size;
        
        int dst_index = indices[index_id];
        int dst_offset = dst_index * inner_size + inner_id;
        
        output[dst_offset] = input[idx];
    }
}

// Scatter-add kernel
__global__ void scatter_add_kernel(
    const float* input,
    const int* indices,
    float* output,
    int num_indices,
    int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices * inner_size) {
        int index_id = idx / inner_size;
        int inner_id = idx % inner_size;
        
        int dst_index = indices[index_id];
        int dst_offset = dst_index * inner_size + inner_id;
        
        atomicAdd(&output[dst_offset], input[idx]);
    }
}

// Masked fill kernel
__global__ void masked_fill_kernel(
    const float* input,
    const bool* mask,
    float* output,
    float fill_value,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = mask[idx] ? fill_value : input[idx];
    }
}

// Embedding lookup kernel
__global__ void embedding_lookup_kernel(
    const float* embeddings,
    const int* indices,
    float* output,
    int vocab_size,
    int embedding_dim,
    int num_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices * embedding_dim) {
        int index_id = idx / embedding_dim;
        int dim_id = idx % embedding_dim;
        
        int word_index = indices[index_id];
        if (word_index >= 0 && word_index < vocab_size) {
            output[idx] = embeddings[word_index * embedding_dim + dim_id];
        } else {
            output[idx] = 0.0f;  // Out of bounds
        }
    }
}

// One-hot encoding kernel
__global__ void one_hot_kernel(
    const int* indices,
    float* output,
    int num_indices,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        int class_id = indices[idx];
        
        // Zero out the row
        for (int c = 0; c < num_classes; ++c) {
            output[idx * num_classes + c] = 0.0f;
        }
        
        // Set the correct class to 1
        if (class_id >= 0 && class_id < num_classes) {
            output[idx * num_classes + class_id] = 1.0f;
        }
    }
}

// Concatenate kernel (simplified for single dimension)
__global__ void concatenate_kernel(
    const float** inputs,
    int num_inputs,
    float* output,
    int* input_sizes,
    int* cumulative_sizes,
    int outer_size,
    int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inner = 0;
    for (int i = 0; i < num_inputs; ++i) {
        total_inner += input_sizes[i];
    }
    
    if (idx < outer_size * total_inner * inner_size) {
        int outer_idx = idx / (total_inner * inner_size);
        int remainder = idx % (total_inner * inner_size);
        int concat_idx = remainder / inner_size;
        int inner_idx = remainder % inner_size;
        
        // Find which input this belongs to
        int input_id = 0;
        for (int i = 0; i < num_inputs; ++i) {
            if (concat_idx < cumulative_sizes[i]) {
                input_id = i;
                break;
            }
        }
        
        int local_concat_idx = concat_idx - (input_id > 0 ? cumulative_sizes[input_id - 1] : 0);
        int src_idx = outer_idx * input_sizes[input_id] * inner_size + 
                      local_concat_idx * inner_size + inner_idx;
        
        output[idx] = inputs[input_id][src_idx];
    }
}

// Host function implementations
void vectorized_copy(
    const void* src,
    void* dst,
    size_t num_elements,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = ((num_elements + 3) / 4 + block_size - 1) / block_size;
        
        vectorized_copy_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(src),
            static_cast<float*>(dst),
            num_elements
        );
    }
}

void transpose_2d(
    const void* input,
    void* output,
    int rows,
    int cols,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        constexpr int TILE_SIZE = 32;
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim(
            (cols + TILE_SIZE - 1) / TILE_SIZE,
            (rows + TILE_SIZE - 1) / TILE_SIZE
        );
        
        transpose_2d_kernel<TILE_SIZE><<<gridDim, blockDim, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            rows,
            cols
        );
    }
}

void transpose_batched(
    const void* input,
    void* output,
    int batch_size,
    int rows,
    int cols,
    Precision precision,
    cudaStream_t stream
) {
    // Launch transpose for each batch
    for (int b = 0; b < batch_size; ++b) {
        const char* input_ptr = static_cast<const char*>(input) + b * rows * cols * sizeof(float);
        char* output_ptr = static_cast<char*>(output) + b * rows * cols * sizeof(float);
        transpose_2d(input_ptr, output_ptr, rows, cols, precision, stream);
    }
}

void gather(
    const void* input,
    const int* indices,
    void* output,
    int num_indices,
    int inner_size,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int total_size = num_indices * inner_size;
        int block_size = 256;
        int grid_size = (total_size + block_size - 1) / block_size;
        
        gather_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            indices,
            static_cast<float*>(output),
            num_indices,
            inner_size
        );
    }
}

void scatter(
    const void* input,
    const int* indices,
    void* output,
    int num_indices,
    int inner_size,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int total_size = num_indices * inner_size;
        int block_size = 256;
        int grid_size = (total_size + block_size - 1) / block_size;
        
        scatter_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            indices,
            static_cast<float*>(output),
            num_indices,
            inner_size
        );
    }
}

void scatter_add(
    const void* input,
    const int* indices,
    void* output,
    int num_indices,
    int inner_size,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int total_size = num_indices * inner_size;
        int block_size = 256;
        int grid_size = (total_size + block_size - 1) / block_size;
        
        scatter_add_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            indices,
            static_cast<float*>(output),
            num_indices,
            inner_size
        );
    }
}

void masked_fill(
    const void* input,
    const bool* mask,
    void* output,
    float fill_value,
    int size,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        
        masked_fill_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            mask,
            static_cast<float*>(output),
            fill_value,
            size
        );
    }
}

void embedding_lookup(
    const void* embeddings,
    const int* indices,
    void* output,
    int vocab_size,
    int embedding_dim,
    int num_indices,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int total_size = num_indices * embedding_dim;
        int block_size = 256;
        int grid_size = (total_size + block_size - 1) / block_size;
        
        embedding_lookup_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(embeddings),
            indices,
            static_cast<float*>(output),
            vocab_size,
            embedding_dim,
            num_indices
        );
    }
}

void one_hot(
    const int* indices,
    void* output,
    int num_indices,
    int num_classes,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (num_indices + block_size - 1) / block_size;
        
        one_hot_kernel<<<grid_size, block_size, 0, stream>>>(
            indices,
            static_cast<float*>(output),
            num_indices,
            num_classes
        );
    }
}

void permute(
    const void* input,
    void* output,
    int* input_shape,
    int* output_strides,
    int* permutation,
    int ndim,
    Precision precision,
    cudaStream_t stream
) {
    // Complex permutation - implementation omitted for brevity
}

void index_select(
    const void* input,
    const int* indices,
    void* output,
    int input_size,
    int num_indices,
    int dim_size,
    Precision precision,
    cudaStream_t stream
) {
    // Index select - similar to gather
    gather(input, indices, output, num_indices, dim_size, precision, stream);
}

void masked_select(
    const void* input,
    const bool* mask,
    void* output,
    int* output_size,
    int size,
    Precision precision,
    cudaStream_t stream
) {
    // Masked select - implementation omitted for brevity
    // Would require stream compaction
}

void concatenate(
    const void** inputs,
    int num_inputs,
    void* output,
    int* input_sizes,
    int concat_dim,
    int outer_size,
    int inner_size,
    Precision precision,
    cudaStream_t stream
) {
    // Simplified concatenate - full implementation omitted
}

void split(
    const void* input,
    void** outputs,
    int num_outputs,
    int* output_sizes,
    int split_dim,
    int outer_size,
    int inner_size,
    Precision precision,
    cudaStream_t stream
) {
    // Split operation - reverse of concatenate
}

void repeat(
    const void* input,
    void* output,
    int* input_shape,
    int* repeats,
    int ndim,
    Precision precision,
    cudaStream_t stream
) {
    // Repeat/tile operation
}

void reshape_view(
    int* old_shape,
    int* new_shape,
    int old_ndim,
    int new_ndim,
    int* output_strides
) {
    // Compute new strides for reshape
    int stride = 1;
    for (int i = new_ndim - 1; i >= 0; --i) {
        output_strides[i] = stride;
        stride *= new_shape[i];
    }
}

} // namespace kernels
} // namespace cuda_nexus
