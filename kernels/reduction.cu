#include "kernels/reduction.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace cuda_nexus {
namespace kernels {

// Warp-level reduction using shuffle operations
__device__ __forceinline__ float warp_reduce_sum_device(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
template<int BLOCK_SIZE>
__device__ float block_reduce_sum_device(float val) {
    __shared__ float shared[32];  // One per warp
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // Warp-level reduction
    val = warp_reduce_sum_device(val);
    
    // Write reduced value from each warp to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // First warp reduces the warp results
    if (wid == 0) {
        val = (threadIdx.x < (BLOCK_SIZE / warpSize)) ? shared[lane] : 0.0f;
        val = warp_reduce_sum_device(val);
    }
    
    return val;
}

// Warp reduction kernel
__global__ void warp_reduce_sum_kernel(
    const float* input,
    float* output,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    
    float sum = 0.0f;
    
    // Grid-stride loop
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum_device(sum);
    
    // First thread in each warp writes to output
    if (lane == 0) {
        atomicAdd(output, sum);
    }
}

// Block reduction kernel
template<int BLOCK_SIZE>
__global__ void block_reduce_sum_kernel(
    const float* input,
    float* output,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Grid-stride loop
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // Block-level reduction
    sum = block_reduce_sum_device<BLOCK_SIZE>(sum);
    
    // First thread in block writes to output
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// Segmented reduction kernel
__global__ void segmented_reduce_kernel(
    const float* input,
    const int* segment_ids,
    float* output,
    int num_segments,
    int size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= size) return;
    
    int seg_id = segment_ids[tid];
    float value = input[tid];
    
    // Atomic add to segment
    atomicAdd(&output[seg_id], value);
}

// Prefix sum (scan) kernel - Blelloch algorithm
__global__ void prefix_sum_kernel(
    const float* input,
    float* output,
    int size,
    bool exclusive
) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory
    if (tid < size) {
        temp[2*tid] = input[2*tid];
        temp[2*tid+1] = input[2*tid+1];
    } else {
        temp[2*tid] = 0;
        temp[2*tid+1] = 0;
    }
    
    // Build sum tree
    for (int d = size >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if (tid == 0) {
        temp[size - 1] = 0;
    }
    
    // Traverse down tree & build scan
    for (int d = 1; d < size; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    // Write results
    if (tid < size) {
        if (exclusive) {
            output[2*tid] = temp[2*tid];
            output[2*tid+1] = temp[2*tid+1];
        } else {
            output[2*tid] = temp[2*tid] + input[2*tid];
            output[2*tid+1] = temp[2*tid+1] + input[2*tid+1];
        }
    }
}

// ArgMax kernel
__global__ void argmax_kernel(
    const float* input,
    int* output_indices,
    int outer_size,
    int reduce_size,
    int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    float max_val = -INFINITY;
    int max_idx = 0;
    
    for (int i = 0; i < reduce_size; ++i) {
        int input_idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        float val = input[input_idx];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    int output_idx = outer_idx * inner_size + inner_idx;
    output_indices[output_idx] = max_idx;
}

// Top-K selection kernel using heap-based approach
__global__ void topk_kernel(
    const float* input,
    float* output_values,
    int* output_indices,
    int size,
    int k,
    bool largest
) {
    // Simple implementation - would use more efficient algorithm in production
    extern __shared__ char shared_mem[];
    float* heap_vals = (float*)shared_mem;
    int* heap_idxs = (int*)(shared_mem + k * sizeof(float));
    
    int tid = threadIdx.x;
    
    // Initialize heap
    if (tid < k) {
        heap_vals[tid] = largest ? -INFINITY : INFINITY;
        heap_idxs[tid] = -1;
    }
    __syncthreads();
    
    // Process elements
    for (int i = tid; i < size; i += blockDim.x) {
        float val = input[i];
        
        // Check if this element should be in top-k
        bool should_insert = largest ? (val > heap_vals[0]) : (val < heap_vals[0]);
        
        if (should_insert) {
            // Insert into heap (simplified - use proper heap operations)
            atomicExch((int*)&heap_vals[0], __float_as_int(val));
            atomicExch(&heap_idxs[0], i);
        }
    }
    __syncthreads();
    
    // Write output
    if (tid < k) {
        output_values[tid] = heap_vals[tid];
        output_indices[tid] = heap_idxs[tid];
    }
}

// Host function implementations
void warp_reduce_sum(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    // Initialize output to zero
    CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));
    
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    warp_reduce_sum_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, size
    );
}

void block_reduce_sum(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    // Initialize output to zero
    CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float), stream));
    
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    block_reduce_sum_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        input, output, size
    );
}

void segmented_reduce(
    const void* input,
    const int* segment_ids,
    void* output,
    int num_segments,
    int size,
    ReductionOp op,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32 && op == ReductionOp::SUM) {
        // Initialize output to zero
        CUDA_CHECK(cudaMemsetAsync(output, 0, num_segments * sizeof(float), stream));
        
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        
        segmented_reduce_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            segment_ids,
            static_cast<float*>(output),
            num_segments,
            size
        );
    }
}

void prefix_sum_inclusive(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    size_t shared_mem_size = size * sizeof(float);
    
    prefix_sum_kernel<<<1, block_size, shared_mem_size, stream>>>(
        input, output, size, false
    );
}

void prefix_sum_exclusive(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    size_t shared_mem_size = size * sizeof(float);
    
    prefix_sum_kernel<<<1, block_size, shared_mem_size, stream>>>(
        input, output, size, true
    );
}

void argmax(
    const void* input,
    int* output_indices,
    int* shape,
    int ndim,
    int reduce_dim,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int outer_size = 1;
        for (int i = 0; i < reduce_dim; ++i) {
            outer_size *= shape[i];
        }
        
        int reduce_size = shape[reduce_dim];
        
        int inner_size = 1;
        for (int i = reduce_dim + 1; i < ndim; ++i) {
            inner_size *= shape[i];
        }
        
        int block_size = 256;
        int grid_size = (outer_size * inner_size + block_size - 1) / block_size;
        
        argmax_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            output_indices,
            outer_size,
            reduce_size,
            inner_size
        );
    }
}

void topk(
    const void* input,
    void* output_values,
    int* output_indices,
    int size,
    int k,
    bool largest,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        size_t shared_mem_size = k * (sizeof(float) + sizeof(int));
        
        topk_kernel<<<1, 256, shared_mem_size, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output_values),
            output_indices,
            size,
            k,
            largest
        );
    }
}

void reduce(
    const void* input,
    void* output,
    int* shape,
    int ndim,
    int reduce_dim,
    ReductionOp op,
    Precision precision,
    cudaStream_t stream
) {
    // General reduction - delegates to specific implementations
    if (op == ReductionOp::SUM) {
        // Use appropriate sum reduction based on dimensions
        int total_size = 1;
        for (int i = 0; i < ndim; ++i) {
            total_size *= shape[i];
        }
        block_reduce_sum(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            total_size,
            stream
        );
    }
}

void multi_reduce(
    const void* input,
    void* output,
    int* shape,
    int ndim,
    int* reduce_dims,
    int num_reduce_dims,
    ReductionOp op,
    Precision precision,
    cudaStream_t stream
) {
    // Multi-dimensional reduction
    // Would iterate over reduce_dims and call single-dim reduction
}

void argmin(
    const void* input,
    int* output_indices,
    int* shape,
    int ndim,
    int reduce_dim,
    Precision precision,
    cudaStream_t stream
) {
    // Similar to argmax but finding minimum
}

void segmented_prefix_sum(
    const float* input,
    const int* segment_ids,
    float* output,
    int size,
    cudaStream_t stream
) {
    // Segmented scan implementation
}

} // namespace kernels
} // namespace cuda_nexus
