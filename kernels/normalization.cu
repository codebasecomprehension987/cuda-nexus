#include "kernels/normalization.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace cuda_nexus {
namespace kernels {

// Layer normalization kernel
__global__ void layer_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int hidden_size,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    const float* x = input + idx * hidden_size;
    float* y = output + idx * hidden_size;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        sum += x[i];
    }
    float mean = sum / hidden_size;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / hidden_size;
    float inv_std = rsqrtf(variance + epsilon);
    
    // Normalize and apply affine transform
    for (int i = 0; i < hidden_size; ++i) {
        float normalized = (x[i] - mean) * inv_std;
        y[i] = gamma[i] * normalized + beta[i];
    }
}

// Optimized layer norm with shared memory
template<int BLOCK_SIZE>
__global__ void layer_norm_optimized_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int hidden_size,
    float epsilon
) {
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_var[BLOCK_SIZE];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * hidden_size;
    float* y = output + batch_idx * hidden_size;
    
    // Parallel reduction for mean
    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        thread_sum += x[i];
    }
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Reduce sum
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / hidden_size;
    __syncthreads();
    
    // Parallel reduction for variance
    float thread_var = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float diff = x[i] - mean;
        thread_var += diff * diff;
    }
    shared_var[tid] = thread_var;
    __syncthreads();
    
    // Reduce variance
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_var[tid] += shared_var[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_var[0] / hidden_size;
    float inv_std = rsqrtf(variance + epsilon);
    
    // Normalize and apply affine
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float normalized = (x[i] - mean) * inv_std;
        y[i] = gamma[i] * normalized + beta[i];
    }
}

// RMS normalization kernel
__global__ void rms_norm_kernel(
    const float* input,
    const float* gamma,
    float* output,
    int batch_size,
    int hidden_size,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    const float* x = input + idx * hidden_size;
    float* y = output + idx * hidden_size;
    
    // Compute RMS
    float sum_squares = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        sum_squares += x[i] * x[i];
    }
    float rms = sqrtf(sum_squares / hidden_size + epsilon);
    float inv_rms = 1.0f / rms;
    
    // Normalize and scale
    for (int i = 0; i < hidden_size; ++i) {
        y[i] = gamma[i] * x[i] * inv_rms;
    }
}

// Batch normalization inference kernel
__global__ void batch_norm_inference_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int spatial_size,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx >= total_size) return;
    
    int c = (idx / spatial_size) % channels;
    
    float mean = running_mean[c];
    float var = running_var[c];
    float inv_std = rsqrtf(var + epsilon);
    
    float normalized = (input[idx] - mean) * inv_std;
    output[idx] = gamma[c] * normalized + beta[c];
}

// Batch normalization training kernel
__global__ void batch_norm_training_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* save_mean,
    float* save_inv_var,
    float* running_mean,
    float* running_var,
    int batch_size,
    int channels,
    int spatial_size,
    float momentum,
    float epsilon
) {
    int c = blockIdx.x;
    int tid = threadIdx.x;
    
    if (c >= channels) return;
    
    extern __shared__ float shared[];
    float* shared_sum = shared;
    float* shared_var = shared + blockDim.x;
    
    int N = batch_size * spatial_size;
    
    // Compute mean
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        int batch_idx = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int input_idx = batch_idx * channels * spatial_size + c * spatial_size + spatial_idx;
        thread_sum += input[input_idx];
    }
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / N;
    __syncthreads();
    
    // Compute variance
    float thread_var = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        int batch_idx = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int input_idx = batch_idx * channels * spatial_size + c * spatial_size + spatial_idx;
        float diff = input[input_idx] - mean;
        thread_var += diff * diff;
    }
    shared_var[tid] = thread_var;
    __syncthreads();
    
    // Reduce variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_var[tid] += shared_var[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_var[0] / N;
    float inv_std = rsqrtf(variance + epsilon);
    
    // Save statistics
    if (tid == 0) {
        save_mean[c] = mean;
        save_inv_var[c] = inv_std;
        
        // Update running statistics
        running_mean[c] = momentum * running_mean[c] + (1 - momentum) * mean;
        running_var[c] = momentum * running_var[c] + (1 - momentum) * variance;
    }
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        int batch_idx = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int input_idx = batch_idx * channels * spatial_size + c * spatial_size + spatial_idx;
        
        float normalized = (input[input_idx] - mean) * inv_std;
        output[input_idx] = gamma[c] * normalized + beta[c];
    }
}

// Host function implementations
void layer_norm(
    const void* input,
    const void* gamma,
    const void* beta,
    void* output,
    int batch_size,
    int hidden_size,
    float epsilon,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        constexpr int BLOCK_SIZE = 256;
        int grid_size = batch_size;
        
        layer_norm_optimized_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(gamma),
            static_cast<const float*>(beta),
            static_cast<float*>(output),
            batch_size,
            hidden_size,
            epsilon
        );
    }
}

void rms_norm(
    const void* input,
    const void* gamma,
    void* output,
    int batch_size,
    int hidden_size,
    float epsilon,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (batch_size + block_size - 1) / block_size;
        
        rms_norm_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(gamma),
            static_cast<float*>(output),
            batch_size,
            hidden_size,
            epsilon
        );
    }
}

void batch_norm_inference(
    const void* input,
    const void* gamma,
    const void* beta,
    const void* running_mean,
    const void* running_var,
    void* output,
    int batch_size,
    int channels,
    int spatial_size,
    float epsilon,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int total_size = batch_size * channels * spatial_size;
        int block_size = 256;
        int grid_size = (total_size + block_size - 1) / block_size;
        
        batch_norm_inference_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(gamma),
            static_cast<const float*>(beta),
            static_cast<const float*>(running_mean),
            static_cast<const float*>(running_var),
            static_cast<float*>(output),
            batch_size,
            channels,
            spatial_size,
            epsilon
        );
    }
}

void batch_norm_training(
    const void* input,
    const void* gamma,
    const void* beta,
    void* output,
    void* save_mean,
    void* save_inv_var,
    void* running_mean,
    void* running_var,
    int batch_size,
    int channels,
    int spatial_size,
    float momentum,
    float epsilon,
    Precision precision,
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        size_t shared_mem_size = 2 * block_size * sizeof(float);
        
        batch_norm_training_kernel<<<channels, block_size, shared_mem_size, stream>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(gamma),
            static_cast<const float*>(beta),
            static_cast<float*>(output),
            static_cast<float*>(save_mean),
            static_cast<float*>(save_inv_var),
            static_cast<float*>(running_mean),
            static_cast<float*>(running_var),
            batch_size,
            channels,
            spatial_size,
            momentum,
            epsilon
        );
    }
}

void group_norm(
    const void* input,
    const void* gamma,
    const void* beta,
    void* output,
    int batch_size,
    int num_groups,
    int channels,
    int spatial_size,
    float epsilon,
    Precision precision,
    cudaStream_t stream
) {
    // Implementation similar to batch norm but with groups
    // Omitted for brevity
}

void instance_norm(
    const void* input,
    const void* gamma,
    const void* beta,
    void* output,
    int batch_size,
    int channels,
    int spatial_size,
    float epsilon,
    Precision precision,
    cudaStream_t stream
) {
    // Implementation similar to batch norm but per instance
    // Omitted for brevity
}

void layer_norm_dropout(
    const void* input,
    const void* gamma,
    const void* beta,
    void* output,
    unsigned char* dropout_mask,
    int batch_size,
    int hidden_size,
    float epsilon,
    float dropout_prob,
    unsigned long long seed,
    Precision precision,
    cudaStream_t stream
) {
    // First apply layer norm
    layer_norm(input, gamma, beta, output, batch_size, hidden_size, epsilon, precision, stream);
    
    // Then apply dropout (implementation omitted)
}

void layer_norm_backward(
    const void* grad_output,
    const void* input,
    const void* gamma,
    const void* mean,
    const void* inv_std,
    void* grad_input,
    void* grad_gamma,
    void* grad_beta,
    int batch_size,
    int hidden_size,
    Precision precision,
    cudaStream_t stream
) {
    // Backward pass implementation
    // Omitted for brevity
}

} // namespace kernels
} // namespace cuda_nexus
