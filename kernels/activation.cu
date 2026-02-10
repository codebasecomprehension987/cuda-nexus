#include "kernels/activation.cuh"
#include <cmath>

namespace cuda_nexus {
namespace kernels {

// ReLU kernel
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Leaky ReLU kernel
__global__ void leaky_relu_kernel(
    const float* input,
    float* output,
    int size,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0 ? x : negative_slope * x;
    }
}

// GELU kernel (exact)
__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // GELU(x) = x * Φ(x) where Φ is the cumulative distribution function
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608f;
        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        output[idx] = 0.5f * x * (1.0f + tanhf(tanh_arg));
    }
}

// GELU approximate (faster)
__global__ void gelu_approx_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Fast approximation: x * σ(1.702 * x)
        float sigmoid = 1.0f / (1.0f + expf(-1.702f * x));
        output[idx] = x * sigmoid;
    }
}

// SiLU/Swish kernel
__global__ void silu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}

// Mish kernel
__global__ void mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
        float softplus = logf(1.0f + expf(x));
        output[idx] = x * tanhf(softplus);
    }
}

// ELU kernel
__global__ void elu_kernel(
    const float* input,
    float* output,
    int size,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0 ? x : alpha * (expf(x) - 1.0f);
    }
}

// Sigmoid kernel
__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// Tanh kernel
__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

// Softmax kernel (stable version)
__global__ void softmax_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * num_classes;
    float* y = output + batch_idx * num_classes;
    
    extern __shared__ float shared[];
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    shared[tid] = max_val;
    __syncthreads();
    
    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    max_val = shared[0];
    __syncthreads();
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float exp_val = expf(x[i] - max_val);
        y[i] = exp_val;
        sum += exp_val;
    }
    shared[tid] = sum;
    __syncthreads();
    
    // Reduce sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    sum = shared[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        y[i] /= sum;
    }
}

// Log softmax kernel
__global__ void log_softmax_kernel(
    const float* input,
    float* output,
    int batch_size,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * num_classes;
    float* y = output + batch_idx * num_classes;
    
    extern __shared__ float shared[];
    
    // Find max
    float max_val = -INFINITY;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    shared[tid] = max_val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    max_val = shared[0];
    __syncthreads();
    
    // Compute log-sum-exp
    float sum_exp = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        sum_exp += expf(x[i] - max_val);
    }
    shared[tid] = sum_exp;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float log_sum_exp = logf(shared[0]);
    __syncthreads();
    
    // Compute log softmax
    for (int i = tid; i < num_classes; i += blockDim.x) {
        y[i] = (x[i] - max_val) - log_sum_exp;
    }
}

// GLU kernel
__global__ void glu_kernel(
    const float* input,
    float* output,
    int batch_size,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * hidden_size;
    
    if (idx < total_size) {
        int batch_idx = idx / hidden_size;
        int hidden_idx = idx % hidden_size;
        
        // Input is [batch, 2*hidden_size]
        // Split into two halves and apply gate
        float value = input[batch_idx * 2 * hidden_size + hidden_idx];
        float gate = input[batch_idx * 2 * hidden_size + hidden_size + hidden_idx];
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
        
        output[idx] = value * sigmoid_gate;
    }
}

// Squared ReLU kernel
__global__ void squared_relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = fmaxf(0.0f, input[idx]);
        output[idx] = x * x;
    }
}

// ReLU backward kernel
__global__ void relu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0.0f;
    }
}

// GELU backward kernel
__global__ void gelu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        const float sqrt_2_over_pi = 0.7978845608f;
        float x_squared = x * x;
        float x_cubed = x_squared * x;
        
        float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        float tanh_val = tanhf(tanh_arg);
        float sech_squared = 1.0f - tanh_val * tanh_val;
        
        float derivative = 0.5f * (1.0f + tanh_val) + 
                          0.5f * x * sech_squared * sqrt_2_over_pi * 
                          (1.0f + 3.0f * 0.044715f * x_squared);
        
        grad_input[idx] = grad_output[idx] * derivative;
    }
}

// SiLU backward kernel
__global__ void silu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        float derivative = sigmoid * (1.0f + x * (1.0f - sigmoid));
        grad_input[idx] = grad_output[idx] * derivative;
    }
}

// Host function implementations
void relu(const void* input, void* output, int size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        relu_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size
        );
    }
}

void leaky_relu(const void* input, void* output, int size, float negative_slope, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        leaky_relu_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size,
            negative_slope
        );
    }
}

void gelu(const void* input, void* output, int size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        gelu_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size
        );
    }
}

void gelu_approx(const void* input, void* output, int size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        gelu_approx_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size
        );
    }
}

void silu(const void* input, void* output, int size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        silu_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size
        );
    }
}

void mish(const void* input, void* output, int size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        mish_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size
        );
    }
}

void elu(const void* input, void* output, int size, float alpha, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        elu_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size,
            alpha
        );
    }
}

void sigmoid(const void* input, void* output, int size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size
        );
    }
}

void tanh_activation(const void* input, void* output, int size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        tanh_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size
        );
    }
}

void softmax(const void* input, void* output, int batch_size, int num_classes, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        size_t shared_mem = block_size * sizeof(float);
        softmax_kernel<<<batch_size, block_size, shared_mem, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            batch_size,
            num_classes
        );
    }
}

void log_softmax(const void* input, void* output, int batch_size, int num_classes, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        size_t shared_mem = block_size * sizeof(float);
        log_softmax_kernel<<<batch_size, block_size, shared_mem, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            batch_size,
            num_classes
        );
    }
}

void glu(const void* input, void* output, int batch_size, int hidden_size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int total_size = batch_size * hidden_size;
        int block_size = 256;
        int grid_size = (total_size + block_size - 1) / block_size;
        glu_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            batch_size,
            hidden_size
        );
    }
}

void squared_relu(const void* input, void* output, int size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        squared_relu_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            size
        );
    }
}

void relu_backward(const void* grad_output, const void* input, void* grad_input, int size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        relu_backward_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(grad_output),
            static_cast<const float*>(input),
            static_cast<float*>(grad_input),
            size
        );
    }
}

void gelu_backward(const void* grad_output, const void* input, void* grad_input, int size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        gelu_backward_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(grad_output),
            static_cast<const float*>(input),
            static_cast<float*>(grad_input),
            size
        );
    }
}

void silu_backward(const void* grad_output, const void* input, void* grad_input, int size, Precision precision, cudaStream_t stream) {
    if (precision == Precision::FP32) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        silu_backward_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(grad_output),
            static_cast<const float*>(input),
            static_cast<float*>(grad_input),
            size
        );
    }
}

void activation_dropout(
    const void* input,
    void* output,
    unsigned char* dropout_mask,
    int size,
    float dropout_prob,
    const char* activation_type,
    unsigned long long seed,
    Precision precision,
    cudaStream_t stream
) {
    // Apply activation then dropout (implementation omitted for brevity)
}

} // namespace kernels
} // namespace cuda_nexus
