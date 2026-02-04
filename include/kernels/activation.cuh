#ifndef CUDA_NEXUS_ACTIVATION_CUH
#define CUDA_NEXUS_ACTIVATION_CUH

#include <cuda_runtime.h>
#include "cuda_nexus.h"

namespace cuda_nexus {
namespace kernels {

// ReLU activation
void relu(
    const void* input,
    void* output,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// Leaky ReLU
void leaky_relu(
    const void* input,
    void* output,
    int size,
    float negative_slope,
    Precision precision,
    cudaStream_t stream = 0
);

// GELU activation (Gaussian Error Linear Unit)
void gelu(
    const void* input,
    void* output,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// Approximate GELU (faster version)
void gelu_approx(
    const void* input,
    void* output,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// SiLU/Swish activation (x * sigmoid(x))
void silu(
    const void* input,
    void* output,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// Mish activation (x * tanh(softplus(x)))
void mish(
    const void* input,
    void* output,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// ELU activation (Exponential Linear Unit)
void elu(
    const void* input,
    void* output,
    int size,
    float alpha,
    Precision precision,
    cudaStream_t stream = 0
);

// Sigmoid activation
void sigmoid(
    const void* input,
    void* output,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// Tanh activation
void tanh_activation(
    const void* input,
    void* output,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// Softmax (stable version)
void softmax(
    const void* input,
    void* output,
    int batch_size,
    int num_classes,
    Precision precision,
    cudaStream_t stream = 0
);

// Log softmax
void log_softmax(
    const void* input,
    void* output,
    int batch_size,
    int num_classes,
    Precision precision,
    cudaStream_t stream = 0
);

// GLU (Gated Linear Unit)
void glu(
    const void* input,
    void* output,
    int batch_size,
    int hidden_size,
    Precision precision,
    cudaStream_t stream = 0
);

// Squared ReLU
void squared_relu(
    const void* input,
    void* output,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// Fused activation + dropout
void activation_dropout(
    const void* input,
    void* output,
    unsigned char* dropout_mask,
    int size,
    float dropout_prob,
    const char* activation_type,
    unsigned long long seed,
    Precision precision,
    cudaStream_t stream = 0
);

// Backward pass for ReLU
void relu_backward(
    const void* grad_output,
    const void* input,
    void* grad_input,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// Backward pass for GELU
void gelu_backward(
    const void* grad_output,
    const void* input,
    void* grad_input,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

// Backward pass for SiLU
void silu_backward(
    const void* grad_output,
    const void* input,
    void* grad_input,
    int size,
    Precision precision,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace cuda_nexus

#endif // CUDA_NEXUS_ACTIVATION_CUH
