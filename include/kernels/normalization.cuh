#ifndef CUDA_NEXUS_NORMALIZATION_CUH
#define CUDA_NEXUS_NORMALIZATION_CUH

#include <cuda_runtime.h>
#include "cuda_nexus.h"

namespace cuda_nexus {
namespace kernels {

// Layer normalization with affine transform
// output = gamma * (input - mean) / sqrt(variance + eps) + beta
void layer_norm(
    const void* input,
    const void* gamma,
    const void* beta,
    void* output,
    int batch_size,
    int hidden_size,
    float epsilon,
    Precision precision,
    cudaStream_t stream = 0
);

// RMS normalization (no mean subtraction)
void rms_norm(
    const void* input,
    const void* gamma,
    void* output,
    int batch_size,
    int hidden_size,
    float epsilon,
    Precision precision,
    cudaStream_t stream = 0
);

// Group normalization
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
    cudaStream_t stream = 0
);

// Batch normalization (inference mode)
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
    cudaStream_t stream = 0
);

// Batch normalization (training mode)
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
    cudaStream_t stream = 0
);

// Instance normalization
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
    cudaStream_t stream = 0
);

// Fused layer norm + dropout
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
    cudaStream_t stream = 0
);

// Backward pass for layer normalization
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
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace cuda_nexus

#endif // CUDA_NEXUS_NORMALIZATION_CUH
