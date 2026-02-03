#ifndef CUDA_NEXUS_GEMM_CUH
#define CUDA_NEXUS_GEMM_CUH

#include <cuda_runtime.h>
#include <mma.h>
#include "cuda_nexus.h"

namespace cuda_nexus {
namespace kernels {

// GEMM configuration
struct GEMMConfig {
    int M, N, K;
    int batch_size = 1;
    Precision precision = Precision::FP32;
    Layout layout_A = Layout::ROW_MAJOR;
    Layout layout_B = Layout::ROW_MAJOR;
    bool use_tensor_cores = true;
    float alpha = 1.0f;
    float beta = 0.0f;
};

// Standard GEMM: C = alpha * A @ B + beta * C
void gemm(
    const void* A,
    const void* B,
    void* C,
    const GEMMConfig& config,
    cudaStream_t stream = 0
);

// Batched GEMM
void gemm_batched(
    const void* const* A,
    const void* const* B,
    void** C,
    const GEMMConfig& config,
    cudaStream_t stream = 0
);

// Strided batched GEMM
void gemm_strided_batched(
    const void* A,
    const void* B,
    void* C,
    int64_t stride_A,
    int64_t stride_B,
    int64_t stride_C,
    const GEMMConfig& config,
    cudaStream_t stream = 0
);

// Fused GEMM with activation
void gemm_fused_activation(
    const void* A,
    const void* B,
    void* C,
    const void* bias,
    const GEMMConfig& config,
    const char* activation, // "relu", "gelu", "silu"
    cudaStream_t stream = 0
);

// Tensor Core WMMA-based GEMM (FP16)
void gemm_wmma_fp16(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    cudaStream_t stream = 0
);

// Advanced persistent kernel GEMM
void gemm_persistent(
    const void* A,
    const void* B,
    void* C,
    const GEMMConfig& config,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace cuda_nexus

#endif // CUDA_NEXUS_GEMM_CUH
