#include "kernels/gemm.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace cuda_nexus {
namespace kernels {

// WMMA tile sizes for Tensor Cores
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Tensor Core GEMM kernel using WMMA API
__global__ void gemm_wmma_kernel(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    float alpha, float beta
) {
    using namespace nvcuda::wmma;
    
    // Calculate warp and tile indices
    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warp_n = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Declare the fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    
    // Initialize accumulator to zero
    fill_fragment(acc_frag, 0.0f);
    
    // Perform matrix multiplication
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warp_m * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warp_n * WMMA_N;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrices from global memory
            load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform matrix multiplication
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // Load C matrix and scale
    int cRow = warp_m * WMMA_M;
    int cCol = warp_n * WMMA_N;
    
    if (cRow < M && cCol < N) {
        if (beta != 0.0f) {
            load_matrix_sync(c_frag, C + cRow * N + cCol, N, mem_row_major);
            
            // Scale and accumulate: C = alpha * acc + beta * C
            for (int i = 0; i < c_frag.num_elements; i++) {
                c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
            }
        } else {
            // Just scale accumulator: C = alpha * acc
            for (int i = 0; i < acc_frag.num_elements; i++) {
                c_frag.x[i] = alpha * acc_frag.x[i];
            }
        }
        
        // Store the output
        store_matrix_sync(C + cRow * N + cCol, c_frag, N, mem_row_major);
    }
}

// Standard FP32 GEMM kernel with shared memory tiling
template<int TILE_SIZE>
__global__ void gemm_fp32_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        if (beta != 0.0f) {
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * sum;
        }
    }
}

// Persistent kernel for GEMM (keeps threads alive across multiple problems)
__global__ void gemm_persistent_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta,
    int num_problems
) {
    // Grid-persistent loop
    for (int problem_id = blockIdx.z; problem_id < num_problems; problem_id += gridDim.z) {
        // Offset pointers for current problem
        const float* A_cur = A + problem_id * M * K;
        const float* B_cur = B + problem_id * K * N;
        float* C_cur = C + problem_id * M * N;
        
        // Standard GEMM computation
        constexpr int TILE_SIZE = 32;
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];
        
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * TILE_SIZE + ty;
        int col = blockIdx.x * TILE_SIZE + tx;
        
        float sum = 0.0f;
        
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
            if (row < M && t * TILE_SIZE + tx < K) {
                As[ty][tx] = A_cur[row * K + t * TILE_SIZE + tx];
            } else {
                As[ty][tx] = 0.0f;
            }
            
            if (col < N && t * TILE_SIZE + ty < K) {
                Bs[ty][tx] = B_cur[(t * TILE_SIZE + ty) * N + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += As[ty][k] * Bs[k][tx];
            }
            
            __syncthreads();
        }
        
        if (row < M && col < N) {
            if (beta != 0.0f) {
                C_cur[row * N + col] = alpha * sum + beta * C_cur[row * N + col];
            } else {
                C_cur[row * N + col] = alpha * sum;
            }
        }
    }
}

// Host function implementations
void gemm_wmma_fp16(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 gridDim((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 blockDim(32, 8);
    
    gemm_wmma_kernel<<<gridDim, blockDim, 0, stream>>>(
        A, B, C, M, N, K, 1.0f, 0.0f
    );
}

void gemm(
    const void* A,
    const void* B,
    void* C,
    const GEMMConfig& config,
    cudaStream_t stream
) {
    if (config.precision == Precision::FP16 && config.use_tensor_cores) {
        gemm_wmma_fp16(
            static_cast<const half*>(A),
            static_cast<const half*>(B),
            static_cast<half*>(C),
            config.M, config.N, config.K,
            stream
        );
    } else if (config.precision == Precision::FP32) {
        constexpr int TILE_SIZE = 32;
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim(
            (config.N + TILE_SIZE - 1) / TILE_SIZE,
            (config.M + TILE_SIZE - 1) / TILE_SIZE
        );
        
        gemm_fp32_kernel<TILE_SIZE><<<gridDim, blockDim, 0, stream>>>(
            static_cast<const float*>(A),
            static_cast<const float*>(B),
            static_cast<float*>(C),
            config.M, config.N, config.K,
            config.alpha, config.beta
        );
    }
}

void gemm_persistent(
    const void* A,
    const void* B,
    void* C,
    const GEMMConfig& config,
    cudaStream_t stream
) {
    constexpr int TILE_SIZE = 32;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (config.N + TILE_SIZE - 1) / TILE_SIZE,
        (config.M + TILE_SIZE - 1) / TILE_SIZE,
        4  // Multiple problems per grid
    );
    
    gemm_persistent_kernel<<<gridDim, blockDim, 0, stream>>>(
        static_cast<const float*>(A),
        static_cast<const float*>(B),
        static_cast<float*>(C),
        config.M, config.N, config.K,
        config.alpha, config.beta,
        config.batch_size
    );
}

void gemm_batched(
    const void* const* A,
    const void* const* B,
    void** C,
    const GEMMConfig& config,
    cudaStream_t stream
) {
    // Launch batched kernels (simplified implementation)
    for (int i = 0; i < config.batch_size; ++i) {
        GEMMConfig single_config = config;
        single_config.batch_size = 1;
        gemm(A[i], B[i], C[i], single_config, stream);
    }
}

void gemm_strided_batched(
    const void* A,
    const void* B,
    void* C,
    int64_t stride_A,
    int64_t stride_B,
    int64_t stride_C,
    const GEMMConfig& config,
    cudaStream_t stream
) {
    // Launch strided batched kernels
    for (int i = 0; i < config.batch_size; ++i) {
        GEMMConfig single_config = config;
        single_config.batch_size = 1;
        
        const char* A_ptr = static_cast<const char*>(A) + i * stride_A;
        const char* B_ptr = static_cast<const char*>(B) + i * stride_B;
        char* C_ptr = static_cast<char*>(C) + i * stride_C;
        
        gemm(A_ptr, B_ptr, C_ptr, single_config, stream);
    }
}

void gemm_fused_activation(
    const void* A,
    const void* B,
    void* C,
    const void* bias,
    const GEMMConfig& config,
    const char* activation,
    cudaStream_t stream
) {
    // First do GEMM
    gemm(A, B, C, config, stream);
    
    // Then apply activation (simplified - would be fused in real implementation)
    // This is where you'd call the activation kernel
}

} // namespace kernels
} // namespace cuda_nexus
