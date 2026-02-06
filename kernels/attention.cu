#include "kernels/attention.cuh"
#include <cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;

namespace cuda_nexus {
namespace kernels {

// Fused attention kernel inspired by Flash Attention
// Uses tiling and on-chip softmax to reduce memory traffic
template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void fused_attention_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* output,
    int batch_size,
    int num_heads,
    int seq_length,
    float scale,
    bool causal
) {
    // Shared memory for tiles
    __shared__ half Q_tile[BLOCK_SIZE][HEAD_DIM];
    __shared__ half K_tile[BLOCK_SIZE][HEAD_DIM];
    __shared__ half V_tile[BLOCK_SIZE][HEAD_DIM];
    __shared__ float scores[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float max_vals[BLOCK_SIZE];
    __shared__ float sum_exp[BLOCK_SIZE];
    
    int batch_idx = blockIdx.z / num_heads;
    int head_idx = blockIdx.z % num_heads;
    int query_block = blockIdx.x;
    int tid = threadIdx.x;
    
    // Calculate base offset for this batch and head
    int base_offset = (batch_idx * num_heads + head_idx) * seq_length * HEAD_DIM;
    
    // Initialize output accumulator
    float output_accum[HEAD_DIM] = {0.0f};
    float max_score = -INFINITY;
    float exp_sum = 0.0f;
    
    // Load Q tile
    int q_idx = query_block * BLOCK_SIZE + tid;
    if (q_idx < seq_length && tid < BLOCK_SIZE) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            Q_tile[tid][d] = Q[base_offset + q_idx * HEAD_DIM + d];
        }
    }
    __syncthreads();
    
    // Process K,V tiles
    for (int key_block = 0; key_block < (seq_length + BLOCK_SIZE - 1) / BLOCK_SIZE; ++key_block) {
        int k_idx = key_block * BLOCK_SIZE + tid;
        
        // Load K and V tiles
        if (k_idx < seq_length && tid < BLOCK_SIZE) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                K_tile[tid][d] = K[base_offset + k_idx * HEAD_DIM + d];
                V_tile[tid][d] = V[base_offset + k_idx * HEAD_DIM + d];
            }
        }
        __syncthreads();
        
        // Compute attention scores for this tile
        if (tid < BLOCK_SIZE && q_idx < seq_length) {
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                int key_pos = key_block * BLOCK_SIZE + k;
                
                // Causal masking
                if (causal && key_pos > q_idx) {
                    scores[tid][k] = -INFINITY;
                    continue;
                }
                
                if (key_pos < seq_length) {
                    float score = 0.0f;
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        score += __half2float(Q_tile[tid][d]) * __half2float(K_tile[k][d]);
                    }
                    scores[tid][k] = score * scale;
                } else {
                    scores[tid][k] = -INFINITY;
                }
            }
        }
        __syncthreads();
        
        // Online softmax: update max and running sum
        if (tid < BLOCK_SIZE && q_idx < seq_length) {
            float local_max = max_score;
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                local_max = fmaxf(local_max, scores[tid][k]);
            }
            
            float exp_correction = expf(max_score - local_max);
            float new_exp_sum = exp_sum * exp_correction;
            
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                float exp_score = expf(scores[tid][k] - local_max);
                new_exp_sum += exp_score;
                scores[tid][k] = exp_score;
            }
            
            // Rescale previous output
            for (int d = 0; d < HEAD_DIM; ++d) {
                output_accum[d] *= exp_correction;
            }
            
            max_score = local_max;
            exp_sum = new_exp_sum;
        }
        __syncthreads();
        
        // Accumulate weighted values
        if (tid < BLOCK_SIZE && q_idx < seq_length) {
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                float weight = scores[tid][k];
                for (int d = 0; d < HEAD_DIM; ++d) {
                    output_accum[d] += weight * __half2float(V_tile[k][d]);
                }
            }
        }
        __syncthreads();
    }
    
    // Final normalization and write output
    if (tid < BLOCK_SIZE && q_idx < seq_length) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            output[base_offset + q_idx * HEAD_DIM + d] = __float2half(output_accum[d] / exp_sum);
        }
    }
}

// Standard attention kernel (non-fused version for reference)
__global__ void standard_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    float* attention_weights,
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim,
    float scale,
    bool causal
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_length) return;
    
    int base_offset = (batch_idx * num_heads + head_idx) * seq_length * head_dim;
    
    // Compute attention scores
    float max_score = -INFINITY;
    extern __shared__ float shared_scores[];
    
    for (int k_idx = 0; k_idx < seq_length; ++k_idx) {
        if (causal && k_idx > q_idx) {
            shared_scores[k_idx] = -INFINITY;
            continue;
        }
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += Q[base_offset + q_idx * head_dim + d] * 
                     K[base_offset + k_idx * head_dim + d];
        }
        shared_scores[k_idx] = score * scale;
        max_score = fmaxf(max_score, shared_scores[k_idx]);
    }
    
    // Softmax
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_length; ++k_idx) {
        shared_scores[k_idx] = expf(shared_scores[k_idx] - max_score);
        sum_exp += shared_scores[k_idx];
    }
    
    for (int k_idx = 0; k_idx < seq_length; ++k_idx) {
        shared_scores[k_idx] /= sum_exp;
    }
    
    // Compute output
    for (int d = 0; d < head_dim; ++d) {
        float out_val = 0.0f;
        for (int k_idx = 0; k_idx < seq_length; ++k_idx) {
            out_val += shared_scores[k_idx] * V[base_offset + k_idx * head_dim + d];
        }
        output[base_offset + q_idx * head_dim + d] = out_val;
    }
}

// Grouped query attention kernel
__global__ void gqa_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* output,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_length,
    int head_dim,
    float scale
) {
    int batch_idx = blockIdx.z;
    int q_head_idx = blockIdx.y;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_length) return;
    
    // Calculate which KV head this Q head should attend to
    int kv_head_idx = q_head_idx * num_kv_heads / num_q_heads;
    
    int q_offset = (batch_idx * num_q_heads + q_head_idx) * seq_length * head_dim;
    int kv_offset = (batch_idx * num_kv_heads + kv_head_idx) * seq_length * head_dim;
    
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    
    // Compute attention scores
    float max_score = -INFINITY;
    for (int k_idx = 0; k_idx < seq_length; ++k_idx) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += __half2float(Q[q_offset + q_idx * head_dim + d]) * 
                     __half2float(K[kv_offset + k_idx * head_dim + d]);
        }
        scores[k_idx] = score * scale;
        max_score = fmaxf(max_score, scores[k_idx]);
    }
    
    // Softmax
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_length; ++k_idx) {
        scores[k_idx] = expf(scores[k_idx] - max_score);
        sum_exp += scores[k_idx];
    }
    
    // Normalize and compute output
    for (int d = 0; d < head_dim; ++d) {
        float out_val = 0.0f;
        for (int k_idx = 0; k_idx < seq_length; ++k_idx) {
            float weight = scores[k_idx] / sum_exp;
            out_val += weight * __half2float(V[kv_offset + k_idx * head_dim + d]);
        }
        output[q_offset + q_idx * head_dim + d] = __float2half(out_val);
    }
}

// Host function implementations
void fused_multi_head_attention(
    const void* Q,
    const void* K,
    const void* V,
    void* output,
    const AttentionConfig& config,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int HEAD_DIM = 64;  // Fixed for now, could be templated
    
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(
        (config.seq_length + BLOCK_SIZE - 1) / BLOCK_SIZE,
        1,
        config.batch_size * config.num_heads
    );
    
    if (config.precision == Precision::FP16) {
        fused_attention_kernel<BLOCK_SIZE, HEAD_DIM><<<gridDim, blockDim, 0, stream>>>(
            static_cast<const half*>(Q),
            static_cast<const half*>(K),
            static_cast<const half*>(V),
            static_cast<half*>(output),
            config.batch_size,
            config.num_heads,
            config.seq_length,
            config.scale,
            config.causal
        );
    }
}

void grouped_query_attention(
    const void* Q,
    const void* K,
    const void* V,
    void* output,
    int num_q_heads,
    int num_kv_heads,
    const AttentionConfig& config,
    cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim(
        (config.seq_length + 255) / 256,
        num_q_heads,
        config.batch_size
    );
    
    size_t shared_mem_size = config.seq_length * sizeof(float);
    
    gqa_kernel<<<gridDim, blockDim, shared_mem_size, stream>>>(
        static_cast<const half*>(Q),
        static_cast<const half*>(K),
        static_cast<const half*>(V),
        static_cast<half*>(output),
        config.batch_size,
        num_q_heads,
        num_kv_heads,
        config.seq_length,
        config.head_dim,
        config.scale
    );
}

void masked_attention(
    const void* Q,
    const void* K,
    const void* V,
    const bool* mask,
    void* output,
    const AttentionConfig& config,
    cudaStream_t stream
) {
    // Implementation similar to fused_multi_head_attention but with mask
    // Omitted for brevity
}

void attention_with_kv_cache(
    const void* Q,
    const void* K_cache,
    const void* V_cache,
    void* output,
    const AttentionConfig& config,
    cudaStream_t stream
) {
    // Implementation for incremental decoding with KV cache
    // Omitted for brevity
}

void attention_backward(
    const void* grad_output,
    const void* Q,
    const void* K,
    const void* V,
    const void* attention_weights,
    void* grad_Q,
    void* grad_K,
    void* grad_V,
    const AttentionConfig& config,
    cudaStream_t stream
) {
    // Backward pass implementation
    // Omitted for brevity
}

} // namespace kernels
} // namespace cuda_nexus
