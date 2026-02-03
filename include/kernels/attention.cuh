#ifndef CUDA_NEXUS_ATTENTION_CUH
#define CUDA_NEXUS_ATTENTION_CUH

#include <cuda_runtime.h>
#include "cuda_nexus.h"

namespace cuda_nexus {
namespace kernels {

// Attention configuration
struct AttentionConfig {
    int batch_size;
    int num_heads;
    int seq_length;
    int head_dim;
    float scale;
    bool causal = false;
    Precision precision = Precision::FP16;
};

// Fused multi-head attention (Flash Attention inspired)
// Output = softmax(Q @ K^T / sqrt(d)) @ V
void fused_multi_head_attention(
    const void* Q,              // [batch, num_heads, seq_len, head_dim]
    const void* K,              // [batch, num_heads, seq_len, head_dim]
    const void* V,              // [batch, num_heads, seq_len, head_dim]
    void* output,               // [batch, num_heads, seq_len, head_dim]
    const AttentionConfig& config,
    cudaStream_t stream = 0
);

// Attention with mask
void masked_attention(
    const void* Q,
    const void* K,
    const void* V,
    const bool* mask,           // [batch, seq_len, seq_len]
    void* output,
    const AttentionConfig& config,
    cudaStream_t stream = 0
);

// Attention forward pass with KV cache
void attention_with_kv_cache(
    const void* Q,              // [batch, num_heads, 1, head_dim] (new token)
    const void* K_cache,        // [batch, num_heads, cache_len, head_dim]
    const void* V_cache,        // [batch, num_heads, cache_len, head_dim]
    void* output,
    const AttentionConfig& config,
    cudaStream_t stream = 0
);

// Grouped query attention (GQA)
void grouped_query_attention(
    const void* Q,              // [batch, num_q_heads, seq_len, head_dim]
    const void* K,              // [batch, num_kv_heads, seq_len, head_dim]
    const void* V,              // [batch, num_kv_heads, seq_len, head_dim]
    void* output,
    int num_q_heads,
    int num_kv_heads,
    const AttentionConfig& config,
    cudaStream_t stream = 0
);

// Attention backward pass
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
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace cuda_nexus

#endif // CUDA_NEXUS_ATTENTION_CUH
