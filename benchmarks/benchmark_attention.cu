#include "cuda_nexus.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

using namespace cuda_nexus;

struct AttentionBenchmarkResult {
    std::string name;
    float time_ms;
    double tokens_per_sec;
    double speedup;
};

void print_attention_results(const std::vector<AttentionBenchmarkResult>& results) {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "                      ATTENTION BENCHMARK RESULTS                               \n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(35) << "Implementation"
              << std::right << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Tokens/sec"
              << std::setw(15) << "Speedup" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(35) << result.name
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3) << result.time_ms
                  << std::setw(15) << std::fixed << std::setprecision(0) << result.tokens_per_sec
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.speedup << "x"
                  << "\n";
    }
    std::cout << "================================================================================\n";
    std::cout << "\n";
}

// Reference attention implementation (non-fused)
__global__ void reference_attention_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* output,
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim,
    float scale
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_length) return;
    
    int base_offset = (batch_idx * num_heads + head_idx) * seq_length * head_dim;
    
    extern __shared__ float scores[];
    
    // Compute attention scores
    float max_score = -INFINITY;
    for (int k_idx = 0; k_idx < seq_length; ++k_idx) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += __half2float(Q[base_offset + q_idx * head_dim + d]) * 
                     __half2float(K[base_offset + k_idx * head_dim + d]);
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
    
    for (int k_idx = 0; k_idx < seq_length; ++k_idx) {
        scores[k_idx] /= sum_exp;
    }
    
    // Compute output
    for (int d = 0; d < head_dim; ++d) {
        float out_val = 0.0f;
        for (int k_idx = 0; k_idx < seq_length; ++k_idx) {
            out_val += scores[k_idx] * __half2float(V[base_offset + k_idx * head_dim + d]);
        }
        output[base_offset + q_idx * head_dim + d] = __float2half(out_val);
    }
}

float benchmark_reference_attention(
    const half* Q,
    const half* K,
    const half* V,
    half* output,
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim,
    int num_iterations
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    dim3 blockDim(256);
    dim3 gridDim(
        (seq_length + 255) / 256,
        num_heads,
        batch_size
    );
    size_t shared_mem = seq_length * sizeof(float);
    
    // Warm-up
    reference_attention_kernel<<<gridDim, blockDim, shared_mem>>>(
        Q, K, V, output, batch_size, num_heads, seq_length, head_dim, scale
    );
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        reference_attention_kernel<<<gridDim, blockDim, shared_mem>>>(
            Q, K, V, output, batch_size, num_heads, seq_length, head_dim, scale
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / num_iterations;
}

float benchmark_fused_attention(
    const half* Q,
    const half* K,
    const half* V,
    half* output,
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim,
    int num_iterations
) {
    kernels::AttentionConfig config;
    config.batch_size = batch_size;
    config.num_heads = num_heads;
    config.seq_length = seq_length;
    config.head_dim = head_dim;
    config.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    config.causal = false;
    config.precision = Precision::FP16;
    
    // Warm-up
    kernels::fused_multi_head_attention(Q, K, V, output, config, 0);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        kernels::fused_multi_head_attention(Q, K, V, output, config, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / num_iterations;
}

void run_attention_benchmark(
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim,
    int num_iterations = 100
) {
    std::cout << "\n";
    std::cout << "Configuration:\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Heads: " << num_heads << "\n";
    std::cout << "  Sequence length: " << seq_length << "\n";
    std::cout << "  Head dimension: " << head_dim << "\n";
    std::cout << "  Iterations: " << num_iterations << "\n";
    
    int qkv_size = batch_size * num_heads * seq_length * head_dim;
    
    // Allocate memory
    half *d_Q, *d_K, *d_V, *d_output;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, qkv_size * sizeof(half)));
    
    // Initialize with random data
    std::vector<half> h_Q(qkv_size);
    std::vector<half> h_K(qkv_size);
    std::vector<half> h_V(qkv_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 0.1f);
    
    for (int i = 0; i < qkv_size; ++i) {
        h_Q[i] = __float2half(dis(gen));
        h_K[i] = __float2half(dis(gen));
        h_V[i] = __float2half(dis(gen));
    }
    
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice));
    
    std::vector<AttentionBenchmarkResult> results;
    
    // Benchmark reference implementation
    float ref_time = benchmark_reference_attention(
        d_Q, d_K, d_V, d_output,
        batch_size, num_heads, seq_length, head_dim,
        num_iterations
    );
    double ref_tokens_per_sec = (batch_size * seq_length * 1000.0) / ref_time;
    results.push_back({"Reference (Non-fused)", ref_time, ref_tokens_per_sec, 1.0});
    
    // Benchmark fused implementation
    float fused_time = benchmark_fused_attention(
        d_Q, d_K, d_V, d_output,
        batch_size, num_heads, seq_length, head_dim,
        num_iterations
    );
    double fused_tokens_per_sec = (batch_size * seq_length * 1000.0) / fused_time;
    double speedup = ref_time / fused_time;
    results.push_back({"CUDA Nexus (Fused)", fused_time, fused_tokens_per_sec, speedup});
    
    print_attention_results(results);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_output));
}

int main(int argc, char** argv) {
    std::cout << "================================================================================\n";
    std::cout << "                      ATTENTION MECHANISM BENCHMARK                             \n";
    std::cout << "================================================================================\n";
    
    // Print device info
    device::print_device_info(0);
    
    // Run benchmarks for different configurations
    std::vector<std::tuple<int, int, int, int>> configs = {
        {1, 8, 128, 64},    // Small: 1 batch, 8 heads, 128 seq, 64 dim
        {2, 8, 256, 64},    // Medium: 2 batch, 8 heads, 256 seq, 64 dim
        {4, 12, 512, 64},   // Large: 4 batch, 12 heads, 512 seq, 64 dim
        {8, 16, 1024, 64},  // XLarge: 8 batch, 16 heads, 1024 seq, 64 dim
    };
    
    for (const auto& [batch, heads, seq, dim] : configs) {
        run_attention_benchmark(batch, heads, seq, dim);
    }
    
    std::cout << "\nBenchmark completed!\n";
    
    return 0;
}
