#include "cuda_nexus.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace cuda_nexus;

int main() {
    // Attention configuration
    const int batch_size = 2;
    const int num_heads = 8;
    const int seq_length = 512;
    const int head_dim = 64;
    
    std::cout << "CUDA Nexus - Fused Multi-Head Attention Example\n";
    std::cout << "Configuration:\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Number of heads: " << num_heads << "\n";
    std::cout << "  Sequence length: " << seq_length << "\n";
    std::cout << "  Head dimension: " << head_dim << "\n\n";
    
    const int qkv_size = batch_size * num_heads * seq_length * head_dim;
    
    // Initialize random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    std::vector<half> h_Q(qkv_size);
    std::vector<half> h_K(qkv_size);
    std::vector<half> h_V(qkv_size);
    std::vector<half> h_output(qkv_size);
    
    std::cout << "Initializing random Q, K, V matrices...\n";
    for (int i = 0; i < qkv_size; ++i) {
        h_Q[i] = __float2half(dis(gen) * 0.1f);
        h_K[i] = __float2half(dis(gen) * 0.1f);
        h_V[i] = __float2half(dis(gen) * 0.1f);
    }
    
    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_output;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, qkv_size * sizeof(half)));
    
    // Copy data to device
    std::cout << "Copying data to device...\n";
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Configure attention
    kernels::AttentionConfig config;
    config.batch_size = batch_size;
    config.num_heads = num_heads;
    config.seq_length = seq_length;
    config.head_dim = head_dim;
    config.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    config.causal = false;
    config.precision = Precision::FP16;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warm-up run
    std::cout << "Warming up...\n";
    kernels::fused_multi_head_attention(d_Q, d_K, d_V, d_output, config, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    std::cout << "Running fused attention...\n";
    CUDA_CHECK(cudaEventRecord(start));
    
    const int num_iterations = 10;
    for (int i = 0; i < num_iterations; ++i) {
        kernels::fused_multi_head_attention(d_Q, d_K, d_V, d_output, config, 0);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / num_iterations;
    
    // Calculate performance metrics
    // FLOPs for attention: 4 * batch * heads * seq^2 * head_dim
    // (QK^T, softmax, weighted sum)
    double flops = 4.0 * batch_size * num_heads * seq_length * seq_length * head_dim;
    double tflops = flops / (avg_time * 1e9);
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "  Average execution time: " << avg_time << " ms\n";
    std::cout << "  Throughput: " << (batch_size * seq_length) / avg_time << " K tokens/sec\n";
    std::cout << "  Performance: " << tflops << " TFLOPS\n\n";
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, qkv_size * sizeof(half), 
                          cudaMemcpyDeviceToHost));
    
    // Print sample output
    std::cout << "Sample output values (first head, first sequence position):\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "  " << __half2float(h_output[i]) << "\n";
    }
    std::cout << "  ...\n\n";
    
    // Basic sanity check
    bool has_valid_output = false;
    for (int i = 0; i < std::min(1000, qkv_size); ++i) {
        float val = __half2float(h_output[i]);
        if (!std::isnan(val) && !std::isinf(val) && std::abs(val) < 10.0f) {
            has_valid_output = true;
            break;
        }
    }
    
    if (has_valid_output) {
        std::cout << "✓ Output values look reasonable!\n";
    } else {
        std::cout << "✗ Warning: Output values may be incorrect!\n";
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "\nExample completed successfully!\n";
    
    return 0;
}
