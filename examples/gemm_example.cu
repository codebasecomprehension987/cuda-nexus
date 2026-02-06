#include "cuda_nexus.h"
#include <iostream>
#include <vector>
#include <random>

using namespace cuda_nexus;

void print_matrix(const std::vector<float>& mat, int rows, int cols, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < std::min(rows, 4); ++i) {
        for (int j = 0; j < std::min(cols, 4); ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        if (cols > 4) std::cout << "...";
        std::cout << "\n";
    }
    if (rows > 4) std::cout << "...\n";
    std::cout << "\n";
}

int main() {
    // Matrix dimensions
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    std::cout << "CUDA Nexus - GEMM Example\n";
    std::cout << "Matrix dimensions: A(" << M << "x" << K << ") x B(" 
              << K << "x" << N << ") = C(" << M << "x" << N << ")\n\n";
    
    // Initialize random matrices on host
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    
    std::cout << "Initializing random matrices...\n";
    for (auto& val : h_A) val = dis(gen);
    for (auto& val : h_B) val = dis(gen);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy data to device
    std::cout << "Copying data to device...\n";
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure GEMM
    kernels::GEMMConfig config;
    config.M = M;
    config.N = N;
    config.K = K;
    config.precision = Precision::FP32;
    config.use_tensor_cores = false;  // FP32 doesn't use tensor cores
    config.alpha = 1.0f;
    config.beta = 0.0f;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warm-up run
    std::cout << "Warming up...\n";
    kernels::gemm(d_A, d_B, d_C, config, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    std::cout << "Running GEMM...\n";
    CUDA_CHECK(cudaEventRecord(start));
    kernels::gemm(d_A, d_B, d_C, config, 0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Calculate performance
    double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "  Execution time: " << milliseconds << " ms\n";
    std::cout << "  Performance: " << gflops << " GFLOPS\n\n";
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print small portion of result
    print_matrix(h_C, M, N, "Result C (first 4x4)");
    
    // Verify result (simple check)
    bool correct = true;
    const float epsilon = 1e-3f;
    int num_checks = std::min(100, M * N);
    
    std::cout << "Verifying results...\n";
    for (int i = 0; i < num_checks; ++i) {
        int row = i / N;
        int col = i % N;
        
        float expected = 0.0f;
        for (int k = 0; k < K; ++k) {
            expected += h_A[row * K + k] * h_B[k * N + col];
        }
        
        float diff = std::abs(h_C[i] - expected);
        if (diff > epsilon) {
            std::cout << "Mismatch at (" << row << "," << col << "): "
                      << "expected " << expected << ", got " << h_C[i] << "\n";
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ Results verified successfully!\n";
    } else {
        std::cout << "✗ Verification failed!\n";
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << "\nExample completed successfully!\n";
    
    return 0;
}
