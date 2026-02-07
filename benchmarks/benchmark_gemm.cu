#include "cuda_nexus.h"
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace cuda_nexus;

struct BenchmarkResult {
    std::string name;
    float time_ms;
    double gflops;
    double speedup;
};

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "                         GEMM BENCHMARK RESULTS                                 \n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(30) << "Implementation"
              << std::right << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS"
              << std::setw(15) << "Speedup" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(30) << result.name
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3) << result.time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.gflops
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.speedup << "x"
                  << "\n";
    }
    std::cout << "================================================================================\n";
    std::cout << "\n";
}

float benchmark_cublas_gemm(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int num_iterations
) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Warm-up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    B, N,
                    A, K,
                    &beta,
                    C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    
    return milliseconds / num_iterations;
}

float benchmark_cuda_nexus_gemm(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int num_iterations
) {
    kernels::GEMMConfig config;
    config.M = M;
    config.N = N;
    config.K = K;
    config.precision = Precision::FP32;
    config.alpha = 1.0f;
    config.beta = 0.0f;
    
    // Warm-up
    kernels::gemm(A, B, C, config, 0);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        kernels::gemm(A, B, C, config, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / num_iterations;
}

float benchmark_cuda_nexus_persistent(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int batch_size,
    int num_iterations
) {
    kernels::GEMMConfig config;
    config.M = M;
    config.N = N;
    config.K = K;
    config.batch_size = batch_size;
    config.precision = Precision::FP32;
    config.alpha = 1.0f;
    config.beta = 0.0f;
    
    // Warm-up
    kernels::gemm_persistent(A, B, C, config, 0);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        kernels::gemm_persistent(A, B, C, config, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / num_iterations;
}

void run_gemm_benchmark(int M, int N, int K, int num_iterations = 100) {
    std::cout << "\n";
    std::cout << "Matrix dimensions: A(" << M << "x" << K << ") x B(" 
              << K << "x" << N << ") = C(" << M << "x" << N << ")\n";
    std::cout << "Iterations: " << num_iterations << "\n";
    
    // Allocate memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Initialize with random data
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : h_A) val = dis(gen);
    for (auto& val : h_B) val = dis(gen);
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate theoretical FLOPs
    double flops = 2.0 * M * N * K;
    
    std::vector<BenchmarkResult> results;
    
    // Benchmark cuBLAS
    float cublas_time = benchmark_cublas_gemm(d_A, d_B, d_C, M, N, K, num_iterations);
    double cublas_gflops = flops / (cublas_time * 1e6);
    results.push_back({"cuBLAS", cublas_time, cublas_gflops, 1.0});
    
    // Benchmark CUDA Nexus
    float nexus_time = benchmark_cuda_nexus_gemm(d_A, d_B, d_C, M, N, K, num_iterations);
    double nexus_gflops = flops / (nexus_time * 1e6);
    double speedup = cublas_time / nexus_time;
    results.push_back({"CUDA Nexus (Standard)", nexus_time, nexus_gflops, speedup});
    
    // Benchmark CUDA Nexus Persistent (batched)
    float persistent_time = benchmark_cuda_nexus_persistent(d_A, d_B, d_C, M, N, K, 4, num_iterations);
    double persistent_gflops = flops / (persistent_time * 1e6);
    double persistent_speedup = cublas_time / persistent_time;
    results.push_back({"CUDA Nexus (Persistent)", persistent_time, persistent_gflops, persistent_speedup});
    
    print_results(results);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv) {
    std::cout << "================================================================================\n";
    std::cout << "                    CUDA NEXUS vs cuBLAS GEMM BENCHMARK                        \n";
    std::cout << "================================================================================\n";
    
    // Print device info
    device::print_device_info(0);
    
    // Run benchmarks for different sizes
    std::vector<std::tuple<int, int, int>> sizes = {
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192}
    };
    
    for (const auto& [M, N, K] : sizes) {
        run_gemm_benchmark(M, N, K);
    }
    
    std::cout << "\nBenchmark completed!\n";
    
    return 0;
}
