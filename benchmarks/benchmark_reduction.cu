#include "cuda_nexus.h"
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace cuda_nexus;

struct ReductionBenchmarkResult {
    std::string name;
    float time_ms;
    double bandwidth_gbps;
    double speedup;
};

void print_reduction_results(const std::vector<ReductionBenchmarkResult>& results) {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "                      REDUCTION BENCHMARK RESULTS                               \n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(35) << "Implementation"
              << std::right << std::setw(15) << "Time (ms)"
              << std::setw(15) << "BW (GB/s)"
              << std::setw(15) << "Speedup" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(35) << result.name
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3) << result.time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.bandwidth_gbps
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.speedup << "x"
                  << "\n";
    }
    std::cout << "================================================================================\n";
    std::cout << "\n";
}

float benchmark_thrust_reduce(
    const float* d_input,
    float* d_output,
    int size,
    int num_iterations
) {
    thrust::device_ptr<const float> input_ptr(d_input);
    
    // Warm-up
    float result = thrust::reduce(input_ptr, input_ptr + size, 0.0f, thrust::plus<float>());
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        result = thrust::reduce(input_ptr, input_ptr + size, 0.0f, thrust::plus<float>());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / num_iterations;
}

float benchmark_warp_reduce(
    const float* d_input,
    float* d_output,
    int size,
    int num_iterations
) {
    // Warm-up
    kernels::warp_reduce_sum(d_input, d_output, size, 0);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        kernels::warp_reduce_sum(d_input, d_output, size, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / num_iterations;
}

float benchmark_block_reduce(
    const float* d_input,
    float* d_output,
    int size,
    int num_iterations
) {
    // Warm-up
    kernels::block_reduce_sum(d_input, d_output, size, 0);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        kernels::block_reduce_sum(d_input, d_output, size, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / num_iterations;
}

void run_reduction_benchmark(int size, int num_iterations = 1000) {
    std::cout << "\n";
    std::cout << "Array size: " << size << " elements (" 
              << (size * sizeof(float)) / (1024.0 * 1024.0) << " MB)\n";
    std::cout << "Iterations: " << num_iterations << "\n";
    
    // Allocate memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    
    // Initialize with random data
    std::vector<float> h_input(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (auto& val : h_input) val = dis(gen);
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate bandwidth (reading input once)
    double bytes = size * sizeof(float);
    
    std::vector<ReductionBenchmarkResult> results;
    
    // Benchmark Thrust
    float thrust_time = benchmark_thrust_reduce(d_input, d_output, size, num_iterations);
    double thrust_bw = (bytes / 1e9) / (thrust_time / 1000.0);
    results.push_back({"Thrust::reduce", thrust_time, thrust_bw, 1.0});
    
    // Benchmark Warp Reduce
    float warp_time = benchmark_warp_reduce(d_input, d_output, size, num_iterations);
    double warp_bw = (bytes / 1e9) / (warp_time / 1000.0);
    double warp_speedup = thrust_time / warp_time;
    results.push_back({"CUDA Nexus (Warp)", warp_time, warp_bw, warp_speedup});
    
    // Benchmark Block Reduce
    float block_time = benchmark_block_reduce(d_input, d_output, size, num_iterations);
    double block_bw = (bytes / 1e9) / (block_time / 1000.0);
    double block_speedup = thrust_time / block_time;
    results.push_back({"CUDA Nexus (Block)", block_time, block_bw, block_speedup});
    
    print_reduction_results(results);
    
    // Verify correctness
    float h_output;
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    float cpu_sum = 0.0f;
    for (float val : h_input) cpu_sum += val;
    
    float error = std::abs(h_output - cpu_sum) / cpu_sum;
    if (error < 1e-4f) {
        std::cout << "✓ Result verified (error: " << error << ")\n";
    } else {
        std::cout << "✗ Verification failed! Error: " << error << "\n";
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Prefix sum benchmark
float benchmark_thrust_scan(
    const float* d_input,
    float* d_output,
    int size,
    int num_iterations
) {
    thrust::device_ptr<const float> input_ptr(d_input);
    thrust::device_ptr<float> output_ptr(d_output);
    
    // Warm-up
    thrust::inclusive_scan(input_ptr, input_ptr + size, output_ptr);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        thrust::inclusive_scan(input_ptr, input_ptr + size, output_ptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / num_iterations;
}

float benchmark_nexus_scan(
    const float* d_input,
    float* d_output,
    int size,
    int num_iterations
) {
    // Warm-up
    kernels::prefix_sum_inclusive(d_input, d_output, size, 0);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; ++i) {
        kernels::prefix_sum_inclusive(d_input, d_output, size, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / num_iterations;
}

void run_scan_benchmark(int size, int num_iterations = 1000) {
    std::cout << "\n";
    std::cout << "PREFIX SUM BENCHMARK\n";
    std::cout << "Array size: " << size << " elements\n";
    std::cout << "Iterations: " << num_iterations << "\n";
    
    // Allocate memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
    
    // Initialize with ones
    std::vector<float> h_input(size, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    double bytes = 2 * size * sizeof(float); // read + write
    
    std::vector<ReductionBenchmarkResult> results;
    
    // Benchmark Thrust
    float thrust_time = benchmark_thrust_scan(d_input, d_output, size, num_iterations);
    double thrust_bw = (bytes / 1e9) / (thrust_time / 1000.0);
    results.push_back({"Thrust::inclusive_scan", thrust_time, thrust_bw, 1.0});
    
    // Benchmark CUDA Nexus
    float nexus_time = benchmark_nexus_scan(d_input, d_output, size, num_iterations);
    double nexus_bw = (bytes / 1e9) / (nexus_time / 1000.0);
    double speedup = thrust_time / nexus_time;
    results.push_back({"CUDA Nexus (Scan)", nexus_time, nexus_bw, speedup});
    
    print_reduction_results(results);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main(int argc, char** argv) {
    std::cout << "================================================================================\n";
    std::cout << "                   REDUCTION OPERATIONS BENCHMARK                               \n";
    std::cout << "================================================================================\n";
    
    // Print device info
    device::print_device_info(0);
    
    // Run reduction benchmarks
    std::cout << "\n========== SUM REDUCTION ==========\n";
    std::vector<int> sizes = {
        1 << 20,   // 1M elements
        1 << 22,   // 4M elements
        1 << 24,   // 16M elements
        1 << 26,   // 64M elements
    };
    
    for (int size : sizes) {
        run_reduction_benchmark(size);
    }
    
    // Run scan benchmarks
    std::cout << "\n========== PREFIX SUM ==========\n";
    for (int size : {1 << 16, 1 << 18, 1 << 20}) {
        run_scan_benchmark(size);
    }
    
    std::cout << "\nBenchmark completed!\n";
    
    return 0;
}
