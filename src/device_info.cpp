#include "cuda_nexus.h"
#include <iostream>
#include <sstream>

namespace cuda_nexus {

DeviceProperties DeviceProperties::get(int device) {
    DeviceProperties props;
    
    cudaDeviceProp cuda_props;
    CUDA_CHECK(cudaGetDeviceProperties(&cuda_props, device));
    
    props.sm_count = cuda_props.multiProcessorCount;
    props.warp_size = cuda_props.warpSize;
    props.max_threads_per_block = cuda_props.maxThreadsPerBlock;
    props.max_shared_memory_per_block = cuda_props.sharedMemPerBlock;
    props.compute_capability_major = cuda_props.major;
    props.compute_capability_minor = cuda_props.minor;
    
    // Tensor Cores available on compute capability 7.0+
    props.tensor_cores_available = (cuda_props.major >= 7);
    
    return props;
}

// Additional device utility functions
namespace device {

int get_device_count() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

int get_current_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

void set_device(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

std::string get_device_name(int device) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    return std::string(props.name);
}

size_t get_total_memory(int device) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    return props.totalGlobalMem;
}

size_t get_free_memory() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

bool supports_tensor_cores(int device) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    return props.major >= 7;
}

bool supports_cooperative_groups(int device) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    return props.cooperativeLaunch;
}

void print_device_info(int device) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    
    std::cout << "\n";
    std::cout << "=================================================================\n";
    std::cout << "                    CUDA DEVICE INFORMATION                      \n";
    std::cout << "=================================================================\n";
    std::cout << "Device " << device << ": " << props.name << "\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "Compute Capability:        " << props.major << "." << props.minor << "\n";
    std::cout << "Total Global Memory:       " << props.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "Shared Memory per Block:   " << props.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "Registers per Block:       " << props.regsPerBlock << "\n";
    std::cout << "Warp Size:                 " << props.warpSize << "\n";
    std::cout << "Max Threads per Block:     " << props.maxThreadsPerBlock << "\n";
    std::cout << "Max Thread Dimensions:     (" << props.maxThreadsDim[0] << ", " 
              << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << ")\n";
    std::cout << "Max Grid Dimensions:       (" << props.maxGridSize[0] << ", " 
              << props.maxGridSize[1] << ", " << props.maxGridSize[2] << ")\n";
    std::cout << "Multiprocessor Count:      " << props.multiProcessorCount << "\n";
    std::cout << "Clock Rate:                " << props.clockRate / 1000 << " MHz\n";
    std::cout << "Memory Clock Rate:         " << props.memoryClockRate / 1000 << " MHz\n";
    std::cout << "Memory Bus Width:          " << props.memoryBusWidth << " bits\n";
    std::cout << "L2 Cache Size:             " << props.l2CacheSize / 1024 << " KB\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "Features:\n";
    std::cout << "  Tensor Cores:            " << (props.major >= 7 ? "YES" : "NO") << "\n";
    std::cout << "  Unified Memory:          " << (props.unifiedAddressing ? "YES" : "NO") << "\n";
    std::cout << "  Cooperative Launch:      " << (props.cooperativeLaunch ? "YES" : "NO") << "\n";
    std::cout << "  Concurrent Kernels:      " << (props.concurrentKernels ? "YES" : "NO") << "\n";
    std::cout << "  ECC Enabled:             " << (props.ECCEnabled ? "YES" : "NO") << "\n";
    std::cout << "=================================================================\n";
    std::cout << "\n";
}

void print_all_devices() {
    int device_count = get_device_count();
    std::cout << "Found " << device_count << " CUDA device(s)\n";
    
    for (int i = 0; i < device_count; ++i) {
        print_device_info(i);
    }
}

// Compute theoretical peak performance
double get_peak_gflops_fp32(int device) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    
    // FP32 FLOPS = SM_count * cores_per_SM * clock_rate * 2 (FMA)
    // Approximate cores per SM based on architecture
    int cores_per_sm;
    if (props.major == 7) cores_per_sm = 64;      // Volta/Turing
    else if (props.major == 8) cores_per_sm = 64; // Ampere (A100 has 64)
    else if (props.major == 9) cores_per_sm = 128; // Hopper
    else cores_per_sm = 64; // Default estimate
    
    double gflops = props.multiProcessorCount * cores_per_sm * 
                    (props.clockRate / 1e6) * 2.0; // FMA = 2 ops
    
    return gflops;
}

double get_peak_gflops_fp16(int device) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    
    if (props.major < 7) return 0.0; // No Tensor Cores
    
    // Tensor Cores can do much higher throughput
    // This is a rough estimate
    return get_peak_gflops_fp32(device) * 8.0; // ~8x for Tensor Cores
}

double get_memory_bandwidth_gbps(int device) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    
    // Memory bandwidth = memory_clock * bus_width / 8 * 2 (DDR)
    double bandwidth = (props.memoryClockRate / 1e6) * 
                       (props.memoryBusWidth / 8.0) * 2.0;
    
    return bandwidth;
}

} // namespace device
} // namespace cuda_nexus
