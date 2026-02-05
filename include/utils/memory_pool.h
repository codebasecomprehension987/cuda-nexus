#include "utils/memory_pool.h"
#include <algorithm>
#include <iostream>

namespace cuda_nexus {
namespace utils {

MemoryPool::MemoryPool(size_t initial_size)
    : total_allocated_(0)
    , total_free_(0)
    , peak_usage_(0)
    , caching_enabled_(true) {
    
    // Pre-allocate initial pool
    if (initial_size > 0) {
        grow_pool(initial_size);
    }
}

MemoryPool::~MemoryPool() {
    clear();
}

void* MemoryPool::allocate(size_t size, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Try to find a free block of sufficient size
    if (caching_enabled_) {
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            auto alloc_it = allocations_.find(*it);
            if (alloc_it != allocations_.end() && alloc_it->second.size >= size) {
                void* ptr = *it;
                free_blocks_.erase(it);
                alloc_it->second.in_use = true;
                alloc_it->second.stream = stream;
                return ptr;
            }
        }
    }
    
    // No suitable block found, allocate new memory
    void* ptr = allocate_from_system(size);
    
    AllocationInfo info;
    info.ptr = ptr;
    info.size = size;
    info.in_use = true;
    info.stream = stream;
    
    allocations_[ptr] = info;
    total_allocated_ += size;
    peak_usage_ = std::max(peak_usage_, total_allocated_ - total_free_);
    
    return ptr;
}

void MemoryPool::free(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Warning: Attempting to free unknown pointer\n";
        return;
    }
    
    if (caching_enabled_) {
        // Return to free pool
        it->second.in_use = false;
        free_blocks_.push_back(ptr);
        total_free_ += it->second.size;
    } else {
        // Free immediately
        size_t size = it->second.size;
        free_to_system(ptr);
        allocations_.erase(it);
        total_allocated_ -= size;
    }
}

void* MemoryPool::allocate_async(size_t size, cudaStream_t stream) {
    return allocate(size, stream);
}

void MemoryPool::free_async(void* ptr, cudaStream_t stream) {
    // Add stream callback to defer freeing
    cudaStreamAddCallback(stream, 
        [](cudaStream_t stream, cudaError_t status, void* userData) {
            MemoryPool* pool = static_cast<MemoryPool*>(userData);
            // In real implementation, would need to store ptr info
        }, 
        this, 0);
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Free all allocations
    for (auto& pair : allocations_) {
        free_to_system(pair.first);
    }
    
    allocations_.clear();
    free_blocks_.clear();
    total_allocated_ = 0;
    total_free_ = 0;
}

void MemoryPool::grow_pool(size_t min_size) {
    // Round up to nearest 256MB
    size_t grow_size = ((min_size + 256*1024*1024 - 1) / (256*1024*1024)) * (256*1024*1024);
    
    void* ptr = allocate_from_system(grow_size);
    
    AllocationInfo info;
    info.ptr = ptr;
    info.size = grow_size;
    info.in_use = false;
    info.stream = 0;
    
    allocations_[ptr] = info;
    free_blocks_.push_back(ptr);
    total_allocated_ += grow_size;
    total_free_ += grow_size;
}

void* MemoryPool::allocate_from_system(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
    }
    return ptr;
}

void MemoryPool::free_to_system(void* ptr) {
    cudaFree(ptr);
}

// Global pool instance
static MemoryPool global_pool;

MemoryPool& get_default_pool() {
    return global_pool;
}

} // namespace utils
} // namespace cuda_nexus
