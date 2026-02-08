#ifndef CUDA_NEXUS_TENSOR_H
#define CUDA_NEXUS_TENSOR_H

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <initializer_list>
#include "cuda_nexus.h"

namespace cuda_nexus {

// Simple tensor wrapper for easier memory management
template<typename T>
class Tensor {
public:
    // Constructors
    Tensor() : data_(nullptr), size_(0), device_id_(0) {}
    
    explicit Tensor(const std::vector<int>& shape, int device = 0)
        : shape_(shape), device_id_(device) {
        
        size_ = 1;
        for (int dim : shape) {
            size_ *= dim;
        }
        
        allocate();
    }
    
    Tensor(std::initializer_list<int> shape, int device = 0)
        : Tensor(std::vector<int>(shape), device) {}
    
    // Destructor
    ~Tensor() {
        free();
    }
    
    // Copy constructor (deep copy)
    Tensor(const Tensor& other)
        : shape_(other.shape_), size_(other.size_), device_id_(other.device_id_) {
        
        allocate();
        if (other.data_) {
            CUDA_CHECK(cudaMemcpy(data_, other.data_, size_ * sizeof(T), 
                                  cudaMemcpyDeviceToDevice));
        }
    }
    
    // Move constructor
    Tensor(Tensor&& other) noexcept
        : data_(other.data_), shape_(std::move(other.shape_)),
          size_(other.size_), device_id_(other.device_id_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    // Copy assignment
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            free();
            shape_ = other.shape_;
            size_ = other.size_;
            device_id_ = other.device_id_;
            allocate();
            if (other.data_) {
                CUDA_CHECK(cudaMemcpy(data_, other.data_, size_ * sizeof(T), 
                                      cudaMemcpyDeviceToDevice));
            }
        }
        return *this;
    }
    
    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            free();
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            size_ = other.size_;
            device_id_ = other.device_id_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Reshape (doesn't copy data)
    void reshape(const std::vector<int>& new_shape) {
        size_t new_size = 1;
        for (int dim : new_shape) {
            new_size *= dim;
        }
        
        if (new_size != size_) {
            throw std::runtime_error("Reshape: new size doesn't match current size");
        }
        
        shape_ = new_shape;
    }
    
    // Copy data from host
    void copy_from_host(const T* host_data) {
        if (!data_) allocate();
        CUDA_CHECK(cudaMemcpy(data_, host_data, size_ * sizeof(T), 
                              cudaMemcpyHostToDevice));
    }
    
    void copy_from_host(const std::vector<T>& host_data) {
        if (host_data.size() != size_) {
            throw std::runtime_error("Host data size doesn't match tensor size");
        }
        copy_from_host(host_data.data());
    }
    
    // Copy data to host
    void copy_to_host(T* host_data) const {
        if (!data_) {
            throw std::runtime_error("Tensor not allocated");
        }
        CUDA_CHECK(cudaMemcpy(host_data, data_, size_ * sizeof(T), 
                              cudaMemcpyDeviceToHost));
    }
    
    std::vector<T> copy_to_host() const {
        std::vector<T> result(size_);
        copy_to_host(result.data());
        return result;
    }
    
    // Fill with value
    void fill(T value) {
        std::vector<T> host_data(size_, value);
        copy_from_host(host_data);
    }
    
    void zeros() { fill(T(0)); }
    void ones() { fill(T(1)); }
    
    // Accessors
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    const std::vector<int>& shape() const { return shape_; }
    int ndim() const { return shape_.size(); }
    int device() const { return device_id_; }
    
    // Shape accessors
    int dim(int idx) const {
        if (idx < 0 || idx >= (int)shape_.size()) {
            throw std::out_of_range("Dimension index out of range");
        }
        return shape_[idx];
    }
    
    // Check if allocated
    bool is_allocated() const { return data_ != nullptr; }
    
    // Memory size in bytes
    size_t bytes() const { return size_ * sizeof(T); }
    
private:
    void allocate() {
        if (size_ > 0) {
            int current_device;
            CUDA_CHECK(cudaGetDevice(&current_device));
            CUDA_CHECK(cudaSetDevice(device_id_));
            CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
            CUDA_CHECK(cudaSetDevice(current_device));
        }
    }
    
    void free() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }
    
    T* data_;
    std::vector<int> shape_;
    size_t size_;
    int device_id_;
};

// Common tensor types
using TensorF32 = Tensor<float>;
using TensorF16 = Tensor<half>;
using TensorI32 = Tensor<int>;
using TensorI8 = Tensor<int8_t>;

} // namespace cuda_nexus

#endif // CUDA_NEXUS_TENSOR_H
