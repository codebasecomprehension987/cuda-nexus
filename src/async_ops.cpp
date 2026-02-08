#include "utils/async_ops.h"
#include <iostream>

namespace cuda_nexus {
namespace utils {

// AsyncOpManager implementation
AsyncOpManager::AsyncOpManager() {}

AsyncOpManager::~AsyncOpManager() {
    synchronize_all();
}

void AsyncOpManager::add_callback(cudaStream_t stream, Callback callback) {
    // Find or create stream info
    StreamInfo* info = nullptr;
    for (auto& si : stream_infos_) {
        if (si.stream == stream) {
            info = &si;
            break;
        }
    }
    
    if (!info) {
        stream_infos_.push_back({stream, {}});
        info = &stream_infos_.back();
    }
    
    info->callbacks.push_back(callback);
}

void AsyncOpManager::synchronize_all() {
    for (auto& info : stream_infos_) {
        synchronize_stream(info.stream);
    }
    stream_infos_.clear();
}

void AsyncOpManager::synchronize_stream(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    process_callbacks(stream);
}

bool AsyncOpManager::is_stream_idle(cudaStream_t stream) {
    cudaError_t status = cudaStreamQuery(stream);
    return status == cudaSuccess;
}

void AsyncOpManager::process_callbacks(cudaStream_t stream) {
    for (auto it = stream_infos_.begin(); it != stream_infos_.end(); ) {
        if (it->stream == stream) {
            // Execute all callbacks
            for (auto& callback : it->callbacks) {
                callback();
            }
            it = stream_infos_.erase(it);
        } else {
            ++it;
        }
    }
}

// StreamPool implementation
StreamPool::StreamPool(int num_streams) 
    : num_streams_(num_streams) {
    
    streams_.resize(num_streams);
    available_.resize(num_streams, true);
    
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
}

StreamPool::~StreamPool() {
    synchronize_all();
    
    for (auto stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

cudaStream_t StreamPool::get_stream() {
    for (int i = 0; i < num_streams_; ++i) {
        if (available_[i]) {
            available_[i] = false;
            return streams_[i];
        }
    }
    
    // No available stream, create a new one
    cudaStream_t new_stream;
    CUDA_CHECK(cudaStreamCreate(&new_stream));
    streams_.push_back(new_stream);
    available_.push_back(false);
    num_streams_++;
    
    return new_stream;
}

void StreamPool::return_stream(cudaStream_t stream) {
    for (int i = 0; i < num_streams_; ++i) {
        if (streams_[i] == stream) {
            // Synchronize before returning
            cudaStreamSynchronize(stream);
            available_[i] = true;
            return;
        }
    }
}

void StreamPool::synchronize_all() {
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }
    
    // Mark all as available
    std::fill(available_.begin(), available_.end(), true);
}

int StreamPool::get_available_count() const {
    return std::count(available_.begin(), available_.end(), true);
}

// GraphCapture implementation
GraphCapture::GraphCapture() 
    : graph_(nullptr)
    , graph_exec_(nullptr)
    , capture_stream_(nullptr) {}

GraphCapture::~GraphCapture() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
    }
}

void GraphCapture::begin_capture(cudaStream_t stream) {
    capture_stream_ = stream;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
}

void GraphCapture::end_capture() {
    if (!capture_stream_) {
        throw std::runtime_error("Graph capture was not started");
    }
    
    CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
    
    capture_stream_ = nullptr;
}

void GraphCapture::execute(cudaStream_t stream) {
    if (!graph_exec_) {
        throw std::runtime_error("Graph has not been captured");
    }
    
    CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
}

// Pipeline implementation
Pipeline::Pipeline(int num_stages) 
    : num_stages_(num_stages) {
    
    streams_.resize(num_stages);
    for (int i = 0; i < num_stages; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
}

Pipeline::~Pipeline() {
    synchronize();
    
    for (auto stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

void Pipeline::add_stage(std::function<void(cudaStream_t)> stage_func) {
    stages_.push_back(stage_func);
}

void Pipeline::execute() {
    if (stages_.empty()) return;
    
    // Execute stages in pipeline fashion
    for (size_t i = 0; i < stages_.size(); ++i) {
        int stream_idx = i % num_stages_;
        stages_[i](streams_[stream_idx]);
    }
}

void Pipeline::synchronize() {
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }
}

// Global instances
static AsyncOpManager global_async_manager;
static StreamPool global_stream_pool(4);

AsyncOpManager& get_async_manager() {
    return global_async_manager;
}

StreamPool& get_stream_pool() {
    return global_stream_pool;
}

} // namespace utils
} // namespace cuda_nexus
