#include "utils/profiler.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace cuda_nexus {
namespace utils {

Profiler& Profiler::instance() {
    static Profiler profiler;
    return profiler;
}

void Profiler::start_timer(const std::string& name, cudaStream_t stream) {
    if (!enabled_) return;
    
    TimerInfo info;
    info.stream = stream;
    
    CUDA_CHECK(cudaEventCreate(&info.start_event));
    CUDA_CHECK(cudaEventCreate(&info.stop_event));
    CUDA_CHECK(cudaEventRecord(info.start_event, stream));
    
    active_timers_[name] = info;
}

void Profiler::stop_timer(const std::string& name, cudaStream_t stream) {
    if (!enabled_) return;
    
    auto it = active_timers_.find(name);
    if (it == active_timers_.end()) {
        std::cerr << "Warning: Timer '" << name << "' was not started\n";
        return;
    }
    
    TimerInfo& info = it->second;
    CUDA_CHECK(cudaEventRecord(info.stop_event, info.stream));
    CUDA_CHECK(cudaEventSynchronize(info.stop_event));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, info.start_event, info.stop_event));
    
    // Store timing
    timings_[name].push_back(milliseconds);
    
    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(info.start_event));
    CUDA_CHECK(cudaEventDestroy(info.stop_event));
    
    active_timers_.erase(it);
}

Profiler::KernelStats Profiler::get_stats(const std::string& name) const {
    KernelStats stats;
    stats.name = name;
    stats.call_count = 0;
    stats.total_time_ms = 0.0f;
    stats.avg_time_ms = 0.0f;
    stats.min_time_ms = std::numeric_limits<float>::max();
    stats.max_time_ms = 0.0f;
    
    auto it = timings_.find(name);
    if (it == timings_.end() || it->second.empty()) {
        return stats;
    }
    
    const auto& times = it->second;
    stats.call_count = times.size();
    
    for (float time : times) {
        stats.total_time_ms += time;
        stats.min_time_ms = std::min(stats.min_time_ms, time);
        stats.max_time_ms = std::max(stats.max_time_ms, time);
    }
    
    stats.avg_time_ms = stats.total_time_ms / stats.call_count;
    
    return stats;
}

std::vector<Profiler::KernelStats> Profiler::get_all_stats() const {
    std::vector<KernelStats> all_stats;
    
    for (const auto& pair : timings_) {
        all_stats.push_back(get_stats(pair.first));
    }
    
    // Sort by total time (descending)
    std::sort(all_stats.begin(), all_stats.end(),
        [](const KernelStats& a, const KernelStats& b) {
            return a.total_time_ms > b.total_time_ms;
        });
    
    return all_stats;
}

void Profiler::reset() {
    timings_.clear();
    active_timers_.clear();
}

void Profiler::print_summary() const {
    auto all_stats = get_all_stats();
    
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "                         CUDA NEXUS PROFILING SUMMARY                          \n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(30) << "Kernel Name"
              << std::right << std::setw(10) << "Calls"
              << std::setw(12) << "Total (ms)"
              << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    
    float grand_total = 0.0f;
    
    for (const auto& stats : all_stats) {
        std::cout << std::left << std::setw(30) << stats.name
                  << std::right << std::setw(10) << stats.call_count
                  << std::setw(12) << std::fixed << std::setprecision(3) << stats.total_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << stats.avg_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << stats.min_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << stats.max_time_ms
                  << "\n";
        grand_total += stats.total_time_ms;
    }
    
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(30) << "TOTAL"
              << std::right << std::setw(10) << ""
              << std::setw(12) << std::fixed << std::setprecision(3) << grand_total
              << "\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";
}

} // namespace utils
} // namespace cuda_nexus
