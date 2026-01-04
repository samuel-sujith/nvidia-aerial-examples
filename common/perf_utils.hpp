#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <map>
#include <iomanip>

namespace perf_utils {

/// High-resolution timer for performance measurement
class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool is_running_{false};
    
public:
    /// Start timing
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = true;
    }
    
    /// Stop timing and return elapsed microseconds
    double stop() {
        if (!is_running_) {
            return 0.0;
        }
        
        end_time_ = std::chrono::high_resolution_clock::now();
        is_running_ = false;
        
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
        return static_cast<double>(duration.count()) / 1000.0; // Convert to microseconds
    }
    
    /// Get elapsed time without stopping
    double elapsed_microseconds() const {
        if (!is_running_) {
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
            return static_cast<double>(duration.count()) / 1000.0;
        } else {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time_);
            return static_cast<double>(duration.count()) / 1000.0;
        }
    }
    
    /// Reset timer
    void reset() {
        is_running_ = false;
    }
    
    bool is_running() const { return is_running_; }
};

/// RAII performance scope timer
class ScopeTimer {
private:
    PerformanceTimer timer_;
    std::string name_;
    bool print_on_destruction_;
    
public:
    explicit ScopeTimer(const std::string& name, bool print_on_destruction = true) 
        : name_(name), print_on_destruction_(print_on_destruction) {
        timer_.start();
    }
    
    ~ScopeTimer() {
        double elapsed = timer_.stop();
        if (print_on_destruction_) {
            std::cout << "[PERF] " << name_ << ": " << elapsed << " μs\n";
        }
    }
    
    double elapsed_microseconds() const {
        return timer_.elapsed_microseconds();
    }
};

/// Macro for easy scope timing
#define PERF_SCOPE(name) perf_utils::ScopeTimer _perf_timer(name)
#define PERF_SCOPE_SILENT(name) perf_utils::ScopeTimer _perf_timer(name, false)

/// Statistics for multiple timing measurements
struct TimingStats {
    std::string operation_name;
    std::vector<double> measurements; // in microseconds
    
    explicit TimingStats(const std::string& name) : operation_name(name) {}
    
    void add_measurement(double microseconds) {
        measurements.push_back(microseconds);
    }
    
    double mean() const {
        if (measurements.empty()) return 0.0;
        return std::accumulate(measurements.begin(), measurements.end(), 0.0) / measurements.size();
    }
    
    double min() const {
        if (measurements.empty()) return 0.0;
        return *std::min_element(measurements.begin(), measurements.end());
    }
    
    double max() const {
        if (measurements.empty()) return 0.0;
        return *std::max_element(measurements.begin(), measurements.end());
    }
    
    double median() const {
        if (measurements.empty()) return 0.0;
        
        auto sorted = measurements;
        std::sort(sorted.begin(), sorted.end());
        
        size_t n = sorted.size();
        if (n % 2 == 0) {
            return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
        } else {
            return sorted[n/2];
        }
    }
    
    double percentile(double p) const {
        if (measurements.empty()) return 0.0;
        if (p < 0.0 || p > 100.0) return 0.0;
        
        auto sorted = measurements;
        std::sort(sorted.begin(), sorted.end());
        
        double index = (p / 100.0) * (sorted.size() - 1);
        size_t lower = static_cast<size_t>(std::floor(index));
        size_t upper = static_cast<size_t>(std::ceil(index));
        
        if (lower == upper) {
            return sorted[lower];
        } else {
            double weight = index - lower;
            return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
        }
    }
    
    double standard_deviation() const {
        if (measurements.size() <= 1) return 0.0;
        
        double m = mean();
        double sum = 0.0;
        for (double val : measurements) {
            double diff = val - m;
            sum += diff * diff;
        }
        
        return std::sqrt(sum / (measurements.size() - 1));
    }
    
    void print_summary() const {
        if (measurements.empty()) {
            std::cout << "No measurements for " << operation_name << std::endl;
            return;
        }
        
        std::cout << "=== Performance Summary: " << operation_name << " ===" << std::endl;
        std::cout << "Samples:     " << measurements.size() << std::endl;
        std::cout << "Mean:        " << std::fixed << std::setprecision(2) << mean() << " μs" << std::endl;
        std::cout << "Median:      " << std::fixed << std::setprecision(2) << median() << " μs" << std::endl;
        std::cout << "Min:         " << std::fixed << std::setprecision(2) << min() << " μs" << std::endl;
        std::cout << "Max:         " << std::fixed << std::setprecision(2) << max() << " μs" << std::endl;
        std::cout << "Std Dev:     " << std::fixed << std::setprecision(2) << standard_deviation() << " μs" << std::endl;
        std::cout << "P95:         " << std::fixed << std::setprecision(2) << percentile(95) << " μs" << std::endl;
        std::cout << "P99:         " << std::fixed << std::setprecision(2) << percentile(99) << " μs" << std::endl;
        std::cout << std::endl;
    }
    
    void save_to_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        file << "measurement_us\n";
        for (double measurement : measurements) {
            file << std::fixed << std::setprecision(6) << measurement << "\n";
        }
    }
};

/// Throughput calculator
class ThroughputCalculator {
private:
    std::string operation_name_;
    size_t operations_completed_{0};
    PerformanceTimer timer_;
    
public:
    explicit ThroughputCalculator(const std::string& name) : operation_name_(name) {}
    
    void start_measurement() {
        operations_completed_ = 0;
        timer_.start();
    }
    
    void add_operations(size_t count) {
        operations_completed_ += count;
    }
    
    double stop_and_calculate_ops_per_second() {
        double elapsed_seconds = timer_.stop() / 1e6; // Convert μs to seconds
        if (elapsed_seconds == 0.0) return 0.0;
        
        return static_cast<double>(operations_completed_) / elapsed_seconds;
    }
    
    double current_ops_per_second() const {
        double elapsed_seconds = timer_.elapsed_microseconds() / 1e6;
        if (elapsed_seconds == 0.0) return 0.0;
        
        return static_cast<double>(operations_completed_) / elapsed_seconds;
    }
    
    void print_throughput() const {
        std::cout << "[THROUGHPUT] " << operation_name_ << ": " 
                  << std::fixed << std::setprecision(2) 
                  << current_ops_per_second() << " ops/sec" << std::endl;
    }
};

/// Bandwidth calculator for data processing operations
class BandwidthCalculator {
private:
    std::string operation_name_;
    size_t bytes_processed_{0};
    PerformanceTimer timer_;
    
public:
    explicit BandwidthCalculator(const std::string& name) : operation_name_(name) {}
    
    void start_measurement() {
        bytes_processed_ = 0;
        timer_.start();
    }
    
    void add_bytes(size_t bytes) {
        bytes_processed_ += bytes;
    }
    
    double stop_and_calculate_gbps() {
        double elapsed_seconds = timer_.stop() / 1e6; // Convert μs to seconds
        if (elapsed_seconds == 0.0) return 0.0;
        
        double bits_per_second = (static_cast<double>(bytes_processed_) * 8.0) / elapsed_seconds;
        return bits_per_second / 1e9; // Convert to Gbps
    }
    
    double current_gbps() const {
        double elapsed_seconds = timer_.elapsed_microseconds() / 1e6;
        if (elapsed_seconds == 0.0) return 0.0;
        
        double bits_per_second = (static_cast<double>(bytes_processed_) * 8.0) / elapsed_seconds;
        return bits_per_second / 1e9;
    }
    
    void print_bandwidth() const {
        std::cout << "[BANDWIDTH] " << operation_name_ << ": " 
                  << std::fixed << std::setprecision(2) 
                  << current_gbps() << " Gbps" << std::endl;
    }
};

/// Benchmark runner for comparing multiple implementations
class BenchmarkRunner {
private:
    std::map<std::string, TimingStats> results_;
    
public:
    /// Run benchmark for a specific implementation
    template<typename Func>
    void benchmark(const std::string& name, Func&& func, int iterations = 100) {
        if (results_.find(name) == results_.end()) {
            results_.emplace(name, TimingStats(name));
        }
        
        auto& stats = results_.at(name);
        
        for (int i = 0; i < iterations; ++i) {
            PerformanceTimer timer;
            timer.start();
            
            func(); // Execute the function
            
            double elapsed = timer.stop();
            stats.add_measurement(elapsed);
        }
    }
    
    /// Print comparison of all benchmarked implementations
    void print_comparison() const {
        if (results_.empty()) {
            std::cout << "No benchmark results to compare." << std::endl;
            return;
        }
        
        std::cout << "=== Benchmark Comparison ===" << std::endl;
        std::cout << std::setw(20) << "Implementation" 
                  << std::setw(12) << "Mean (μs)" 
                  << std::setw(12) << "Median (μs)"
                  << std::setw(12) << "P95 (μs)"
                  << std::setw(12) << "Std Dev"
                  << std::endl;
        std::cout << std::string(68, '-') << std::endl;
        
        for (const auto& [name, stats] : results_) {
            std::cout << std::setw(20) << name
                      << std::setw(12) << std::fixed << std::setprecision(2) << stats.mean()
                      << std::setw(12) << std::fixed << std::setprecision(2) << stats.median()
                      << std::setw(12) << std::fixed << std::setprecision(2) << stats.percentile(95)
                      << std::setw(12) << std::fixed << std::setprecision(2) << stats.standard_deviation()
                      << std::endl;
        }
        std::cout << std::endl;
    }
    
    /// Get results for specific implementation
    const TimingStats* get_results(const std::string& name) const {
        auto it = results_.find(name);
        return (it != results_.end()) ? &it->second : nullptr;
    }
    
    /// Clear all results
    void clear() {
        results_.clear();
    }
};

/// Memory usage profiler
class MemoryProfiler {
private:
    std::string name_;
    size_t initial_memory_{0};
    
public:
    explicit MemoryProfiler(const std::string& name) : name_(name) {
        // Note: Actual memory measurement would require platform-specific code
        // This is a placeholder for the interface
    }
    
    void start_profiling() {
        // Platform-specific memory measurement would go here
        initial_memory_ = 0; // Placeholder
    }
    
    void stop_and_print() {
        // Platform-specific memory measurement would go here
        size_t final_memory = 0; // Placeholder
        size_t memory_used = final_memory - initial_memory_;
        
        std::cout << "[MEMORY] " << name_ << ": Used " 
                  << (memory_used / 1024.0 / 1024.0) << " MB" << std::endl;
    }
};

/// Utility functions for performance analysis
namespace utils {

/// Convert microseconds to human-readable time string
inline std::string format_time(double microseconds) {
    if (microseconds < 1000.0) {
        return std::to_string(static_cast<int>(microseconds)) + " μs";
    } else if (microseconds < 1000000.0) {
        return std::to_string(static_cast<int>(microseconds / 1000.0)) + " ms";
    } else {
        return std::to_string(static_cast<int>(microseconds / 1000000.0)) + " s";
    }
}

/// Convert bytes to human-readable size string
inline std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double size = static_cast<double>(bytes);
    int unit_index = 0;
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    return std::to_string(static_cast<int>(size)) + " " + units[unit_index];
}

/// Calculate efficiency metric (operations per microsecond)
inline double calculate_efficiency(size_t operations, double microseconds) {
    if (microseconds == 0.0) return 0.0;
    return static_cast<double>(operations) / microseconds;
}

} // namespace utils

} // namespace perf_utils