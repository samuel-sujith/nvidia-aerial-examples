/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file mimo_detection_example.cpp
 * @brief Complete example demonstrating MIMO detection pipeline and GPU acceleration
 *
 * This example shows:
 * 1. How to create and configure a MIMO detection module
 * 2. How to set up a GPU-based MIMO detection pipeline
 * 3. How to execute different MIMO detection algorithms (ZF, MMSE, ML)
 * 4. Performance analysis and real-time streaming capabilities
 */

#include <iostream>
#include <vector>
#include <memory>
#include <complex>
#include <random>
#include <chrono>
#include <map>
#include <iomanip>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "mimo_detector.hpp"
#include "mimo_pipeline.hpp"
#include "tensor/tensor_info.hpp"
#include "pipeline/pipeline_spec.hpp"

using namespace mimo_detection;

/**
 * Generate synthetic MIMO channel and signal data
 */
class SyntheticMIMOGenerator {
public:
    explicit SyntheticMIMOGenerator(uint32_t seed = 13579) : rng_(seed) {}

    struct ChannelModel {
        std::vector<std::vector<std::complex<float>>> channel_matrix;
        float snr_db{20.0f};
        size_t num_tx{2};
        size_t num_rx{4};
    };

    ChannelModel generate_rayleigh_channel(size_t num_tx, size_t num_rx, float snr_db) {
        ChannelModel model;
        model.num_tx = num_tx;
        model.num_rx = num_rx;
        model.snr_db = snr_db;
        model.channel_matrix.resize(num_rx, std::vector<std::complex<float>>(num_tx));
        
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t rx = 0; rx < num_rx; ++rx) {
            for (size_t tx = 0; tx < num_tx; ++tx) {
                float real_part = dist(rng_);
                float imag_part = dist(rng_);
                model.channel_matrix[rx][tx] = std::complex<float>(real_part, imag_part) / std::sqrt(2.0f);
            }
        }
        
        return model;
    }

    std::vector<std::complex<float>> generate_transmitted_symbols(size_t num_symbols, size_t num_tx) {
        std::vector<std::complex<float>> symbols(num_symbols * num_tx);
        std::uniform_int_distribution<int> symbol_dist(0, 3);
        
        // QPSK constellation
        std::vector<std::complex<float>> constellation = {
            { 1.0f/std::sqrt(2.0f),  1.0f/std::sqrt(2.0f)},
            { 1.0f/std::sqrt(2.0f), -1.0f/std::sqrt(2.0f)},
            {-1.0f/std::sqrt(2.0f),  1.0f/std::sqrt(2.0f)},
            {-1.0f/std::sqrt(2.0f), -1.0f/std::sqrt(2.0f)}
        };
        
        for (size_t i = 0; i < symbols.size(); ++i) {
            symbols[i] = constellation[symbol_dist(rng_)];
        }
        
        return symbols;
    }

    std::vector<std::complex<float>> apply_channel_and_noise(
        const std::vector<std::complex<float>>& tx_symbols,
        const ChannelModel& channel,
        size_t num_symbols) {
        
        std::vector<std::complex<float>> rx_signals(num_symbols * channel.num_rx);
        std::normal_distribution<float> noise_dist(0.0f, std::sqrt(std::pow(10.0f, -channel.snr_db / 10.0f) / 2.0f));
        
        for (size_t sym = 0; sym < num_symbols; ++sym) {
            for (size_t rx = 0; rx < channel.num_rx; ++rx) {
                std::complex<float> signal(0.0f, 0.0f);
                
                // Apply channel
                for (size_t tx = 0; tx < channel.num_tx; ++tx) {
                    signal += channel.channel_matrix[rx][tx] * tx_symbols[sym * channel.num_tx + tx];
                }
                
                // Add noise
                signal += std::complex<float>(noise_dist(rng_), noise_dist(rng_));
                
                rx_signals[sym * channel.num_rx + rx] = signal;
            }
        }
        
        return rx_signals;
    }

private:
    std::mt19937 rng_;
};

/**
 * Performance measurement utility
 */
class PerformanceMeasurement {
public:
    void start_measurement(const std::string& name) {
        measurements_[name].start_time = std::chrono::high_resolution_clock::now();
    }
    
    void end_measurement(const std::string& name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto& measurement = measurements_[name];
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - measurement.start_time
        );
        
        measurement.total_time_us += duration.count();
        measurement.count++;
        measurement.last_time_us = duration.count();
    }
    
    void print_results() const {
        std::cout << "\n=== Performance Results ===\n";
        for (const auto& [name, measurement] : measurements_) {
            double avg_time_us = static_cast<double>(measurement.total_time_us) / measurement.count;
            std::cout << name << ":\n";
            std::cout << "  Average: " << avg_time_us << " μs\n";
            std::cout << "  Throughput: " << (1000000.0 / avg_time_us) << " executions/second\n\n";
        }
    }

    double get_average_time(const std::string& name) const {
        auto it = measurements_.find(name);
        if (it != measurements_.end() && it->second.count > 0) {
            return static_cast<double>(it->second.total_time_us) / it->second.count;
        }
        return 0.0;
    }

private:
    struct Measurement {
        std::chrono::high_resolution_clock::time_point start_time;
        uint64_t total_time_us{0};
        uint64_t last_time_us{0};
        uint32_t count{0};
    };
    
    std::map<std::string, Measurement> measurements_;
};

/**
 * Example configuration for different scenarios
 */
struct ExampleConfig {
    size_t num_tx{2};
    size_t num_rx{4};
    size_t num_symbols{5000};
    float snr_db{20.0f};
    int num_iterations{100};
    MIMOAlgorithm algorithm{MIMOAlgorithm::MMSE};
    bool use_cuda_graphs{true};
    bool validate_results{true};
};

/**
 * MIMO detection validation utility
 */
class MIMOValidator {
public:
    static double calculate_symbol_error_rate(const std::vector<std::complex<float>>& original,
                                             const std::vector<std::complex<float>>& detected) {
        if (original.size() != detected.size()) {
            return 1.0;
        }
        
        size_t errors = 0;
        const float threshold = 0.5f;
        
        for (size_t i = 0; i < original.size(); ++i) {
            // Simple QPSK detection
            int orig_real = original[i].real() > 0 ? 1 : 0;
            int orig_imag = original[i].imag() > 0 ? 1 : 0;
            int det_real = detected[i].real() > 0 ? 1 : 0;
            int det_imag = detected[i].imag() > 0 ? 1 : 0;
            
            if (orig_real != det_real || orig_imag != det_imag) {
                errors++;
            }
        }
        
        return static_cast<double>(errors) / original.size();
    }
    
    static double calculate_evm_percent(const std::vector<std::complex<float>>& original,
                                       const std::vector<std::complex<float>>& detected) {
        double error_power = 0.0;
        double signal_power = 0.0;
        
        for (size_t i = 0; i < original.size(); ++i) {
            auto error = detected[i] - original[i];
            error_power += std::norm(error);
            signal_power += std::norm(original[i]);
        }
        
        return 100.0 * std::sqrt(error_power / signal_power);
    }
};

/**
 * Main example function demonstrating the MIMO detection pipeline
 */
int run_mimo_example(const ExampleConfig& config) {
    std::cout << "=== MIMO Detection Pipeline Example ===\n";
    std::cout << "Configuration:\n";
    std::cout << "  MIMO: " << config.num_tx << "x" << config.num_rx << "\n";
    std::cout << "  Number of symbols: " << config.num_symbols << "\n";
    std::cout << "  SNR: " << config.snr_db << " dB\n";
    std::cout << "  Algorithm: ";
    switch (config.algorithm) {
        case MIMOAlgorithm::ZeroForcing: std::cout << "Zero Forcing"; break;
        case MIMOAlgorithm::MMSE: std::cout << "MMSE"; break;
        case MIMOAlgorithm::MaximumLikelihood: std::cout << "Maximum Likelihood"; break;
    }
    std::cout << "\n";
    std::cout << "  Iterations: " << config.num_iterations << "\n";
    std::cout << "  CUDA Graphs: " << (config.use_cuda_graphs ? "Enabled" : "Disabled") << "\n\n";

    PerformanceMeasurement perf;

    try {
        // Step 1: Create pipeline configuration
        perf.start_measurement("Pipeline Creation");
        
        MIMOPipelineConfig pipeline_config;
        pipeline_config.num_tx_antennas = config.num_tx;
        pipeline_config.num_rx_antennas = config.num_rx;
        pipeline_config.max_batch_size = 1024;
        pipeline_config.enable_cuda_graphs = config.use_cuda_graphs;
        pipeline_config.supported_algorithms = {config.algorithm};
        
        auto pipeline = MIMOPipelineFactory::create_pipeline(pipeline_config);
        
        perf.end_measurement("Pipeline Creation");
        
        // Step 2: Setup pipeline
        aerial::pipeline::PipelineSpec spec;
        if (!pipeline->setup(spec)) {
            std::cerr << "Failed to setup MIMO pipeline\n";
            return 1;
        }
        
        // Step 3: Generate test data
        perf.start_measurement("Data Generation");
        
        SyntheticMIMOGenerator mimo_gen(54321);
        auto channel_model = mimo_gen.generate_rayleigh_channel(config.num_tx, config.num_rx, config.snr_db);
        auto tx_symbols = mimo_gen.generate_transmitted_symbols(config.num_symbols, config.num_tx);
        auto rx_signals = mimo_gen.apply_channel_and_noise(tx_symbols, channel_model, config.num_symbols);
        
        perf.end_measurement("Data Generation");
        
        // Step 4: Setup GPU memory
        perf.start_measurement("GPU Memory Setup");
        
        std::complex<float>* d_rx_signals = nullptr;
        std::complex<float>* d_channel_matrix = nullptr;
        std::complex<float>* d_detected_symbols = nullptr;
        
        size_t rx_size = config.num_symbols * config.num_rx;
        size_t channel_size = config.num_rx * config.num_tx;
        size_t tx_size = config.num_symbols * config.num_tx;
        
        cudaError_t err = cudaMalloc(&d_rx_signals, rx_size * sizeof(std::complex<float>));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate rx signals: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        
        err = cudaMalloc(&d_channel_matrix, channel_size * sizeof(std::complex<float>));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate channel matrix: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        
        err = cudaMalloc(&d_detected_symbols, tx_size * sizeof(std::complex<float>));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate detected symbols: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        
        // Copy data to GPU
        cudaMemcpy(d_rx_signals, rx_signals.data(), rx_size * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
        
        // Flatten and copy channel matrix
        std::vector<std::complex<float>> flat_channel(channel_size);
        for (size_t rx = 0; rx < config.num_rx; ++rx) {
            for (size_t tx = 0; tx < config.num_tx; ++tx) {
                flat_channel[rx * config.num_tx + tx] = channel_model.channel_matrix[rx][tx];
            }
        }
        cudaMemcpy(d_channel_matrix, flat_channel.data(), channel_size * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
        
        perf.end_measurement("GPU Memory Setup");
        
        // Step 5: Setup tensor info objects
        std::vector<tensor::TensorInfo> inputs(2);
        std::vector<tensor::TensorInfo> outputs(1);
        
        inputs[0].set_data(d_rx_signals);
        inputs[0].set_dimensions({config.num_symbols, config.num_rx});
        inputs[0].set_element_type(tensor::ElementType::COMPLEX_FLOAT32);
        
        inputs[1].set_data(d_channel_matrix);
        inputs[1].set_dimensions({config.num_rx, config.num_tx});
        inputs[1].set_element_type(tensor::ElementType::COMPLEX_FLOAT32);
        
        outputs[0].set_data(d_detected_symbols);
        outputs[0].set_dimensions({config.num_symbols, config.num_tx});
        outputs[0].set_element_type(tensor::ElementType::COMPLEX_FLOAT32);
        
        // Step 6: Warm-up execution
        std::cout << "Performing warm-up execution...\n";
        task::CancellationToken token;
        
        std::vector<std::complex<float>> detected_symbols;
        auto result = pipeline->detect_symbols(rx_signals, channel_model.channel_matrix, 
                                             detected_symbols, config.algorithm);
        
        if (!result.is_success()) {
            std::cerr << "Warm-up execution failed: " << result.message << "\n";
            return 1;
        }
        
        std::cout << "Warm-up completed successfully\n";
        
        // Step 7: Validation check
        if (config.validate_results) {
            perf.start_measurement("Validation");
            
            double ser = MIMOValidator::calculate_symbol_error_rate(tx_symbols, detected_symbols);
            double evm = MIMOValidator::calculate_evm_percent(tx_symbols, detected_symbols);
            
            std::cout << "Symbol Error Rate: " << std::scientific << std::setprecision(2) << ser 
                     << " (" << (ser * 100) << "%)\n";
            std::cout << "EVM: " << std::fixed << std::setprecision(2) << evm << "%\n";
            
            if (ser < 0.1) { // Less than 10% error rate
                std::cout << "Detection quality: GOOD\n";
            } else if (ser < 0.3) {
                std::cout << "Detection quality: FAIR\n";
            } else {
                std::cout << "Detection quality: POOR\n";
            }
            
            // Print sample symbols
            std::cout << "Sample detected symbols:\n";
            for (size_t i = 0; i < std::min(size_t(4), detected_symbols.size()); i += config.num_tx) {
                std::cout << "  Symbol set[" << i/config.num_tx << "]:\n";
                for (size_t tx = 0; tx < config.num_tx; ++tx) {
                    size_t idx = i + tx;
                    if (idx < detected_symbols.size()) {
                        std::cout << "    TX" << tx << ": " << detected_symbols[idx].real() 
                                 << " + " << detected_symbols[idx].imag() << "j\n";
                    }
                }
            }
            
            perf.end_measurement("Validation");
        }
        
        // Step 8: Performance benchmark
        std::cout << "Running performance benchmark (" << config.num_iterations << " iterations)...\n";
        
        for (int i = 0; i < config.num_iterations; ++i) {
            detected_symbols.clear();
            
            if (config.use_cuda_graphs && i > 0) {
                perf.start_measurement("Pipeline Execution (Graph)");
                result = pipeline->execute_pipeline_graph(inputs, outputs, token);
                perf.end_measurement("Pipeline Execution (Graph)");
            } else {
                perf.start_measurement("Pipeline Execution (Stream)");
                result = pipeline->detect_symbols(rx_signals, channel_model.channel_matrix, 
                                                detected_symbols, config.algorithm);
                perf.end_measurement("Pipeline Execution (Stream)");
            }
            
            if (!result.is_success()) {
                std::cerr << "Execution " << i << " failed: " << result.message << "\n";
                return 1;
            }
            
            // Progress indication
            if ((i + 1) % (config.num_iterations / 10) == 0) {
                std::cout << "  Completed " << (i + 1) << "/" << config.num_iterations << " iterations\n";
            }
        }
        
        // Step 9: Calculate throughput metrics
        double avg_time_us = perf.get_average_time("Pipeline Execution (Stream)");
        if (config.use_cuda_graphs) {
            avg_time_us = perf.get_average_time("Pipeline Execution (Graph)");
        }
        
        double symbols_per_sec = (config.num_symbols * config.num_tx) / (avg_time_us / 1000000.0);
        double msymbols_per_sec = symbols_per_sec / 1000000.0;
        
        std::cout << "\n=== Throughput Analysis ===\n";
        std::cout << "MIMO configuration: " << config.num_tx << "x" << config.num_rx << "\n";
        std::cout << "Processing rate: " << msymbols_per_sec << " Msymbols/sec\n";
        std::cout << "Detection rate: " << (1000000.0 / avg_time_us) << " detections/sec\n";
        std::cout << "Spectral efficiency: " << config.num_tx << " streams\n";
        
        // Step 10: Print pipeline statistics
        auto pipeline_stats = pipeline->get_mimo_stats();
        std::cout << "\n=== Pipeline Statistics ===\n";
        std::cout << "Total symbols detected: " << pipeline_stats.total_symbols_detected << "\n";
        std::cout << "Average detection time: " << pipeline_stats.average_detection_time_us() << " μs\n";
        std::cout << "Peak throughput: " << pipeline_stats.average_throughput_msymbols_per_sec() << " Msymbols/sec\n";
        
        // Step 11: Cleanup
        perf.start_measurement("Cleanup");
        pipeline->teardown();
        
        cudaFree(d_rx_signals);
        cudaFree(d_channel_matrix);
        cudaFree(d_detected_symbols);
        perf.end_measurement("Cleanup");
        
        // Step 12: Print performance results
        perf.print_results();
        
        std::cout << "MIMO detection example completed successfully!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << "\n";
        return 1;
    }
}

/**
 * Compare different MIMO detection algorithms
 */
int run_algorithm_comparison(size_t num_tx, size_t num_rx, float snr_db) {
    std::cout << "\n=== MIMO Algorithm Comparison ===\n";
    std::cout << "Configuration: " << num_tx << "x" << num_rx << " MIMO, SNR: " << snr_db << " dB\n\n";
    
    std::vector<MIMOAlgorithm> algorithms = {
        MIMOAlgorithm::ZeroForcing,
        MIMOAlgorithm::MMSE,
        MIMOAlgorithm::MaximumLikelihood
    };
    
    std::vector<std::string> algorithm_names = {"Zero Forcing", "MMSE", "Maximum Likelihood"};
    
    PerformanceMeasurement global_perf;
    
    for (size_t i = 0; i < algorithms.size(); ++i) {
        std::cout << "Testing " << algorithm_names[i] << ":\n";
        
        ExampleConfig config;
        config.num_tx = num_tx;
        config.num_rx = num_rx;
        config.snr_db = snr_db;
        config.algorithm = algorithms[i];
        config.num_symbols = 2000;
        config.num_iterations = 50;
        config.validate_results = false;
        
        global_perf.start_measurement(algorithm_names[i]);
        int result = run_mimo_example(config);
        global_perf.end_measurement(algorithm_names[i]);
        
        if (result != 0) {
            std::cerr << "Failed to run " << algorithm_names[i] << " test\n";
            return result;
        }
    }
    
    std::cout << "\n=== Algorithm Comparison Results ===\n";
    global_perf.print_results();
    
    return 0;
}

/**
 * Test different MIMO configurations
 */
int run_mimo_configuration_comparison() {
    std::cout << "\n=== MIMO Configuration Comparison ===\n";
    
    std::vector<std::pair<size_t, size_t>> mimo_configs = {
        {2, 2}, {2, 4}, {4, 4}, {4, 8}
    };
    
    PerformanceMeasurement global_perf;
    
    for (const auto& mimo_config : mimo_configs) {
        size_t num_tx = mimo_config.first;
        size_t num_rx = mimo_config.second;
        
        std::cout << "\nTesting " << num_tx << "x" << num_rx << " MIMO:\n";
        
        ExampleConfig config;
        config.num_tx = num_tx;
        config.num_rx = num_rx;
        config.snr_db = 20.0f;
        config.algorithm = MIMOAlgorithm::MMSE;
        config.num_symbols = 1000;
        config.num_iterations = 30;
        config.validate_results = false;
        
        std::string config_name = std::to_string(num_tx) + "x" + std::to_string(num_rx);
        global_perf.start_measurement(config_name);
        int result = run_mimo_example(config);
        global_perf.end_measurement(config_name);
        
        if (result != 0) {
            std::cerr << "Failed to run " << config_name << " MIMO test\n";
            return result;
        }
    }
    
    std::cout << "\n=== MIMO Configuration Comparison Results ===\n";
    global_perf.print_results();
    
    return 0;
}

/**
 * Main function with different example scenarios
 */
int main(int argc, char* argv[]) {
    std::cout << "NVIDIA Aerial Framework - MIMO Detection Pipeline Example\n";
    std::cout << "=========================================================\n\n";
    
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    
    // Example 1: Basic 2x4 MIMO with MMSE detection
    std::cout << "Running Example 1: Basic 2x4 MIMO with MMSE\n";
    ExampleConfig config1;
    config1.num_tx = 2;
    config1.num_rx = 4;
    config1.snr_db = 20.0f;
    config1.algorithm = MIMOAlgorithm::MMSE;
    config1.num_symbols = 10000;
    config1.num_iterations = 100;
    
    int result = run_mimo_example(config1);
    if (result != 0) {
        return result;
    }
    
    // Example 2: High-throughput 4x4 MIMO with CUDA graphs
    std::cout << "\n\nRunning Example 2: High-Throughput 4x4 MIMO\n";
    ExampleConfig config2;
    config2.num_tx = 4;
    config2.num_rx = 4;
    config2.snr_db = 25.0f;
    config2.algorithm = MIMOAlgorithm::ZeroForcing;
    config2.num_symbols = 50000;
    config2.num_iterations = 200;
    config2.use_cuda_graphs = true;
    
    result = run_mimo_example(config2);
    if (result != 0) {
        return result;
    }
    
    // Example 3: Low-SNR Maximum Likelihood detection
    std::cout << "\n\nRunning Example 3: Low-SNR Maximum Likelihood\n";
    ExampleConfig config3;
    config3.num_tx = 2;
    config3.num_rx = 4;
    config3.snr_db = 10.0f;
    config3.algorithm = MIMOAlgorithm::MaximumLikelihood;
    config3.num_symbols = 5000;
    config3.num_iterations = 50;
    config3.validate_results = true;
    
    result = run_mimo_example(config3);
    if (result != 0) {
        return result;
    }
    
    // Comparison tests
    result = run_algorithm_comparison(2, 4, 15.0f);
    if (result != 0) {
        return result;
    }
    
    result = run_mimo_configuration_comparison();
    if (result != 0) {
        return result;
    }
    
    std::cout << "\n\nAll MIMO examples completed successfully!\n";
    return 0;
}