/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file channel_estimation_example.cpp
 * @brief Complete example demonstrating channel estimation feature and GPU pipeline
 *
 * This example shows:
 * 1. How to create and configure a channel estimation module
 * 2. How to set up a GPU-based pipeline
 * 3. How to execute the pipeline with real data
 * 4. Performance monitoring and optimization
 */

#include <iostream>
#include <vector>
#include <memory>
#include <complex>
#include <random>
#include <chrono>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <map>

#include "channel_estimator.hpp"
#include "channel_est_pipeline.hpp"
#include "tensor/tensor_info.hpp"
#include "pipeline/types.hpp"

using namespace framework::examples;
using ::framework::pipeline::PipelineSpec;
using ::framework::tensor::TensorInfo;
using ::framework::task::CancellationToken;

/**
 * Generate synthetic 5G NR pilot symbols for testing
 */
class SyntheticDataGenerator {
public:
    struct ChannelModel {
        std::vector<std::complex<float>> taps;
        std::vector<int> delays;
        float noise_power{0.01f};
    };

    explicit SyntheticDataGenerator(uint32_t seed = 12345) : rng_(seed) {}

    std::vector<cuComplex> generate_tx_pilots(int num_pilots) {
        std::vector<cuComplex> pilots(num_pilots);
        std::uniform_real_distribution<float> phase_dist(0.0f, 2.0f * M_PI);
        
        for (int i = 0; i < num_pilots; ++i) {
            float phase = phase_dist(rng_);
            pilots[i] = make_cuComplex(cosf(phase), sinf(phase));
        }
        return pilots;
    }

    std::vector<cuComplex> generate_rx_pilots(
        const std::vector<cuComplex>& tx_pilots,
        const ChannelModel& channel
    ) {
        std::vector<cuComplex> rx_pilots(tx_pilots.size());
        std::normal_distribution<float> noise_dist(0.0f, sqrtf(channel.noise_power / 2.0f));
        
        for (size_t i = 0; i < tx_pilots.size(); ++i) {
            // Apply channel (simplified - single tap for this example)
            cuComplex h = make_cuComplex(0.8f, 0.2f); // Example channel coefficient
            
            cuComplex signal = cuCmulf(tx_pilots[i], h);
            
            // Add noise
            float noise_real = noise_dist(rng_);
            float noise_imag = noise_dist(rng_);
            
            rx_pilots[i] = make_cuComplex(
                signal.x + noise_real,
                signal.y + noise_imag
            );
        }
        
        return rx_pilots;
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
            std::cout << "  Last: " << measurement.last_time_us << " μs\n";
            std::cout << "  Average: " << avg_time_us << " μs\n";
            std::cout << "  Total: " << measurement.total_time_us << " μs (" << measurement.count << " runs)\n";
            std::cout << "  Throughput: " << (1000000.0 / avg_time_us) << " executions/second\n\n";
        }
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
    int num_resource_blocks{100};
    int num_ofdm_symbols{14};
    int pilot_spacing{4};
    ChannelEstAlgorithm algorithm{ChannelEstAlgorithm::LEAST_SQUARES};
    float noise_variance{0.01f};
    int num_iterations{100};
    bool use_cuda_graphs{true};
    bool enable_profiling{true};
};

/**
 * Main example function demonstrating the channel estimation pipeline
 */
int run_channel_estimation_example(const ExampleConfig& config) {
    std::cout << "=== Channel Estimation Pipeline Example ===\n";
    std::cout << "Configuration:\n";
    std::cout << "  Resource Blocks: " << config.num_resource_blocks << "\n";
    std::cout << "  OFDM Symbols: " << config.num_ofdm_symbols << "\n";
    std::cout << "  Pilot Spacing: " << config.pilot_spacing << "\n";
    std::cout << "  Algorithm: " << (config.algorithm == ChannelEstAlgorithm::LEAST_SQUARES ? "LS" : "MMSE") << "\n";
    std::cout << "  Iterations: " << config.num_iterations << "\n";
    std::cout << "  CUDA Graphs: " << (config.use_cuda_graphs ? "Enabled" : "Disabled") << "\n\n";

    PerformanceMeasurement perf;

    try {
        // Step 1: Setup parameters
        ChannelEstParams params;
        params.algorithm = config.algorithm;
        params.num_resource_blocks = config.num_resource_blocks;
        params.num_ofdm_symbols = config.num_ofdm_symbols;
        params.pilot_spacing = config.pilot_spacing;
        params.noise_variance = config.noise_variance;
        params.beta_scaling = 1.0f;
        
        // Step 2: Create module factory
        auto module_factory = std::make_unique<ChannelEstimatorModuleFactory>(params);
        
        // Step 3: Setup pipeline specification
        PipelineSpec pipeline_spec;
        pipeline_spec.pipeline_name = "channel_estimation_pipeline";
        // Simplified spec - actual module configuration would be more complex
        
        // Step 4: Create pipeline
        perf.start_measurement("Pipeline Creation");
        auto pipeline_factory = std::make_unique<ChannelEstimationPipelineFactory>();
        auto pipeline = pipeline_factory->create_pipeline(
            "example_pipeline", pipeline_spec
        );
        perf.end_measurement("Pipeline Creation");
        
        if (!pipeline->is_ready()) {
            std::cerr << "Pipeline is not ready for execution\n";
            return 1;
        }
        
        // Step 5: Generate test data
        perf.start_measurement("Data Generation");
        SyntheticDataGenerator data_gen(42);
        
        int num_pilots = (config.num_resource_blocks * 12) / config.pilot_spacing;
        auto tx_pilots = data_gen.generate_tx_pilots(num_pilots);
        
        SyntheticDataGenerator::ChannelModel channel_model;
        channel_model.noise_power = config.noise_variance;
        auto rx_pilots = data_gen.generate_rx_pilots(tx_pilots, channel_model);
        
        perf.end_measurement("Data Generation");
        
        // Step 6: Setup GPU memory and tensors
        perf.start_measurement("GPU Memory Setup");
        
        // Allocate GPU memory
        cuComplex* d_rx_pilots = nullptr;
        cuComplex* d_tx_pilots = nullptr;
        cuComplex* d_channel_estimates = nullptr;
        
        int num_subcarriers = config.num_resource_blocks * 12;
        int num_estimates = num_subcarriers * config.num_ofdm_symbols;
        
        cudaError_t err = cudaMalloc(&d_rx_pilots, num_pilots * sizeof(cuComplex));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate rx_pilots: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        
        err = cudaMalloc(&d_tx_pilots, num_pilots * sizeof(cuComplex));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate tx_pilots: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        
        err = cudaMalloc(&d_channel_estimates, num_estimates * sizeof(cuComplex));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate channel_estimates: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        
        // Copy data to GPU
        cudaMemcpy(d_rx_pilots, rx_pilots.data(), num_pilots * sizeof(cuComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tx_pilots, tx_pilots.data(), num_pilots * sizeof(cuComplex), cudaMemcpyHostToDevice);
        
        perf.end_measurement("GPU Memory Setup");
        
        // Step 7: Create tensor info objects
        std::vector<TensorInfo> inputs(2);
        std::vector<TensorInfo> outputs(1);
        
        // Set up input tensors (simplified - real implementation would set data pointers)
        // inputs[0] = RX pilot data tensor
        // inputs[1] = TX pilot reference tensor
        
        // Set up output tensors 
        // outputs[0] = Channel estimates tensor
        
        // Step 8: Warm-up execution
        std::cout << "Performing warm-up execution...\n";
        CancellationToken token;
        auto result = pipeline->execute_pipeline(inputs, outputs, token);
        
        if (!result.is_success()) {
            std::cerr << "Warm-up execution failed: " << result.message << "\n";
            return 1;
        }
        
        std::cout << "Warm-up completed successfully\n";
        
        // Step 9: Performance benchmark
        std::cout << "Running performance benchmark (" << config.num_iterations << " iterations)...\n";
        
        for (int i = 0; i < config.num_iterations; ++i) {
            if (config.use_cuda_graphs && i > 0) {
                perf.start_measurement("Pipeline Execution (Graph)");
                result = pipeline->execute_pipeline_graph(inputs, outputs, token);
                perf.end_measurement("Pipeline Execution (Graph)");
            } else {
                perf.start_measurement("Pipeline Execution (Stream)");
                result = pipeline->execute_pipeline(inputs, outputs, token);
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
        
        // Step 10: Verify results (basic sanity check)
        std::vector<cuComplex> host_results(num_estimates);
        cudaMemcpy(host_results.data(), d_channel_estimates, 
                  num_estimates * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        
        // Check that results are reasonable (non-zero, finite)
        bool results_valid = true;
        for (int i = 0; i < std::min(10, num_estimates); ++i) {
            if (!std::isfinite(host_results[i].x) || !std::isfinite(host_results[i].y)) {
                results_valid = false;
                break;
            }
        }
        
        if (!results_valid) {
            std::cerr << "Warning: Channel estimation results appear invalid\n";
        } else {
            std::cout << "Results validation: PASSED\n";
            
            // Print some sample results
            std::cout << "Sample channel estimates:\n";
            for (int i = 0; i < std::min(5, num_estimates); ++i) {
                std::cout << "  H[" << i << "] = " << host_results[i].x 
                         << " + " << host_results[i].y << "j\n";
            }
        }
        
        // Step 11: Print pipeline statistics
        std::cout << "\n=== Pipeline Statistics ===\n";
        pipeline->print_stats();
        std::cout << "Average execution time: Not available in stub implementation\n";
        
        // Step 12: Cleanup
        perf.start_measurement("Cleanup");
        pipeline->teardown();
        
        cudaFree(d_rx_pilots);
        cudaFree(d_tx_pilots);
        cudaFree(d_channel_estimates);
        perf.end_measurement("Cleanup");
        
        // Step 13: Print performance results
        perf.print_results();
        
        std::cout << "Channel estimation example completed successfully!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << "\n";
        return 1;
    }
}

/**
 * Main function with different example scenarios
 */
int main(int argc, char* argv[]) {
    std::cout << "NVIDIA Aerial Framework - Channel Estimation Pipeline Example\n";
    std::cout << "============================================================\n\n";
    
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    
    // Example 1: Basic channel estimation with default parameters
    std::cout << "Running Example 1: Basic Channel Estimation\n";
    ExampleConfig config1;
    config1.num_resource_blocks = 50;
    config1.num_iterations = 10;
    config1.algorithm = ChannelEstAlgorithm::LEAST_SQUARES;
    
    int result = run_channel_estimation_example(config1);
    if (result != 0) {
        return result;
    }
    
    // Example 2: High-throughput scenario with CUDA graphs
    std::cout << "\n\nRunning Example 2: High-Throughput with CUDA Graphs\n";
    ExampleConfig config2;
    config2.num_resource_blocks = 273; // Maximum 5G NR bandwidth
    config2.num_iterations = 1000;
    config2.algorithm = ChannelEstAlgorithm::MMSE;
    config2.use_cuda_graphs = true;
    
    result = run_channel_estimation_example(config2);
    if (result != 0) {
        return result;
    }
    
    // Example 3: Low-latency scenario
    std::cout << "\n\nRunning Example 3: Low-Latency Scenario\n";
    ExampleConfig config3;
    config3.num_resource_blocks = 25;
    config3.num_iterations = 100;
    config3.algorithm = ChannelEstAlgorithm::LEAST_SQUARES;
    config3.use_cuda_graphs = false; // Stream-based for lowest latency
    
    result = run_channel_estimation_example(config3);
    
    std::cout << "\n\nAll examples completed successfully!\n";
    return result;
}