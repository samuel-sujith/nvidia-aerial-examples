/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file modulation_mapping_example.cpp
 * @brief Complete example demonstrating modulation mapping pipeline and GPU acceleration
 *
 * This example shows:
 * 1. How to create and configure a modulation mapping module
 * 2. How to set up a GPU-based modulation pipeline
 * 3. How to execute the pipeline with different modulation schemes
 * 4. Performance monitoring and batch processing optimization
 */

#include <iostream>
#include <vector>
#include <memory>
#include <complex>
#include <random>
#include <chrono>
#include <map>

#include <cuda_runtime.h>
#include <curand.h>

#include "modulator.hpp"
#include "modulation_pipeline.hpp"
#include "tensor/tensor_info.hpp"
#include "pipeline/pipeline_spec.hpp"

using namespace modulation_mapping;

/**
 * Generate synthetic bit streams for testing
 */
class SyntheticBitGenerator {
public:
    explicit SyntheticBitGenerator(uint32_t seed = 54321) : rng_(seed) {}

    std::vector<uint8_t> generate_random_bits(size_t num_bits) {
        std::vector<uint8_t> bits(num_bits);
        std::uniform_int_distribution<int> bit_dist(0, 1);
        
        for (size_t i = 0; i < num_bits; ++i) {
            bits[i] = static_cast<uint8_t>(bit_dist(rng_));
        }
        return bits;
    }

    std::vector<uint8_t> generate_pattern_bits(size_t num_bits, const std::string& pattern = "01010101") {
        std::vector<uint8_t> bits(num_bits);
        
        for (size_t i = 0; i < num_bits; ++i) {
            bits[i] = static_cast<uint8_t>(pattern[i % pattern.length()] - '0');
        }
        return bits;
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
 * Main example function demonstrating the modulation mapping pipeline
 */
int run_modulation_example() {
    std::cout << "=== Modulation Mapping Pipeline Example ===\n";

    PerformanceMeasurement perf;

    try {
        // Step 1: Create pipeline configuration
        perf.start_measurement("Pipeline Creation");
        
        ModulationPipelineConfig pipeline_config;
        pipeline_config.modulation_orders = {ModulationOrder::QAM16};
        pipeline_config.max_batch_size = 1024;
        pipeline_config.enable_cuda_graphs = true;
        
        auto pipeline = ModulationPipelineFactory::create_pipeline(pipeline_config);
        
        perf.end_measurement("Pipeline Creation");
        
        // Step 2: Setup pipeline
        aerial::pipeline::PipelineSpec spec;
        if (!pipeline->setup(spec)) {
            std::cerr << "Failed to setup modulation pipeline\n";
            return 1;
        }
        
        // Step 3: Generate test data
        perf.start_measurement("Data Generation");
        
        SyntheticBitGenerator bit_gen(12345);
        size_t num_symbols = 10000;
        size_t bits_per_symbol = 4; // 16-QAM
        size_t total_bits = num_symbols * bits_per_symbol;
        
        auto input_bits = bit_gen.generate_random_bits(total_bits);
        
        perf.end_measurement("Data Generation");
        
        // Step 4: Execute modulation
        std::cout << "Performing modulation mapping...\n";
        
        std::vector<std::complex<float>> output_symbols;
        
        perf.start_measurement("Modulation Execution");
        auto result = pipeline->modulate_bits(input_bits, output_symbols, ModulationOrder::QAM16);
        perf.end_measurement("Modulation Execution");
        
        if (!result.is_success()) {
            std::cerr << "Modulation failed: " << result.message << "\n";
            return 1;
        }
        
        std::cout << "Modulation completed successfully!\n";
        std::cout << "Input bits: " << input_bits.size() << "\n";
        std::cout << "Output symbols: " << output_symbols.size() << "\n";
        
        // Step 5: Sample output display
        std::cout << "\nSample output symbols:\n";
        for (size_t i = 0; i < std::min(size_t(8), output_symbols.size()); ++i) {
            std::cout << "  Symbol[" << i << "] = " << output_symbols[i].real() 
                     << " + " << output_symbols[i].imag() << "j\n";
        }
        
        // Step 6: Print pipeline statistics
        auto pipeline_stats = pipeline->get_modulation_stats();
        std::cout << "\n=== Pipeline Statistics ===\n";
        std::cout << "Total symbols processed: " << pipeline_stats.total_symbols_processed << "\n";
        std::cout << "Average modulation time: " << pipeline_stats.average_modulation_time_us() << " μs\n";
        std::cout << "Peak throughput: " << pipeline_stats.peak_throughput_msymbols_per_sec() << " Msymbols/sec\n";
        
        // Step 7: Cleanup
        pipeline->teardown();
        
        // Step 8: Print performance results
        perf.print_results();
        
        std::cout << "Modulation mapping example completed successfully!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << "\n";
        return 1;
    }
}

/**
 * Main function
 */
int main() {
    std::cout << "NVIDIA Aerial Framework - Modulation Mapping Pipeline Example\n";
    std::cout << "=============================================================\n\n";
    
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    
    return run_modulation_example();
}