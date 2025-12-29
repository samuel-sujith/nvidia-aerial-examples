/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file fft_processing_example.cpp
 * @brief Complete example demonstrating FFT processing pipeline and GPU acceleration
 *
 * This example shows:
 * 1. How to create and configure an FFT processing module
 * 2. How to set up a GPU-based FFT pipeline with cuFFT
 * 3. How to execute forward and inverse FFT operations
 * 4. OFDM symbol processing and performance optimization
 */

#include <iostream>
#include <vector>
#include <memory>
#include <complex>
#include <random>
#include <chrono>
#include <map>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_module.hpp"
#include "fft_pipeline.hpp"
#include "tensor/tensor_info.hpp"
#include "pipeline/pipeline_spec.hpp"

using namespace fft_processing;

/**
 * Generate synthetic OFDM test signals
 */
class SyntheticOFDMGenerator {
public:
    explicit SyntheticOFDMGenerator(uint32_t seed = 98765) : rng_(seed) {}

    std::vector<std::complex<float>> generate_frequency_domain_signal(size_t fft_size, size_t num_active_carriers) {
        std::vector<std::complex<float>> freq_signal(fft_size, {0.0f, 0.0f});
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        size_t start_carrier = (fft_size - num_active_carriers) / 2;
        
        for (size_t i = 0; i < num_active_carriers; ++i) {
            size_t carrier_idx = start_carrier + i;
            if (carrier_idx < fft_size) {
                float real_part = dist(rng_);
                float imag_part = dist(rng_);
                freq_signal[carrier_idx] = std::complex<float>(real_part, imag_part) / std::sqrt(2.0f);
            }
        }
        
        return freq_signal;
    }

    std::vector<std::complex<float>> generate_time_domain_signal(size_t fft_size, float frequency_bin = 1.0f) {
        std::vector<std::complex<float>> time_signal(fft_size);
        
        for (size_t i = 0; i < fft_size; ++i) {
            float phase = 2.0f * M_PI * frequency_bin * i / fft_size;
            time_signal[i] = std::complex<float>(std::cos(phase), std::sin(phase));
        }
        
        return time_signal;
    }

    std::vector<std::complex<float>> add_cyclic_prefix(const std::vector<std::complex<float>>& ofdm_symbol, size_t cp_length) {
        std::vector<std::complex<float>> symbol_with_cp(ofdm_symbol.size() + cp_length);
        
        // Copy cyclic prefix (last cp_length samples from the symbol)
        for (size_t i = 0; i < cp_length; ++i) {
            symbol_with_cp[i] = ofdm_symbol[ofdm_symbol.size() - cp_length + i];
        }
        
        // Copy the symbol
        for (size_t i = 0; i < ofdm_symbol.size(); ++i) {
            symbol_with_cp[cp_length + i] = ofdm_symbol[i];
        }
        
        return symbol_with_cp;
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
    size_t fft_size{1024};
    size_t num_active_carriers{600};
    size_t cp_length{72};
    int num_iterations{100};
    bool use_cuda_graphs{true};
    FFTPrecision precision{FFTPrecision::Single};
    std::string test_type{"ofdm"};
};

/**
 * FFT validation utility
 */
class FFTValidator {
public:
    static bool validate_fft_pair(const std::vector<std::complex<float>>& original,
                                  const std::vector<std::complex<float>>& reconstructed,
                                  float tolerance = 1e-5f) {
        if (original.size() != reconstructed.size()) {
            return false;
        }
        
        double mse = 0.0;
        for (size_t i = 0; i < original.size(); ++i) {
            auto diff = original[i] - reconstructed[i];
            mse += std::norm(diff);
        }
        mse /= original.size();
        
        return mse < tolerance;
    }
    
    static double calculate_snr(const std::vector<std::complex<float>>& original,
                                const std::vector<std::complex<float>>& reconstructed) {
        double signal_power = 0.0;
        double noise_power = 0.0;
        
        for (size_t i = 0; i < original.size(); ++i) {
            signal_power += std::norm(original[i]);
            auto diff = original[i] - reconstructed[i];
            noise_power += std::norm(diff);
        }
        
        return 10.0 * std::log10(signal_power / noise_power);
    }
};

/**
 * Main example function demonstrating the FFT processing pipeline
 */
int run_fft_example(const ExampleConfig& config) {
    std::cout << "=== FFT Processing Pipeline Example ===\n";
    std::cout << "Configuration:\n";
    std::cout << "  FFT Size: " << config.fft_size << "\n";
    std::cout << "  Active Carriers: " << config.num_active_carriers << "\n";
    std::cout << "  Cyclic Prefix: " << config.cp_length << "\n";
    std::cout << "  Iterations: " << config.num_iterations << "\n";
    std::cout << "  Test Type: " << config.test_type << "\n";
    std::cout << "  CUDA Graphs: " << (config.use_cuda_graphs ? "Enabled" : "Disabled") << "\n\n";

    PerformanceMeasurement perf;

    try {
        // Step 1: Create pipeline configuration
        perf.start_measurement("Pipeline Creation");
        
        FFTPipelineConfig pipeline_config;
        pipeline_config.fft_sizes = {config.fft_size};
        pipeline_config.precision = config.precision;
        pipeline_config.max_batch_size = 32;
        pipeline_config.enable_cuda_graphs = config.use_cuda_graphs;
        
        auto pipeline = FFTPipelineFactory::create_pipeline(pipeline_config);
        
        perf.end_measurement("Pipeline Creation");
        
        // Step 2: Setup pipeline
        aerial::pipeline::PipelineSpec spec;
        if (!pipeline->setup(spec)) {
            std::cerr << "Failed to setup FFT pipeline\n";
            return 1;
        }
        
        // Step 3: Generate test data
        perf.start_measurement("Data Generation");
        
        SyntheticOFDMGenerator ofdm_gen(42);
        std::vector<std::complex<float>> input_signal;
        
        if (config.test_type == "ofdm") {
            input_signal = ofdm_gen.generate_frequency_domain_signal(config.fft_size, config.num_active_carriers);
        } else {
            input_signal = ofdm_gen.generate_time_domain_signal(config.fft_size, 5.0f);
        }
        
        perf.end_measurement("Data Generation");
        
        // Step 4: Setup GPU memory
        perf.start_measurement("GPU Memory Setup");
        
        std::complex<float>* d_input = nullptr;
        std::complex<float>* d_output = nullptr;
        std::complex<float>* d_reconstructed = nullptr;
        
        cudaError_t err = cudaMalloc(&d_input, config.fft_size * sizeof(std::complex<float>));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate input: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        
        err = cudaMalloc(&d_output, config.fft_size * sizeof(std::complex<float>));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate output: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        
        err = cudaMalloc(&d_reconstructed, config.fft_size * sizeof(std::complex<float>));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate reconstructed: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        
        // Copy input data to GPU
        cudaMemcpy(d_input, input_signal.data(), config.fft_size * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
        
        perf.end_measurement("GPU Memory Setup");
        
        // Step 5: Setup tensor info objects
        std::vector<tensor::TensorInfo> inputs(1);
        std::vector<tensor::TensorInfo> outputs(1);
        
        inputs[0].set_data(d_input);
        inputs[0].set_dimensions({config.fft_size});
        inputs[0].set_element_type(tensor::ElementType::COMPLEX_FLOAT32);
        
        outputs[0].set_data(d_output);
        outputs[0].set_dimensions({config.fft_size});
        outputs[0].set_element_type(tensor::ElementType::COMPLEX_FLOAT32);
        
        // Step 6: Warm-up execution - Forward FFT
        std::cout << "Performing warm-up execution (Forward FFT)...\n";
        task::CancellationToken token;
        
        std::vector<std::complex<float>> fft_result;
        auto result = pipeline->execute_forward_fft(input_signal, fft_result, config.fft_size);
        
        if (!result.is_success()) {
            std::cerr << "Forward FFT warm-up failed: " << result.message << "\n";
            return 1;
        }
        
        std::cout << "Forward FFT warm-up completed\n";
        
        // Step 7: Test inverse FFT for reconstruction
        std::cout << "Testing FFT/IFFT pair...\n";
        std::vector<std::complex<float>> reconstructed_signal;
        result = pipeline->execute_inverse_fft(fft_result, reconstructed_signal, config.fft_size);
        
        if (!result.is_success()) {
            std::cerr << "Inverse FFT failed: " << result.message << "\n";
            return 1;
        }
        
        // Step 8: Validate reconstruction
        bool reconstruction_valid = FFTValidator::validate_fft_pair(input_signal, reconstructed_signal);
        double snr_db = FFTValidator::calculate_snr(input_signal, reconstructed_signal);
        
        std::cout << "FFT/IFFT validation: " << (reconstruction_valid ? "PASSED" : "FAILED") << "\n";
        std::cout << "Reconstruction SNR: " << snr_db << " dB\n";
        
        // Step 9: OFDM processing demonstration
        if (config.test_type == "ofdm") {
            std::cout << "Performing OFDM symbol processing...\n";
            
            std::vector<std::complex<float>> ofdm_time_signal;
            perf.start_measurement("OFDM Processing");
            result = pipeline->execute_ofdm_processing(input_signal, ofdm_time_signal, 
                                                     config.fft_size, config.cp_length);
            perf.end_measurement("OFDM Processing");
            
            if (result.is_success()) {
                std::cout << "OFDM processing successful!\n";
                std::cout << "Input frequency symbols: " << input_signal.size() << "\n";
                std::cout << "Output time samples (with CP): " << ofdm_time_signal.size() << "\n";
                
                // Calculate PAPR
                float peak_power = 0.0f;
                float avg_power = 0.0f;
                for (const auto& sample : ofdm_time_signal) {
                    float power = std::norm(sample);
                    peak_power = std::max(peak_power, power);
                    avg_power += power;
                }
                avg_power /= ofdm_time_signal.size();
                
                float papr_db = 10.0f * std::log10(peak_power / avg_power);
                std::cout << "OFDM Symbol PAPR: " << papr_db << " dB\n";
            } else {
                std::cerr << "OFDM processing failed: " << result.message << "\n";
            }
        }
        
        // Step 10: Performance benchmark
        std::cout << "Running performance benchmark (" << config.num_iterations << " iterations)...\n";
        
        for (int i = 0; i < config.num_iterations; ++i) {
            fft_result.clear();
            
            if (config.use_cuda_graphs && i > 0) {
                perf.start_measurement("Pipeline Execution (Graph)");
                result = pipeline->execute_pipeline_graph(inputs, outputs, token);
                perf.end_measurement("Pipeline Execution (Graph)");
            } else {
                perf.start_measurement("Pipeline Execution (Stream)");
                result = pipeline->execute_forward_fft(input_signal, fft_result, config.fft_size);
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
        
        // Step 11: Calculate throughput metrics
        double avg_time_us = perf.get_average_time("Pipeline Execution (Stream)");
        if (config.use_cuda_graphs) {
            avg_time_us = perf.get_average_time("Pipeline Execution (Graph)");
        }
        
        double samples_per_sec = config.fft_size / (avg_time_us / 1000000.0);
        double msamples_per_sec = samples_per_sec / 1000000.0;
        
        std::cout << "\n=== Throughput Analysis ===\n";
        std::cout << "FFT Size: " << config.fft_size << " points\n";
        std::cout << "Processing rate: " << msamples_per_sec << " Msamples/sec\n";
        std::cout << "FFT rate: " << (1000000.0 / avg_time_us) << " FFTs/sec\n";
        
        // Step 12: Print pipeline statistics
        auto pipeline_stats = pipeline->get_fft_stats();
        std::cout << "\n=== Pipeline Statistics ===\n";
        std::cout << "Total FFTs processed: " << pipeline_stats.total_ffts_processed << "\n";
        std::cout << "Average FFT time: " << pipeline_stats.average_latency_us() << " μs\n";
        std::cout << "Peak throughput: " << pipeline_stats.average_throughput_msamples_per_sec() << " Msamples/sec\n";
        
        // Step 13: Cleanup
        perf.start_measurement("Cleanup");
        pipeline->teardown();
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_reconstructed);
        perf.end_measurement("Cleanup");
        
        // Step 14: Print performance results
        perf.print_results();
        
        std::cout << "FFT processing example completed successfully!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << "\n";
        return 1;
    }
}

/**
 * Test different FFT sizes
 */
int run_fft_size_comparison() {
    std::cout << "\n=== FFT Size Comparison ===\n";
    
    std::vector<size_t> fft_sizes = {256, 512, 1024, 2048, 4096};
    PerformanceMeasurement global_perf;
    
    for (size_t fft_size : fft_sizes) {
        std::cout << "\nTesting FFT size " << fft_size << ":\n";
        
        ExampleConfig config;
        config.fft_size = fft_size;
        config.num_active_carriers = fft_size * 3 / 4; // 75% utilization
        config.num_iterations = 50;
        config.test_type = "signal";
        
        std::string size_name = "FFT_" + std::to_string(fft_size);
        global_perf.start_measurement(size_name);
        int result = run_fft_example(config);
        global_perf.end_measurement(size_name);
        
        if (result != 0) {
            std::cerr << "Failed to run FFT size " << fft_size << " test\n";
            return result;
        }
    }
    
    std::cout << "\n=== FFT Size Comparison Results ===\n";
    global_perf.print_results();
    
    return 0;
}

/**
 * Main function with different example scenarios
 */
int main(int argc, char* argv[]) {
    std::cout << "NVIDIA Aerial Framework - FFT Processing Pipeline Example\n";
    std::cout << "=========================================================\n\n";
    
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    
    // Example 1: Basic FFT processing
    std::cout << "Running Example 1: Basic FFT Processing\n";
    ExampleConfig config1;
    config1.fft_size = 1024;
    config1.num_iterations = 100;
    config1.test_type = "signal";
    
    int result = run_fft_example(config1);
    if (result != 0) {
        return result;
    }
    
    // Example 2: OFDM symbol processing with CUDA graphs
    std::cout << "\n\nRunning Example 2: OFDM Symbol Processing\n";
    ExampleConfig config2;
    config2.fft_size = 2048;
    config2.num_active_carriers = 1536;
    config2.cp_length = 144;
    config2.num_iterations = 200;
    config2.test_type = "ofdm";
    config2.use_cuda_graphs = true;
    
    result = run_fft_example(config2);
    if (result != 0) {
        return result;
    }
    
    // Example 3: High precision double precision FFT
    std::cout << "\n\nRunning Example 3: High Precision FFT\n";
    ExampleConfig config3;
    config3.fft_size = 512;
    config3.precision = FFTPrecision::Double;
    config3.num_iterations = 50;
    config3.test_type = "signal";
    
    result = run_fft_example(config3);
    if (result != 0) {
        return result;
    }
    
    // Comparison test
    result = run_fft_size_comparison();
    if (result != 0) {
        return result;
    }
    
    std::cout << "\n\nAll FFT examples completed successfully!\n";
    return 0;
}