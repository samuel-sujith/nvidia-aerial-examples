#include "fft_pipeline.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>

using namespace fft_processing;

/// Generate test sinusoid signal
std::vector<std::complex<float>> generate_test_signal(size_t fft_size, float frequency_bin = 1.0f) {
    std::vector<std::complex<float>> signal(fft_size);
    
    for (size_t i = 0; i < fft_size; ++i) {
        float phase = 2.0f * M_PI * frequency_bin * i / fft_size;
        signal[i] = std::complex<float>(std::cos(phase), std::sin(phase));
    }
    
    return signal;
}

/// Generate OFDM-like test data
std::vector<std::complex<float>> generate_ofdm_data(size_t fft_size, size_t num_active_subcarriers) {
    std::vector<std::complex<float>> freq_data(fft_size, {0.0f, 0.0f});
    
    // Place data in center subcarriers
    size_t start_sc = (fft_size - num_active_subcarriers) / 2;
    
    for (size_t i = 0; i < num_active_subcarriers; ++i) {
        size_t sc_idx = start_sc + i;
        // Generate random QPSK symbols
        float real_part = (i % 2 == 0) ? 1.0f : -1.0f;
        float imag_part = ((i/2) % 2 == 0) ? 1.0f : -1.0f;
        freq_data[sc_idx] = std::complex<float>(real_part / std::sqrt(2.0f), imag_part / std::sqrt(2.0f));
    }
    
    return freq_data;
}

/// Demonstrate basic FFT operations
void demonstrate_basic_fft() {
    std::cout << "=== Basic FFT Operations Demo ===\n";
    
    // Create FFT pipeline
    auto config = FFTPipelineFactory::get_default_config({1024});
    auto pipeline = FFTPipelineFactory::create_pipeline(config);
    
    // Setup pipeline
    aerial::pipeline::PipelineSpec spec;
    if (!pipeline->setup(spec)) {
        std::cerr << "Failed to setup FFT pipeline\n";
        return;
    }
    
    const size_t fft_size = 1024;
    
    // Test forward FFT
    std::cout << "\nTesting Forward FFT:\n";
    auto time_signal = generate_test_signal(fft_size, 10.0f); // 10 frequency bins
    std::vector<std::complex<float>> freq_result;
    
    auto result = pipeline->execute_forward_fft(time_signal, freq_result, fft_size);
    
    if (result.is_success()) {
        std::cout << "Forward FFT successful!\n";
        std::cout << "Input size: " << time_signal.size() << "\n";
        std::cout << "Output size: " << freq_result.size() << "\n";
        
        // Find peak frequency bin
        size_t peak_bin = 0;
        float peak_magnitude = 0.0f;
        for (size_t i = 0; i < freq_result.size(); ++i) {
            float magnitude = std::abs(freq_result[i]);
            if (magnitude > peak_magnitude) {
                peak_magnitude = magnitude;
                peak_bin = i;
            }
        }
        std::cout << "Peak at frequency bin: " << peak_bin << " (magnitude: " << peak_magnitude << ")\n";
    } else {
        std::cerr << "Forward FFT failed: " << result.message << "\n";
    }
    
    // Test inverse FFT
    std::cout << "\nTesting Inverse FFT:\n";
    std::vector<std::complex<float>> reconstructed_signal;
    
    result = pipeline->execute_inverse_fft(freq_result, reconstructed_signal, fft_size);
    
    if (result.is_success()) {
        std::cout << "Inverse FFT successful!\n";
        
        // Check reconstruction error
        double mse = 0.0;
        for (size_t i = 0; i < std::min(time_signal.size(), reconstructed_signal.size()); ++i) {
            auto diff = time_signal[i] - reconstructed_signal[i];
            mse += std::norm(diff);
        }
        mse /= time_signal.size();
        
        std::cout << "Reconstruction MSE: " << mse << "\n";
        
        if (mse < 1e-6) {
            std::cout << "Perfect reconstruction achieved!\n";
        }
    } else {
        std::cerr << "Inverse FFT failed: " << result.message << "\n";
    }
    
    // Show performance statistics
    auto stats = pipeline->get_fft_stats();
    std::cout << "\nPerformance Statistics:\n";
    std::cout << "Total FFTs processed: " << stats.total_ffts_processed << "\n";
    std::cout << "Average throughput: " << stats.average_throughput_msamples_per_sec() << " Msamples/sec\n";
    std::cout << "Average latency: " << stats.average_latency_us() << " μs\n";
    
    pipeline->teardown();
    std::cout << "\n";
}

/// Demonstrate batch FFT processing
void demonstrate_batch_fft() {
    std::cout << "=== Batch FFT Processing Demo ===\n";
    
    // High performance configuration
    auto config = FFTPipelineFactory::get_high_performance_config({512, 1024, 2048});
    auto pipeline = FFTPipelineFactory::create_pipeline(config);
    
    aerial::pipeline::PipelineSpec spec;
    if (!pipeline->setup(spec)) {
        std::cerr << "Failed to setup pipeline\n";
        return;
    }
    
    // Generate batch of different size signals
    std::vector<std::vector<std::complex<float>>> input_batches;
    std::vector<std::vector<std::complex<float>>> output_batches;
    std::vector<size_t> fft_sizes = {512, 1024, 512, 2048, 1024};
    
    for (size_t i = 0; i < fft_sizes.size(); ++i) {
        size_t fft_size = fft_sizes[i];
        auto signal = generate_test_signal(fft_size, 5.0f + i * 2.0f);
        input_batches.push_back(signal);
    }
    
    output_batches.resize(input_batches.size());
    
    // Process mixed batch
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto result = pipeline->execute_mixed_batch(input_batches, output_batches, fft_sizes, FFTType::Forward);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (result.is_success()) {
        size_t total_samples = 0;
        for (const auto& batch : output_batches) {
            total_samples += batch.size();
        }
        
        double throughput = static_cast<double>(total_samples) / duration.count();
        
        std::cout << "Batch processing successful!\n";
        std::cout << "Processed " << input_batches.size() << " FFTs\n";
        std::cout << "Total samples: " << total_samples << "\n";
        std::cout << "Processing time: " << duration.count() << " μs\n";
        std::cout << "Throughput: " << throughput << " Msamples/sec\n";
        
    } else {
        std::cerr << "Batch processing failed: " << result.message << "\n";
    }
    
    pipeline->teardown();
    std::cout << "\n";
}

/// Demonstrate OFDM processing
void demonstrate_ofdm_processing() {
    std::cout << "=== OFDM Processing Demo ===\n";
    
    // OFDM-specific configuration
    const size_t subcarriers = 1024;
    const size_t cp_length = 72; // Cyclic prefix length
    const size_t active_subcarriers = 600;
    
    auto config = FFTPipelineFactory::get_ofdm_config(subcarriers);
    auto pipeline = FFTPipelineFactory::create_pipeline(config);
    
    aerial::pipeline::PipelineSpec spec;
    if (!pipeline->setup(spec)) {
        std::cerr << "Failed to setup OFDM pipeline\n";
        return;
    }
    
    // Generate OFDM frequency domain symbols
    auto freq_symbols = generate_ofdm_data(subcarriers, active_subcarriers);
    std::vector<std::complex<float>> time_samples;
    
    std::cout << "Processing OFDM symbol:\n";
    std::cout << "FFT size: " << subcarriers << "\n";
    std::cout << "Active subcarriers: " << active_subcarriers << "\n";
    std::cout << "Cyclic prefix length: " << cp_length << "\n";
    
    // Execute OFDM processing (IFFT + CP addition)
    auto result = pipeline->execute_ofdm_processing(freq_symbols, time_samples, subcarriers, cp_length);
    
    if (result.is_success()) {
        std::cout << "OFDM processing successful!\n";
        std::cout << "Input symbols: " << freq_symbols.size() << "\n";
        std::cout << "Output samples: " << time_samples.size() << "\n";
        std::cout << "Expected output size: " << (subcarriers + cp_length) << "\n";
        
        // Calculate PAPR (Peak-to-Average Power Ratio)
        float peak_power = 0.0f;
        float avg_power = 0.0f;
        
        for (const auto& sample : time_samples) {
            float power = std::norm(sample);
            peak_power = std::max(peak_power, power);
            avg_power += power;
        }
        avg_power /= time_samples.size();
        
        float papr_db = 10.0f * std::log10(peak_power / avg_power);
        std::cout << "PAPR: " << papr_db << " dB\n";
        
    } else {
        std::cerr << "OFDM processing failed: " << result.message << "\n";
    }
    
    pipeline->teardown();
    std::cout << "\n";
}

/// Benchmark different FFT sizes
void demonstrate_fft_benchmarks() {
    std::cout << "=== FFT Size Benchmarks ===\n";
    
    std::vector<size_t> fft_sizes = {128, 256, 512, 1024, 2048, 4096, 8192};
    const int num_iterations = 20;
    const size_t batch_size = 16;
    
    for (size_t fft_size : fft_sizes) {
        std::cout << "\nBenchmarking FFT size " << fft_size << ":\n";
        
        auto config = FFTPipelineFactory::get_high_performance_config({fft_size});
        auto pipeline = FFTPipelineFactory::create_pipeline(config);
        
        aerial::pipeline::PipelineSpec spec;
        if (!pipeline->setup(spec)) {
            std::cerr << "  Failed to setup pipeline\n";
            continue;
        }
        
        // Generate test data
        auto test_signal = generate_test_signal(fft_size * batch_size);
        std::vector<std::complex<float>> fft_result;
        
        // Warmup
        pipeline->execute_forward_fft(test_signal, fft_result, fft_size, batch_size);
        
        // Benchmark
        std::vector<double> execution_times;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            fft_result.clear();
            
            auto start_time = std::chrono::high_resolution_clock::now();
            auto result = pipeline->execute_forward_fft(test_signal, fft_result, fft_size, batch_size);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            if (result.is_success()) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                execution_times.push_back(duration.count());
            }
        }
        
        if (!execution_times.empty()) {
            double avg_time = 0.0;
            double min_time = execution_times[0];
            
            for (double time : execution_times) {
                avg_time += time;
                min_time = std::min(min_time, time);
            }
            avg_time /= execution_times.size();
            
            double throughput = static_cast<double>(fft_size * batch_size) / avg_time;
            double efficiency = throughput / (fft_size * std::log2(fft_size)); // Normalized by theoretical complexity
            
            std::cout << "  Average time: " << avg_time << " μs\n";
            std::cout << "  Min time: " << min_time << " μs\n";
            std::cout << "  Throughput: " << throughput << " Msamples/sec\n";
            std::cout << "  Efficiency: " << efficiency << " (normalized)\n";
        }
        
        pipeline->teardown();
    }
    
    std::cout << "\n";
}

/// Demonstrate precision comparison
void demonstrate_precision_modes() {
    std::cout << "=== Precision Modes Demo ===\n";
    
    const size_t fft_size = 1024;
    auto test_signal = generate_test_signal(fft_size, 7.5f);
    
    std::vector<FFTPrecision> precisions = {FFTPrecision::Single, FFTPrecision::Double};
    std::vector<std::string> precision_names = {"Single", "Double"};
    
    for (size_t i = 0; i < precisions.size(); ++i) {
        std::cout << "\nTesting " << precision_names[i] << " precision:\n";
        
        FFTPipelineConfig config;
        config.fft_sizes = {fft_size};
        config.precision = precisions[i];
        config.max_batch_size = 32;
        
        auto pipeline = FFTPipelineFactory::create_pipeline(config);
        
        aerial::pipeline::PipelineSpec spec;
        if (!pipeline->setup(spec)) {
            std::cerr << "  Failed to setup pipeline\n";
            continue;
        }
        
        // Forward FFT
        std::vector<std::complex<float>> freq_result;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = pipeline->execute_forward_fft(test_signal, freq_result, fft_size);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (result.is_success()) {
            // Inverse FFT for error measurement
            std::vector<std::complex<float>> reconstructed;
            pipeline->execute_inverse_fft(freq_result, reconstructed, fft_size);
            
            // Calculate reconstruction error
            double mse = 0.0;
            for (size_t j = 0; j < test_signal.size(); ++j) {
                auto diff = test_signal[j] - reconstructed[j];
                mse += std::norm(diff);
            }
            mse /= test_signal.size();
            
            double throughput = static_cast<double>(fft_size) / duration.count();
            
            std::cout << "  Processing time: " << duration.count() << " μs\n";
            std::cout << "  Throughput: " << throughput << " Msamples/sec\n";
            std::cout << "  Reconstruction MSE: " << mse << "\n";
            std::cout << "  Numerical accuracy: " << (-10.0 * std::log10(mse)) << " dB\n";
        } else {
            std::cerr << "  FFT failed: " << result.message << "\n";
        }
        
        pipeline->teardown();
    }
    
    std::cout << "\n";
}

int main() {
    try {
        std::cout << "NVIDIA Aerial Framework - FFT Pipeline Examples\n";
        std::cout << "==============================================\n\n";
        
        demonstrate_basic_fft();
        demonstrate_batch_fft();
        demonstrate_ofdm_processing();
        demonstrate_fft_benchmarks();
        demonstrate_precision_modes();
        
        std::cout << "All FFT demos completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}