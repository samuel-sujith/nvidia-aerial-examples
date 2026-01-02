#include "mimo_pipeline.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace mimo_detection;

/// Generate random channel matrix
std::vector<std::vector<std::complex<float>>> generate_channel_matrix(
    size_t num_rx, size_t num_tx, float snr_db = 20.0f) {
    
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<std::vector<std::complex<float>>> channel(num_rx, std::vector<std::complex<float>>(num_tx));
    
    for (size_t i = 0; i < num_rx; ++i) {
        for (size_t j = 0; j < num_tx; ++j) {
            float real_part = dist(gen);
            float imag_part = dist(gen);
            // Rayleigh fading channel
            channel[i][j] = std::complex<float>(real_part, imag_part) / std::sqrt(2.0f);
        }
    }
    
    return channel;
}

/// Generate transmitted symbols (QPSK modulation)
std::vector<std::complex<float>> generate_transmitted_symbols(
    size_t num_symbols, size_t num_tx) {
    
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dist(0, 3);
    
    std::vector<std::complex<float>> symbols(num_symbols * num_tx);
    std::vector<std::complex<float>> qpsk_constellation = {
        {1.0f / std::sqrt(2.0f), 1.0f / std::sqrt(2.0f)},   // 00
        {1.0f / std::sqrt(2.0f), -1.0f / std::sqrt(2.0f)},  // 01
        {-1.0f / std::sqrt(2.0f), 1.0f / std::sqrt(2.0f)},  // 10
        {-1.0f / std::sqrt(2.0f), -1.0f / std::sqrt(2.0f)}  // 11
    };
    
    for (size_t i = 0; i < num_symbols; ++i) {
        for (size_t tx = 0; tx < num_tx; ++tx) {
            int symbol_idx = dist(gen);
            symbols[i * num_tx + tx] = qpsk_constellation[symbol_idx];
        }
    }
    
    return symbols;
}

/// Apply channel and add noise to create received signal
std::vector<std::complex<float>> apply_channel_and_noise(
    const std::vector<std::complex<float>>& tx_symbols,
    const std::vector<std::vector<std::complex<float>>>& channel,
    size_t num_symbols, size_t num_tx, size_t num_rx,
    float snr_db = 20.0f) {
    
    std::mt19937 gen(456);
    float noise_var = std::pow(10.0f, -snr_db / 10.0f) / 2.0f; // Per dimension
    std::normal_distribution<float> noise_dist(0.0f, std::sqrt(noise_var));
    
    std::vector<std::complex<float>> rx_signals(num_symbols * num_rx);
    
    for (size_t sym = 0; sym < num_symbols; ++sym) {
        for (size_t rx = 0; rx < num_rx; ++rx) {
            std::complex<float> signal(0.0f, 0.0f);
            
            // Apply channel
            for (size_t tx = 0; tx < num_tx; ++tx) {
                signal += channel[rx][tx] * tx_symbols[sym * num_tx + tx];
            }
            
            // Add noise
            signal += std::complex<float>(noise_dist(gen), noise_dist(gen));
            
            rx_signals[sym * num_rx + rx] = signal;
        }
    }
    
    return rx_signals;
}

/// Calculate Symbol Error Rate
double calculate_ser(const std::vector<std::complex<float>>& original,
                     const std::vector<std::complex<float>>& detected) {
    if (original.size() != detected.size()) return 1.0;
    
    size_t errors = 0;
    const float threshold = 0.5f; // Threshold for QPSK detection
    
    for (size_t i = 0; i < original.size(); ++i) {
        // Map symbols to bits and compare
        auto orig_bits = std::make_pair(
            original[i].real() > 0 ? 1 : 0,
            original[i].imag() > 0 ? 1 : 0
        );
        auto det_bits = std::make_pair(
            detected[i].real() > 0 ? 1 : 0,
            detected[i].imag() > 0 ? 1 : 0
        );
        
        if (orig_bits != det_bits) {
            errors++;
        }
    }
    
    return static_cast<double>(errors) / original.size();
}

/// Demonstrate basic MIMO detection
void demonstrate_basic_mimo_detection() {
    std::cout << "=== Basic MIMO Detection Demo ===\n";
    
    // MIMO configuration
    const size_t num_tx = 2;
    const size_t num_rx = 4;
    const size_t num_symbols = 1000;
    const float snr_db = 15.0f;
    
    std::cout << "Configuration: " << num_tx << "x" << num_rx << " MIMO\n";
    std::cout << "Number of symbols: " << num_symbols << "\n";
    std::cout << "SNR: " << snr_db << " dB\n\n";
    
    // Create MIMO pipeline
    auto config = MIMOPipelineFactory::get_default_config(num_tx, num_rx);
    auto pipeline = MIMOPipelineFactory::create_pipeline(config);
    
    ::framework::pipeline::PipelineSpec spec;
    if (!pipeline->setup(spec)) {
        std::cerr << "Failed to setup MIMO pipeline\n";
        return;
    }
    
    // Generate test data
    auto channel = generate_channel_matrix(num_rx, num_tx, snr_db);
    auto tx_symbols = generate_transmitted_symbols(num_symbols, num_tx);
    auto rx_signals = apply_channel_and_noise(tx_symbols, channel, 
                                             num_symbols, num_tx, num_rx, snr_db);
    
    // Test different detection algorithms
    std::vector<MIMOAlgorithm> algorithms = {
        MIMOAlgorithm::ZeroForcing,
        MIMOAlgorithm::MMSE,
        MIMOAlgorithm::MaximumLikelihood
    };
    
    std::vector<std::string> algorithm_names = {"Zero Forcing", "MMSE", "Maximum Likelihood"};
    
    for (size_t i = 0; i < algorithms.size(); ++i) {
        std::cout << "Testing " << algorithm_names[i] << " detection:\n";
        
        std::vector<std::complex<float>> detected_symbols;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto result = pipeline->detect_symbols(rx_signals, channel, 
                                              detected_symbols, algorithms[i]);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (result.is_success()) {
            double ser = calculate_ser(tx_symbols, detected_symbols);
            double throughput = static_cast<double>(num_symbols * num_tx) / duration.count();
            
            std::cout << "  Detection successful!\n";
            std::cout << "  Processing time: " << duration.count() << " μs\n";
            std::cout << "  Throughput: " << throughput << " Msymbols/sec\n";
            std::cout << "  Symbol Error Rate: " << std::scientific << std::setprecision(2) 
                     << ser << " (" << (ser * 100) << "%)\n";
            
        } else {
            std::cerr << "  Detection failed: " << result.message << "\n";
        }
        std::cout << "\n";
    }
    
    // Show overall statistics
    auto stats = pipeline->get_mimo_stats();
    std::cout << "Overall Statistics:\n";
    std::cout << "Total symbols detected: " << stats.total_symbols_detected << "\n";
    std::cout << "Average detection time: " << stats.average_detection_time_us() << " μs\n";
    std::cout << "Average throughput: " << stats.average_throughput_msymbols_per_sec() << " Msymbols/sec\n";
    
    pipeline->teardown();
    std::cout << "\n";
}

/// Demonstrate batch processing
void demonstrate_batch_mimo_processing() {
    std::cout << "=== Batch MIMO Processing Demo ===\n";
    
    const size_t num_tx = 2;
    const size_t num_rx = 2;
    const size_t symbols_per_batch = 256;
    const size_t num_batches = 8;
    const float snr_db = 20.0f;
    
    std::cout << "Batch configuration:\n";
    std::cout << "MIMO: " << num_tx << "x" << num_rx << "\n";
    std::cout << "Symbols per batch: " << symbols_per_batch << "\n";
    std::cout << "Number of batches: " << num_batches << "\n";
    std::cout << "SNR: " << snr_db << " dB\n\n";
    
    // High-performance configuration
    auto config = MIMOPipelineFactory::get_high_performance_config(num_tx, num_rx);
    auto pipeline = MIMOPipelineFactory::create_pipeline(config);
    
    ::framework::pipeline::PipelineSpec spec;
    if (!pipeline->setup(spec)) {
        std::cerr << "Failed to setup batch MIMO pipeline\n";
        return;
    }
    
    // Generate batch data
    std::vector<std::vector<std::complex<float>>> rx_batches;
    std::vector<std::vector<std::vector<std::complex<float>>>> channel_batches;
    std::vector<std::vector<std::complex<float>>> tx_batches; // For SER calculation
    
    for (size_t batch = 0; batch < num_batches; ++batch) {
        auto channel = generate_channel_matrix(num_rx, num_tx, snr_db);
        auto tx_symbols = generate_transmitted_symbols(symbols_per_batch, num_tx);
        auto rx_signals = apply_channel_and_noise(tx_symbols, channel,
                                                 symbols_per_batch, num_tx, num_rx, snr_db);
        
        rx_batches.push_back(rx_signals);
        channel_batches.push_back(channel);
        tx_batches.push_back(tx_symbols);
    }
    
    // Process batch
    std::vector<std::vector<std::complex<float>>> detected_batches;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto result = pipeline->detect_batch_symbols(rx_batches, channel_batches,
                                               detected_batches, MIMOAlgorithm::MMSE);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (result.is_success()) {
        size_t total_symbols = num_batches * symbols_per_batch * num_tx;
        double throughput = static_cast<double>(total_symbols) / (duration.count() * 1000);
        
        // Calculate average SER across batches
        double total_ser = 0.0;
        for (size_t batch = 0; batch < num_batches; ++batch) {
            total_ser += calculate_ser(tx_batches[batch], detected_batches[batch]);
        }
        double avg_ser = total_ser / num_batches;
        
        std::cout << "Batch processing successful!\n";
        std::cout << "Total symbols: " << total_symbols << "\n";
        std::cout << "Processing time: " << duration.count() << " ms\n";
        std::cout << "Throughput: " << throughput << " Msymbols/sec\n";
        std::cout << "Average SER: " << std::scientific << std::setprecision(2) 
                 << avg_ser << "\n";
        
    } else {
        std::cerr << "Batch processing failed: " << result.message << "\n";
    }
    
    pipeline->teardown();
    std::cout << "\n";
}

/// Demonstrate different MIMO configurations
void demonstrate_mimo_configurations() {
    std::cout << "=== MIMO Configuration Comparison ===\n";
    
    std::vector<std::pair<size_t, size_t>> mimo_configs = {
        {2, 2}, {2, 4}, {4, 4}, {4, 8}, {8, 8}
    };
    
    const size_t num_symbols = 500;
    const float snr_db = 18.0f;
    
    for (const auto& config_pair : mimo_configs) {
        size_t num_tx = config_pair.first;
        size_t num_rx = config_pair.second;
        
        std::cout << "\nTesting " << num_tx << "x" << num_rx << " MIMO:\n";
        
        auto config = MIMOPipelineFactory::get_default_config(num_tx, num_rx);
        auto pipeline = MIMOPipelineFactory::create_pipeline(config);
        
        ::framework::pipeline::PipelineSpec spec;
        if (!pipeline->setup(spec)) {
            std::cerr << "  Failed to setup pipeline\n";
            continue;
        }
        
        // Generate test data
        auto channel = generate_channel_matrix(num_rx, num_tx, snr_db);
        auto tx_symbols = generate_transmitted_symbols(num_symbols, num_tx);
        auto rx_signals = apply_channel_and_noise(tx_symbols, channel,
                                                 num_symbols, num_tx, num_rx, snr_db);
        
        // Test MMSE detection
        std::vector<std::complex<float>> detected_symbols;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = pipeline->detect_symbols(rx_signals, channel,
                                              detected_symbols, MIMOAlgorithm::MMSE);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (result.is_success()) {
            double ser = calculate_ser(tx_symbols, detected_symbols);
            double throughput = static_cast<double>(num_symbols * num_tx) / duration.count();
            
            // Calculate diversity order (approximation)
            size_t diversity_order = num_rx - num_tx + 1;
            
            std::cout << "  Processing time: " << duration.count() << " μs\n";
            std::cout << "  Throughput: " << std::fixed << std::setprecision(1) 
                     << throughput << " Msymbols/sec\n";
            std::cout << "  Symbol Error Rate: " << std::scientific << std::setprecision(2) 
                     << ser << "\n";
            std::cout << "  Diversity order: " << diversity_order << "\n";
            std::cout << "  Spectral efficiency: " << num_tx << " bits/s/Hz\n";
            
        } else {
            std::cerr << "  Detection failed: " << result.message << "\n";
        }
        
        pipeline->teardown();
    }
    
    std::cout << "\n";
}

/// Demonstrate performance vs SNR
void demonstrate_snr_performance() {
    std::cout << "=== SNR Performance Analysis ===\n";
    
    const size_t num_tx = 2;
    const size_t num_rx = 4;
    const size_t num_symbols = 2000;
    std::vector<float> snr_values = {5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 30.0f};
    
    auto config = MIMOPipelineFactory::get_default_config(num_tx, num_rx);
    auto pipeline = MIMOPipelineFactory::create_pipeline(config);
    
    ::framework::pipeline::PipelineSpec spec;
    if (!pipeline->setup(spec)) {
        std::cerr << "Failed to setup SNR analysis pipeline\n";
        return;
    }
    
    std::cout << "Configuration: " << num_tx << "x" << num_rx << " MIMO\n";
    std::cout << "Number of symbols: " << num_symbols << "\n\n";
    
    std::cout << "SNR (dB)  | ZF SER     | MMSE SER   | Throughput (Msym/s)\n";
    std::cout << "----------|------------|------------|-------------------\n";
    
    for (float snr_db : snr_values) {
        // Generate test data for this SNR
        auto channel = generate_channel_matrix(num_rx, num_tx, snr_db);
        auto tx_symbols = generate_transmitted_symbols(num_symbols, num_tx);
        auto rx_signals = apply_channel_and_noise(tx_symbols, channel,
                                                 num_symbols, num_tx, num_rx, snr_db);
        
        // Test Zero Forcing
        std::vector<std::complex<float>> zf_detected;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto zf_result = pipeline->detect_symbols(rx_signals, channel,
                                                 zf_detected, MIMOAlgorithm::ZeroForcing);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto zf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Test MMSE
        std::vector<std::complex<float>> mmse_detected;
        start_time = std::chrono::high_resolution_clock::now();
        auto mmse_result = pipeline->detect_symbols(rx_signals, channel,
                                                   mmse_detected, MIMOAlgorithm::MMSE);
        end_time = std::chrono::high_resolution_clock::now();
        auto mmse_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (zf_result.is_success() && mmse_result.is_success()) {
            double zf_ser = calculate_ser(tx_symbols, zf_detected);
            double mmse_ser = calculate_ser(tx_symbols, mmse_detected);
            double avg_throughput = static_cast<double>(num_symbols * num_tx * 2) / 
                                  (zf_duration.count() + mmse_duration.count());
            
            std::cout << std::setw(8) << std::fixed << std::setprecision(1) << snr_db
                     << "  | " << std::scientific << std::setprecision(1) << zf_ser
                     << "  | " << std::scientific << std::setprecision(1) << mmse_ser
                     << "  | " << std::fixed << std::setprecision(1) << avg_throughput << "\n";
        } else {
            std::cout << std::setw(8) << snr_db << "  | FAIL       | FAIL       | N/A\n";
        }
    }
    
    pipeline->teardown();
    std::cout << "\n";
}

/// Demonstrate real-time streaming
void demonstrate_streaming_mimo() {
    std::cout << "=== Real-time MIMO Streaming Demo ===\n";
    
    const size_t num_tx = 2;
    const size_t num_rx = 2;
    const size_t symbols_per_frame = 128;
    const size_t num_frames = 50;
    const float snr_db = 20.0f;
    
    std::cout << "Streaming configuration:\n";
    std::cout << "MIMO: " << num_tx << "x" << num_rx << "\n";
    std::cout << "Symbols per frame: " << symbols_per_frame << "\n";
    std::cout << "Number of frames: " << num_frames << "\n";
    std::cout << "Target frame rate: 1000 fps\n\n";
    
    auto config = MIMOPipelineFactory::get_low_latency_config(num_tx, num_rx);
    auto pipeline = MIMOPipelineFactory::create_pipeline(config);
    
    ::framework::pipeline::PipelineSpec spec;
    if (!pipeline->setup(spec)) {
        std::cerr << "Failed to setup streaming pipeline\n";
        return;
    }
    
    // Pre-generate channel (assume slowly varying)
    auto channel = generate_channel_matrix(num_rx, num_tx, snr_db);
    
    std::vector<double> frame_processing_times;
    double total_ser = 0.0;
    
    std::cout << "Processing frames in real-time...\n";
    
    for (size_t frame = 0; frame < num_frames; ++frame) {
        // Generate frame data
        auto tx_symbols = generate_transmitted_symbols(symbols_per_frame, num_tx);
        auto rx_signals = apply_channel_and_noise(tx_symbols, channel,
                                                 symbols_per_frame, num_tx, num_rx, snr_db);
        
        // Process frame
        std::vector<std::complex<float>> detected_symbols;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = pipeline->detect_symbols(rx_signals, channel,
                                              detected_symbols, MIMOAlgorithm::MMSE);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        frame_processing_times.push_back(duration.count());
        
        if (result.is_success()) {
            double frame_ser = calculate_ser(tx_symbols, detected_symbols);
            total_ser += frame_ser;
        }
        
        // Simple progress indicator
        if ((frame + 1) % 10 == 0) {
            std::cout << "  Processed " << (frame + 1) << "/" << num_frames << " frames\r" << std::flush;
        }
        
        // Simulate real-time constraint (1ms per frame for 1000 fps)
        if (duration.count() > 1000) {
            std::cout << "\n  Warning: Frame " << frame << " exceeded real-time constraint!\n";
        }
    }
    
    std::cout << "\n\nStreaming Results:\n";
    
    // Calculate statistics
    double avg_processing_time = 0.0;
    double max_processing_time = 0.0;
    for (double time : frame_processing_times) {
        avg_processing_time += time;
        max_processing_time = std::max(max_processing_time, time);
    }
    avg_processing_time /= frame_processing_times.size();
    
    double avg_ser = total_ser / num_frames;
    double total_symbols = num_frames * symbols_per_frame * num_tx;
    double total_time_s = avg_processing_time * num_frames / 1e6;
    double overall_throughput = total_symbols / total_time_s / 1e6;
    
    std::cout << "Average frame time: " << std::fixed << std::setprecision(1) 
             << avg_processing_time << " μs\n";
    std::cout << "Max frame time: " << std::fixed << std::setprecision(1) 
             << max_processing_time << " μs\n";
    std::cout << "Real-time capability: " << (max_processing_time < 1000 ? "YES" : "NO") 
             << " (1000 μs target)\n";
    std::cout << "Average SER: " << std::scientific << std::setprecision(2) << avg_ser << "\n";
    std::cout << "Overall throughput: " << std::fixed << std::setprecision(1) 
             << overall_throughput << " Msymbols/sec\n";
    
    pipeline->teardown();
    std::cout << "\n";
}

int main() {
    try {
        std::cout << "NVIDIA Aerial Framework - MIMO Detection Pipeline Examples\n";
        std::cout << "========================================================\n\n";
        
        demonstrate_basic_mimo_detection();
        demonstrate_batch_mimo_processing();
        demonstrate_mimo_configurations();
        demonstrate_snr_performance();
        demonstrate_streaming_mimo();
        
        std::cout << "All MIMO demos completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}