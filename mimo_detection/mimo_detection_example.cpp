#include "mimo_detection_pipeline.hpp"

#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <memory>
#include <cuda_runtime.h>

using namespace mimo_detection;

/**
 * @brief Generate synthetic channel matrix for MIMO system
 * @param num_rx Number of receive antennas
 * @param num_tx Number of transmit antennas 
 * @param num_subcarriers Number of subcarriers
 * @param snr_db SNR in dB for channel noise
 * @return Channel matrix H of size [num_rx x num_tx x num_subcarriers]
 */
std::vector<std::complex<float>> generate_channel_matrix(
    int num_rx, int num_tx, int num_subcarriers, float snr_db = 20.0f) {
    
    std::vector<std::complex<float>> channel(num_rx * num_tx * num_subcarriers);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Generate Rayleigh fading channel coefficients
    float sigma = std::sqrt(0.5f); // For unit power channel
    
    for (size_t i = 0; i < channel.size(); ++i) {
        float real_part = dist(gen) * sigma;
        float imag_part = dist(gen) * sigma;
        channel[i] = std::complex<float>(real_part, imag_part);
    }
    
    return channel;
}

/**
 * @brief Generate QPSK modulated symbols
 * @param num_symbols Number of symbols to generate
 * @return Vector of QPSK symbols (+/-1 +/- j)
 */
std::vector<std::complex<float>> generate_qpsk_symbols(int num_symbols) {
    std::vector<std::complex<float>> symbols(num_symbols);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> bit_dist(0, 1);
    
    const std::complex<float> qpsk_constellation[4] = {
        {1.0f, 1.0f},   // 00
        {1.0f, -1.0f},  // 01  
        {-1.0f, 1.0f},  // 10
        {-1.0f, -1.0f}  // 11
    };
    
    for (int i = 0; i < num_symbols; ++i) {
        int bits = (bit_dist(gen) << 1) | bit_dist(gen);
        symbols[i] = qpsk_constellation[bits];
    }
    
    return symbols;
}

/**
 * @brief Simulate MIMO transmission: y = H * x + n
 * @param transmitted_symbols TX symbols [num_tx x num_subcarriers x num_symbols]
 * @param channel_matrix Channel matrix [num_rx x num_tx x num_subcarriers]
 * @param noise_variance Noise power
 * @param num_rx Number of RX antennas
 * @param num_tx Number of TX antennas
 * @param num_subcarriers Number of subcarriers
 * @param num_symbols Number of OFDM symbols
 * @return Received symbols [num_rx x num_subcarriers x num_symbols]
 */
std::vector<std::complex<float>> simulate_mimo_transmission(
    const std::vector<std::complex<float>>& transmitted_symbols,
    const std::vector<std::complex<float>>& channel_matrix,
    float noise_variance,
    int num_rx, int num_tx, int num_subcarriers, int num_symbols) {
    
    std::vector<std::complex<float>> received_symbols(num_rx * num_subcarriers * num_symbols, {0.0f, 0.0f});
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise_dist(0.0f, std::sqrt(noise_variance / 2.0f));
    
    // Simulate transmission: y = H * x + n
    for (int symbol_idx = 0; symbol_idx < num_symbols; ++symbol_idx) {
        for (int subcarrier = 0; subcarrier < num_subcarriers; ++subcarrier) {
            for (int rx = 0; rx < num_rx; ++rx) {
                std::complex<float> signal = {0.0f, 0.0f};
                
                // Multiply by channel matrix
                for (int tx = 0; tx < num_tx; ++tx) {
                    int tx_idx = (tx * num_subcarriers + subcarrier) * num_symbols + symbol_idx;
                    int h_idx = (rx * num_tx + tx) * num_subcarriers + subcarrier;
                    
                    signal += channel_matrix[h_idx] * transmitted_symbols[tx_idx];
                }
                
                // Add noise
                float noise_real = noise_dist(gen);
                float noise_imag = noise_dist(gen);
                signal += std::complex<float>(noise_real, noise_imag);
                
                int rx_idx = (rx * num_subcarriers + subcarrier) * num_symbols + symbol_idx;
                received_symbols[rx_idx] = signal;
            }
        }
    }
    
    return received_symbols;
}

/**
 * @brief Calculate Symbol Error Rate (SER)
 * @param original Original transmitted symbols
 * @param detected Detected symbols
 * @return SER as percentage
 */
double calculate_ser(
    const std::vector<std::complex<float>>& original,
    const std::vector<std::complex<float>>& detected) {
    
    if (original.size() != detected.size()) {
        return 100.0; // Invalid comparison
    }
    
    int errors = 0;
    float threshold = 0.5f; // Decision threshold for QPSK
    
    for (size_t i = 0; i < original.size(); ++i) {
        // Hard decision decoding
        float detected_real = detected[i].real() > 0.0f ? 1.0f : -1.0f;
        float detected_imag = detected[i].imag() > 0.0f ? 1.0f : -1.0f;
        
        if (std::abs(detected_real - original[i].real()) > threshold ||
            std::abs(detected_imag - original[i].imag()) > threshold) {
            errors++;
        }
    }
    
    return (static_cast<double>(errors) / original.size()) * 100.0;
}

/**
 * @brief Print complex vector statistics
 */
void print_signal_stats(const std::vector<std::complex<float>>& signal, const std::string& name) {
    if (signal.empty()) return;
    
    double power = 0.0;
    for (const auto& sample : signal) {
        power += std::norm(sample);
    }
    power /= signal.size();
    
    std::cout << name << " - Samples: " << signal.size() 
              << ", Avg Power: " << power 
              << ", Peak: " << std::abs(*std::max_element(signal.begin(), signal.end(),
                 [](const auto& a, const auto& b) { return std::abs(a) < std::abs(b); }))
              << std::endl;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "     NVIDIA Aerial MIMO Detection      \n";
    std::cout << "========================================\n\n";
    
    // MIMO system configuration
    const int num_tx_antennas = 2;
    const int num_rx_antennas = 4;
    const int num_subcarriers = 1024;
    const int num_ofdm_symbols = 14;
    const float snr_db = 15.0f;
    
    std::cout << "MIMO Configuration:\n";
    std::cout << "  TX Antennas: " << num_tx_antennas << "\n";
    std::cout << "  RX Antennas: " << num_rx_antennas << "\n";
    std::cout << "  Subcarriers: " << num_subcarriers << "\n";
    std::cout << "  OFDM Symbols: " << num_ofdm_symbols << "\n";
    std::cout << "  SNR: " << snr_db << " dB\n\n";
    
    try {
        // Configure MIMO parameters
        MIMOParams mimo_params;
        mimo_params.num_tx_antennas = num_tx_antennas;
        mimo_params.num_rx_antennas = num_rx_antennas;
        mimo_params.num_subcarriers = num_subcarriers;
        mimo_params.num_ofdm_symbols = num_ofdm_symbols;
        mimo_params.constellation_size = 4; // QPSK
        mimo_params.algorithm = MIMODetectionAlgorithm::MMSE;
        mimo_params.noise_variance = std::pow(10.0f, -snr_db / 10.0f);
        mimo_params.soft_output = false;
        
        // Create pipeline
        MIMODetectionPipeline::PipelineConfig config;
        config.mimo_params = mimo_params;
        config.enable_profiling = true;
        
        auto pipeline = std::make_unique<MIMODetectionPipeline>(config);
        
        std::cout << "Initializing MIMO detection pipeline...\n";
        if (!pipeline->initialize()) {
            std::cerr << "Failed to initialize MIMO detection pipeline\n";
            return -1;
        }
        std::cout << "Pipeline initialized successfully!\n\n";
        
        // Generate test data
        std::cout << "Generating test data...\n";
        
        // Generate transmitted symbols
        int total_tx_symbols = num_tx_antennas * num_subcarriers * num_ofdm_symbols;
        auto transmitted_symbols = generate_qpsk_symbols(total_tx_symbols);
        
        // Generate channel matrix
        auto channel_matrix = generate_channel_matrix(num_rx_antennas, num_tx_antennas, num_subcarriers, snr_db);
        
        // Simulate MIMO transmission
        auto received_symbols = simulate_mimo_transmission(
            transmitted_symbols, channel_matrix, mimo_params.noise_variance,
            num_rx_antennas, num_tx_antennas, num_subcarriers, num_ofdm_symbols
        );
        
        print_signal_stats(transmitted_symbols, "Transmitted Symbols");
        print_signal_stats(channel_matrix, "Channel Matrix");
        print_signal_stats(received_symbols, "Received Symbols");
        std::cout << "\n";
        
        // Test Zero Forcing detection
        std::cout << "Testing Zero Forcing Detection...\n";
        mimo_params.algorithm = MIMODetectionAlgorithm::ZERO_FORCING;
        pipeline->update_parameters(mimo_params);
        
        std::vector<std::complex<float>> detected_symbols_zf;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool success = pipeline->process(received_symbols, channel_matrix, detected_symbols_zf);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (success) {
            double ser_zf = calculate_ser(transmitted_symbols, detected_symbols_zf);
            std::cout << "  Zero Forcing SER: " << ser_zf << "%\n";
            std::cout << "  Processing Time: " << duration.count() / 1000.0 << " ms\n";
            print_signal_stats(detected_symbols_zf, "  ZF Detected Symbols");
        } else {
            std::cerr << "  Zero Forcing detection failed\n";
        }
        
        // Test MMSE detection
        std::cout << "\nTesting MMSE Detection...\n";
        mimo_params.algorithm = MIMODetectionAlgorithm::MMSE;
        pipeline->update_parameters(mimo_params);
        
        std::vector<std::complex<float>> detected_symbols_mmse;
        start_time = std::chrono::high_resolution_clock::now();
        
        success = pipeline->process(received_symbols, channel_matrix, detected_symbols_mmse);
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (success) {
            double ser_mmse = calculate_ser(transmitted_symbols, detected_symbols_mmse);
            std::cout << "  MMSE SER: " << ser_mmse << "%\n";
            std::cout << "  Processing Time: " << duration.count() / 1000.0 << " ms\n";
            print_signal_stats(detected_symbols_mmse, "  MMSE Detected Symbols");
        } else {
            std::cerr << "  MMSE detection failed\n";
        }
        
        // Performance metrics
        std::cout << "\nPerformance Metrics:\n";
        auto metrics = pipeline->get_performance_metrics();
        std::cout << "  Average Processing Time: " << metrics.avg_processing_time_ms << " ms\n";
        std::cout << "  Peak Processing Time: " << metrics.peak_processing_time_ms << " ms\n";
        std::cout << "  Total Processed Frames: " << metrics.total_processed_frames << "\n";
        std::cout << "  Throughput: " << metrics.throughput_mbps << " MB/s\n";
        
        // Test multiple SNR points
        std::cout << "\nSNR Performance Analysis:\n";
        std::vector<float> snr_points = {0, 5, 10, 15, 20, 25};
        
        for (float test_snr : snr_points) {
            mimo_params.noise_variance = std::pow(10.0f, -test_snr / 10.0f);
            mimo_params.algorithm = MIMODetectionAlgorithm::MMSE;
            pipeline->update_parameters(mimo_params);
            
            // Generate new noisy received symbols
            auto test_received = simulate_mimo_transmission(
                transmitted_symbols, channel_matrix, mimo_params.noise_variance,
                num_rx_antennas, num_tx_antennas, num_subcarriers, num_ofdm_symbols
            );
            
            std::vector<std::complex<float>> test_detected;
            success = pipeline->process(test_received, channel_matrix, test_detected);
            
            if (success) {
                double ser = calculate_ser(transmitted_symbols, test_detected);
                std::cout << "  SNR " << test_snr << " dB: SER = " << ser << "%\n";
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "\n========================================\n";
    std::cout << "MIMO Detection Example completed successfully!\n";
    std::cout << "========================================\n";
    
    return 0;
}