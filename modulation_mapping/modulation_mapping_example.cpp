#include "modulation_mapping_module.hpp"
#include "modulation_mapping_pipeline.hpp"

#include <complex>
#include <vector>
#include <random>
#include <memory>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "\n"
              << "Modulation Mapping Example - NVIDIA Aerial Framework\n"
              << "\n"
              << "This example demonstrates QAM modulation/demodulation using\n"
              << "the NVIDIA Aerial Framework's pipeline interface.\n"
              << "\n"
              << "Options:\n"
              << "  --scheme <scheme>      Modulation scheme: QPSK, 16QAM, 64QAM (default: QPSK)\n"
              << "  --subcarriers <num>    Number of subcarriers (default: 1200)\n"
              << "  --symbols <num>        Number of OFDM symbols (default: 14)\n"
              << "  --test-ber             Run BER performance test\n"
              << "  --test-evm             Test Error Vector Magnitude\n"
              << "  --constellation        Display constellation diagram\n"
              << "  --help                 Show this help message\n";
}

modulation_mapping::ModulationScheme parse_modulation_scheme(const std::string& scheme_str) {
    if (scheme_str == "QPSK") return modulation_mapping::ModulationScheme::QPSK;
    if (scheme_str == "16QAM") return modulation_mapping::ModulationScheme::QAM16;
    if (scheme_str == "64QAM") return modulation_mapping::ModulationScheme::QAM64;
    
    std::cerr << "Unknown modulation scheme: " << scheme_str << std::endl;
    return modulation_mapping::ModulationScheme::QPSK;
}

void generate_random_bits(std::vector<uint8_t>& bits) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    
    for (auto& bit : bits) {
        bit = static_cast<uint8_t>(dis(gen));
    }
}

void add_awgn(std::vector<std::complex<float>>& symbols, double snr_db) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Calculate noise variance
    double signal_power = 1.0;
    double noise_power = signal_power / std::pow(10.0, snr_db / 10.0);
    double noise_std = std::sqrt(noise_power / 2.0);
    
    std::normal_distribution<float> noise_dist(0.0f, static_cast<float>(noise_std));
    
    for (auto& symbol : symbols) {
        float noise_i = noise_dist(gen);
        float noise_q = noise_dist(gen);
        symbol += std::complex<float>(noise_i, noise_q);
    }
}

void display_constellation_diagram(const modulation_mapping::ModulationPipeline& pipeline) {
    std::cout << "\n=== Constellation Diagram ===\n";
    
    std::vector<std::complex<float>> constellation_points;
    std::vector<std::string> labels;
    pipeline.get_constellation_diagram(constellation_points, labels);
    
    std::cout << "Constellation points and bit mappings:\n";
    for (size_t i = 0; i < constellation_points.size(); ++i) {
        const auto& point = constellation_points[i];
        std::cout << std::fixed << std::setprecision(3)
                  << "  " << labels[i] << " -> ("
                  << std::setw(7) << point.real() << ", "
                  << std::setw(7) << point.imag() << ")\n";
    }
}

int count_bit_errors(const std::vector<uint8_t>& original, const std::vector<uint8_t>& received) {
    int errors = 0;
    size_t min_size = std::min(original.size(), received.size());
    
    for (size_t i = 0; i < min_size; ++i) {
        if (original[i] != received[i]) {
            errors++;
        }
    }
    return errors;
}

void run_ber_test(
    modulation_mapping::ModulationPipeline& pipeline,
    [[maybe_unused]] const modulation_mapping::ModulationParams& params
) {
    std::cout << "\n=== BER Performance Test ===\n";
    
    // SNR range for testing (dB)
    std::vector<double> snr_db_values = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    
    // Calculate number of bits per test
    size_t total_bits = pipeline.calculate_total_bits();
    std::cout << "Testing with " << total_bits << " bits per SNR point\n\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "SNR (dB)  | Measured BER | Theoretical BER | Modulation Time | Demodulation Time\n";
    std::cout << "----------|--------------|-----------------|-----------------|------------------\n";
    
    for (double snr_db : snr_db_values) {
        // Generate random bits
        std::vector<uint8_t> tx_bits(total_bits);
        generate_random_bits(tx_bits);
        
        // Modulation
        std::vector<std::complex<float>> modulated_symbols;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool success = pipeline.modulate(tx_bits, modulated_symbols);
        if (!success) {
            std::cerr << "Modulation failed for SNR " << snr_db << " dB\n";
            continue;
        }
        
        auto mod_end_time = std::chrono::high_resolution_clock::now();
        auto mod_duration = std::chrono::duration_cast<std::chrono::microseconds>(mod_end_time - start_time);
        
        // Add AWGN
        add_awgn(modulated_symbols, snr_db);
        
        // Demodulation
        std::vector<uint8_t> rx_bits;
        auto demod_start_time = std::chrono::high_resolution_clock::now();
        
        success = pipeline.demodulate(modulated_symbols, rx_bits);
        if (!success) {
            std::cerr << "Demodulation failed for SNR " << snr_db << " dB\n";
            continue;
        }
        
        auto demod_end_time = std::chrono::high_resolution_clock::now();
        auto demod_duration = std::chrono::duration_cast<std::chrono::microseconds>(demod_end_time - demod_start_time);
        
        // Calculate BER
        int bit_errors = count_bit_errors(tx_bits, rx_bits);
        double measured_ber = static_cast<double>(bit_errors) / total_bits;
        double theoretical_ber = pipeline.calculate_theoretical_ser(snr_db);
        
        std::cout << std::setw(8) << snr_db << "  | "
                  << std::setw(12) << measured_ber << " | "
                  << std::setw(15) << theoretical_ber << " | "
                  << std::setw(13) << (mod_duration.count() / 1000.0) << " ms | "
                  << std::setw(14) << (demod_duration.count() / 1000.0) << " ms\n";
    }
    
    // Display performance metrics
    auto metrics = pipeline.get_performance_metrics();
    std::cout << "\nPerformance Summary:\n";
    std::cout << "  Processed frames: " << metrics.total_processed_frames << "\n";
    std::cout << "  Avg processing time: " << std::fixed << std::setprecision(3) 
              << metrics.avg_processing_time_ms << " ms\n";
    std::cout << "  Peak processing time: " << metrics.peak_processing_time_ms << " ms\n";
    std::cout << "  Estimated throughput: " << std::fixed << std::setprecision(2) 
              << metrics.throughput_mbps << " Mbps\n";
}

void run_evm_test(
    modulation_mapping::ModulationPipeline& pipeline,
    [[maybe_unused]] const modulation_mapping::ModulationParams& params
) {
    std::cout << "\n=== Error Vector Magnitude Test ===\n";
    
    size_t total_bits = pipeline.calculate_total_bits();
    
    // Generate random bits
    std::vector<uint8_t> tx_bits(total_bits);
    generate_random_bits(tx_bits);
    
    // Modulation
    std::vector<std::complex<float>> clean_symbols;
    bool success = pipeline.modulate(tx_bits, clean_symbols);
    if (!success) {
        std::cerr << "Modulation failed for EVM test\n";
        return;
    }
    
    std::cout << "Generated " << clean_symbols.size() << " symbols\n";
    std::cout << "Testing EVM with different noise levels:\n\n";
    std::cout << "SNR (dB) | RMS EVM (%) | Peak EVM (%)\n";
    std::cout << "---------|-------------|-------------\n";
    
    std::vector<double> snr_values = {5, 10, 15, 20, 25, 30};
    
    for (double snr_db : snr_values) {
        auto noisy_symbols = clean_symbols;
        add_awgn(noisy_symbols, snr_db);
        
        // Calculate EVM
        double sum_error_power = 0.0;
        double sum_signal_power = 0.0;
        double peak_evm = 0.0;
        
        for (size_t i = 0; i < clean_symbols.size(); ++i) {
            std::complex<float> error = noisy_symbols[i] - clean_symbols[i];
            double error_power = std::norm(error);
            double signal_power = std::norm(clean_symbols[i]);
            
            sum_error_power += error_power;
            sum_signal_power += signal_power;
            
            if (signal_power > 0) {
                double symbol_evm = std::sqrt(error_power / signal_power) * 100.0;
                peak_evm = std::max(peak_evm, symbol_evm);
            }
        }
        
        double rms_evm = (sum_signal_power > 0) ? 
                         std::sqrt(sum_error_power / sum_signal_power) * 100.0 : 0.0;
        
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(8) << snr_db << " | "
                  << std::setw(11) << rms_evm << " | "
                  << std::setw(11) << peak_evm << "\n";
    }
}

void run_basic_test(
    modulation_mapping::ModulationPipeline& pipeline,
    const modulation_mapping::ModulationParams& params
) {
    std::cout << "\n=== Basic Modulation/Demodulation Test ===\n";
    
    // Calculate data sizes
    size_t total_bits = pipeline.calculate_total_bits();
    size_t total_symbols = pipeline.calculate_total_symbols();
    
    std::cout << "Configuration:\n";
    std::cout << "  Modulation scheme: ";
    switch (params.scheme) {
        case modulation_mapping::ModulationScheme::BPSK: std::cout << "BPSK"; break;
        case modulation_mapping::ModulationScheme::QPSK: std::cout << "QPSK"; break;
        case modulation_mapping::ModulationScheme::QAM16: std::cout << "16-QAM"; break;
        case modulation_mapping::ModulationScheme::QAM64: std::cout << "64-QAM"; break;
        case modulation_mapping::ModulationScheme::QAM256: std::cout << "256-QAM"; break;
    }
    std::cout << "\n";
    std::cout << "  Subcarriers: " << params.num_subcarriers << "\n";
    std::cout << "  OFDM symbols: " << params.num_ofdm_symbols << "\n";
    std::cout << "  Total bits: " << total_bits << "\n";
    std::cout << "  Total symbols: " << total_symbols << "\n";
    
    // Generate test data
    std::vector<uint8_t> tx_bits(total_bits);
    for (size_t i = 0; i < tx_bits.size(); ++i) {
        tx_bits[i] = static_cast<uint8_t>(i % 2);  // Alternating pattern
    }
    
    std::cout << "\nFirst 16 input bits: ";
    for (size_t i = 0; i < std::min<size_t>(16, tx_bits.size()); ++i) {
        std::cout << static_cast<int>(tx_bits[i]);
    }
    std::cout << "\n";
    
    // Modulation
    std::vector<std::complex<float>> modulated_symbols;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = pipeline.modulate(tx_bits, modulated_symbols);
    if (!success) {
        std::cerr << "Modulation failed!\n";
        return;
    }
    
    auto mod_end_time = std::chrono::high_resolution_clock::now();
    auto mod_duration = std::chrono::duration_cast<std::chrono::microseconds>(mod_end_time - start_time);
    
    std::cout << "Modulated " << modulated_symbols.size() << " symbols in " 
              << (mod_duration.count() / 1000.0) << " ms\n";
    
    // Display first few symbols
    std::cout << "First 4 symbols: ";
    for (size_t i = 0; i < std::min<size_t>(4, modulated_symbols.size()); ++i) {
        std::cout << "(" << std::fixed << std::setprecision(3)
                  << modulated_symbols[i].real() << ", " 
                  << modulated_symbols[i].imag() << ") ";
    }
    std::cout << "\n";
    
    // Demodulation
    std::vector<uint8_t> rx_bits;
    auto demod_start_time = std::chrono::high_resolution_clock::now();
    
    success = pipeline.demodulate(modulated_symbols, rx_bits);
    if (!success) {
        std::cerr << "Demodulation failed!\n";
        return;
    }
    
    auto demod_end_time = std::chrono::high_resolution_clock::now();
    auto demod_duration = std::chrono::duration_cast<std::chrono::microseconds>(demod_end_time - demod_start_time);
    
    std::cout << "Demodulated " << rx_bits.size() << " bits in " 
              << (demod_duration.count() / 1000.0) << " ms\n";
    
    std::cout << "First 16 output bits: ";
    for (size_t i = 0; i < std::min<size_t>(16, rx_bits.size()); ++i) {
        std::cout << static_cast<int>(rx_bits[i]);
    }
    std::cout << "\n";
    
    // Verify correctness
    int bit_errors = count_bit_errors(tx_bits, rx_bits);
    double ber = static_cast<double>(bit_errors) / total_bits;
    
    std::cout << "\nResults:\n";
    std::cout << "  Bit errors: " << bit_errors << " / " << total_bits << "\n";
    std::cout << "  BER: " << std::scientific << std::setprecision(3) << ber << "\n";
    
    if (bit_errors == 0) {
        std::cout << "  ✓ Perfect reconstruction!\n";
    } else {
        std::cout << "  ✗ Errors detected\n";
    }
}

int main(int argc, char* argv[]) {
    // Default parameters
    modulation_mapping::ModulationScheme scheme = modulation_mapping::ModulationScheme::QPSK;
    int num_subcarriers = 1200;
    int num_ofdm_symbols = 14;
    bool test_ber = false;
    bool test_evm = false;
    bool show_constellation = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--scheme" && i + 1 < argc) {
            scheme = parse_modulation_scheme(argv[++i]);
        } else if (arg == "--subcarriers" && i + 1 < argc) {
            num_subcarriers = std::atoi(argv[++i]);
        } else if (arg == "--symbols" && i + 1 < argc) {
            num_ofdm_symbols = std::atoi(argv[++i]);
        } else if (arg == "--test-ber") {
            test_ber = true;
        } else if (arg == "--test-evm") {
            test_evm = true;
        } else if (arg == "--constellation") {
            show_constellation = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "NVIDIA Aerial Framework - Modulation Mapping Example\n";
    std::cout << "=====================================================\n";
    
    try {
        // Initialize CUDA
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        
        // Create modulation configuration
        modulation_mapping::ModulationParams params;
        params.scheme = scheme;
        params.num_subcarriers = num_subcarriers;
        params.num_ofdm_symbols = num_ofdm_symbols;
        params.soft_output = true;
        params.noise_variance = 0.1f;
        
        modulation_mapping::ModulationPipeline::PipelineConfig config;
        config.module_id = "1001";
        config.modulation_params = params;
        config.enable_profiling = true;
        
        // Create and initialize pipeline
        modulation_mapping::ModulationPipeline pipeline(config);
        if (!pipeline.initialize()) {
            std::cerr << "Failed to initialize modulation pipeline\n";
            return 1;
        }
        
        std::cout << "Initialized modulation pipeline successfully\n";
        
        // Show constellation diagram if requested
        if (show_constellation) {
            display_constellation_diagram(pipeline);
        }
        
        // Run basic test
        run_basic_test(pipeline, params);
        
        // Run performance tests if requested
        if (test_ber) {
            run_ber_test(pipeline, params);
        }
        
        if (test_evm) {
            run_evm_test(pipeline, params);
        }
        
        std::cout << "\nExample completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}