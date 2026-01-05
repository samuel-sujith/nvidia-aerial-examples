#include "neural_beamforming_module.hpp"
#include "neural_beamforming_pipeline.hpp"

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
              << "Neural Beamforming Example - NVIDIA Aerial Framework\n"
              << "\n"
              << "This example demonstrates massive MIMO beamforming using both\n"
              << "classical algorithms and neural network approaches.\n"
              << "\n"
              << "Options:\n"
              << "  --algorithm <alg>      Beamforming algorithm: CONVENTIONAL, MVDR, ZF, NEURAL (default: MVDR)\n"
              << "  --antennas <num>       Number of antenna elements (default: 64)\n"
              << "  --users <num>          Number of users (default: 4)\n"
              << "  --subcarriers <num>    Number of subcarriers (default: 1200)\n"
              << "  --symbols <num>        Number of OFDM symbols (default: 14)\n"
              << "  --test-gain            Run beamforming gain test\n"
              << "  --test-sinr            Test SINR performance\n"
              << "  --steering-vector      Display steering vector pattern\n"
              << "  --model <path>    Path to the neural network model\n"
              << "  --help                 Show this help message\n";
}

neural_beamforming::BeamformingAlgorithm parse_algorithm(const std::string& alg_str) {
    if (alg_str == "CONVENTIONAL") return neural_beamforming::BeamformingAlgorithm::CONVENTIONAL;
    if (alg_str == "MVDR") return neural_beamforming::BeamformingAlgorithm::MVDR;
    if (alg_str == "ZF" || alg_str == "ZERO_FORCING") return neural_beamforming::BeamformingAlgorithm::ZERO_FORCING;
    if (alg_str == "NEURAL") return neural_beamforming::BeamformingAlgorithm::NEURAL_NETWORK;
    
    std::cerr << "Unknown beamforming algorithm: " << alg_str << std::endl;
    return neural_beamforming::BeamformingAlgorithm::MVDR;
}

void generate_random_symbols(std::vector<std::complex<float>>& symbols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& symbol : symbols) {
        float real = dist(gen);
        float imag = dist(gen);
        symbol = std::complex<float>(real, imag);
    }
}

void generate_channel_matrix(
    std::vector<std::complex<float>>& channel_matrix,
    int num_users,
    int num_antennas,
    int num_subcarriers
) {
    channel_matrix.resize(num_users * num_antennas * num_subcarriers);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Generate realistic channel coefficients
    for (int user = 0; user < num_users; ++user) {
        for (int ant = 0; ant < num_antennas; ++ant) {
            for (int sc = 0; sc < num_subcarriers; ++sc) {
                int idx = user * num_antennas * num_subcarriers + ant * num_subcarriers + sc;
                
                // Add some correlation based on antenna spacing and frequency
                float phase_offset = 2.0f * M_PI * ant * std::sin(user * M_PI / 8.0f) / num_antennas;
                float freq_response = 1.0f - 0.1f * std::abs(sc - num_subcarriers/2) / (num_subcarriers/2);
                
                float magnitude = freq_response * (1.0f + 0.3f * dist(gen));
                float phase = phase_offset + 0.2f * dist(gen);
                
                channel_matrix[idx] = std::complex<float>(
                    magnitude * std::cos(phase),
                    magnitude * std::sin(phase)
                );
            }
        }
    }
}

void add_noise_to_symbols(std::vector<std::complex<float>>& symbols, float snr_db) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Calculate noise variance from SNR
    float signal_power = 1.0f;
    float noise_power = signal_power / std::pow(10.0f, snr_db / 10.0f);
    float noise_std = std::sqrt(noise_power / 2.0f);
    
    std::normal_distribution<float> noise_dist(0.0f, noise_std);
    
    for (auto& symbol : symbols) {
        float noise_real = noise_dist(gen);
        float noise_imag = noise_dist(gen);
        symbol += std::complex<float>(noise_real, noise_imag);
    }
}

void display_steering_vector_pattern(const neural_beamforming::NeuralBeamformingPipeline& pipeline) {
    std::cout << "\n=== Steering Vector Pattern ===\n";
    
    std::cout << "Angle (deg) | Steering Vector (first 4 elements)\n";
    std::cout << "------------|----------------------------------\n";
    
    for (int angle = 0; angle <= 180; angle += 30) {
        std::vector<std::complex<float>> steering_vector;
        pipeline.get_steering_vector(static_cast<float>(angle), steering_vector);
        
        std::cout << std::setw(10) << angle << "  | ";
        for (int i = 0; i < std::min(4, static_cast<int>(steering_vector.size())); ++i) {
            std::cout << "(" << std::fixed << std::setprecision(2) 
                      << steering_vector[i].real() << "," 
                      << steering_vector[i].imag() << ") ";
        }
        std::cout << "\n";
    }
}

void run_beamforming_gain_test(
    neural_beamforming::NeuralBeamformingPipeline& pipeline,
    const neural_beamforming::BeamformingParams& params
) {
    std::cout << "\n=== Beamforming Gain Test ===\n";
    
    std::vector<neural_beamforming::BeamformingAlgorithm> algorithms = {
        neural_beamforming::BeamformingAlgorithm::CONVENTIONAL,
        neural_beamforming::BeamformingAlgorithm::MVDR,
        neural_beamforming::BeamformingAlgorithm::ZERO_FORCING,
        neural_beamforming::BeamformingAlgorithm::NEURAL_NETWORK
    };
    
    std::vector<std::string> alg_names = {
        "Conventional", "MVDR", "Zero-Forcing", "Neural Network"
    };
    
    std::vector<int> antenna_counts = {16, 32, 64, 128};
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Antennas | ";
    for (const auto& name : alg_names) {
        std::cout << std::setw(12) << name << " | ";
    }
    std::cout << "\n";
    std::cout << "---------|";
    for (size_t i = 0; i < alg_names.size(); ++i) {
        std::cout << "-------------|";
    }
    std::cout << "\n";
    
    for (int num_antennas : antenna_counts) {
        std::cout << std::setw(8) << num_antennas << " | ";
        
        for (auto algorithm : algorithms) {
            double gain = pipeline.calculate_theoretical_gain(num_antennas, algorithm);
            std::cout << std::setw(10) << gain << " dB | ";
        }
        std::cout << "\n";
    }
}

void run_sinr_performance_test(
    neural_beamforming::NeuralBeamformingPipeline& pipeline,
    [[maybe_unused]] const neural_beamforming::BeamformingParams& params
) {
    std::cout << "\n=== SINR Performance Test ===\n";
    
    // Test parameters
    size_t input_symbols_count = pipeline.calculate_input_symbols();
    size_t channel_estimates_count = params.num_users * params.num_antennas * params.num_subcarriers;
    
    std::cout << "Testing with " << input_symbols_count << " input symbols and " 
              << channel_estimates_count << " channel estimates\n\n";
    
    std::vector<float> snr_db_values = {0, 5, 10, 15, 20, 25, 30};
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "SNR (dB) | Processing Time | Avg SINR | Beamforming Gain | Throughput\n";
    std::cout << "---------|-----------------|----------|------------------|------------\n";
    
    for (float snr_db : snr_db_values) {
        // Generate test data
        std::vector<std::complex<float>> input_symbols(input_symbols_count);
        std::vector<std::complex<float>> channel_estimates(channel_estimates_count);
        
        generate_random_symbols(input_symbols);
        generate_channel_matrix(channel_estimates, params.num_users, params.num_antennas, params.num_subcarriers);
        
        // Add noise
        add_noise_to_symbols(input_symbols, snr_db);
        
        // Output buffers
        std::vector<std::complex<float>> output_symbols;
        std::vector<std::complex<float>> beamforming_weights;
        std::vector<float> performance_metrics;
        
        // Measure processing time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool success = pipeline.process_beamforming(
            input_symbols, channel_estimates, output_symbols, 
            beamforming_weights, performance_metrics
        );
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (!success) {
            std::cout << std::setw(8) << snr_db << " | Processing failed\n";
            continue;
        }
        
        // Calculate average SINR from performance metrics
        double avg_sinr = 0.0;
        for (float sinr : performance_metrics) {
            avg_sinr += sinr;
        }
        avg_sinr /= performance_metrics.size();
        
        // Get pipeline metrics
        auto pipeline_metrics = pipeline.get_performance_metrics();
        
        std::cout << std::setw(8) << snr_db << " | "
                  << std::setw(13) << (duration.count() / 1000.0) << " ms | "
                  << std::setw(8) << avg_sinr << " | "
                  << std::setw(14) << pipeline_metrics.beamforming_gain_db << " dB | "
                  << std::setw(8) << pipeline_metrics.throughput_mbps << " Mbps\n";
    }
}

void run_basic_beamforming_test(
    neural_beamforming::NeuralBeamformingPipeline& pipeline,
    const neural_beamforming::BeamformingParams& params
) {
    std::cout << "\n=== Basic Beamforming Test ===\n";
    
    // Configuration display
    std::cout << "Configuration:\n";
    std::cout << "  Algorithm: ";
    switch (params.algorithm) {
        case neural_beamforming::BeamformingAlgorithm::CONVENTIONAL: std::cout << "Conventional"; break;
        case neural_beamforming::BeamformingAlgorithm::MVDR: std::cout << "MVDR"; break;
        case neural_beamforming::BeamformingAlgorithm::ZERO_FORCING: std::cout << "Zero-Forcing"; break;
        case neural_beamforming::BeamformingAlgorithm::NEURAL_NETWORK: std::cout << "Neural Network"; break;
    }
    std::cout << "\n";
    std::cout << "  Antennas: " << params.num_antennas << "\n";
    std::cout << "  Users: " << params.num_users << "\n";
    std::cout << "  Subcarriers: " << params.num_subcarriers << "\n";
    std::cout << "  OFDM symbols: " << params.num_ofdm_symbols << "\n";
    
    // Calculate data sizes
    size_t input_symbols_count = pipeline.calculate_input_symbols();
    size_t output_symbols_count = pipeline.calculate_output_symbols();
    size_t channel_estimates_count = params.num_users * params.num_antennas * params.num_subcarriers;
    
    std::cout << "  Input symbols: " << input_symbols_count << "\n";
    std::cout << "  Output symbols: " << output_symbols_count << "\n";
    std::cout << "  Channel estimates: " << channel_estimates_count << "\n";
    
    // Generate test data
    std::vector<std::complex<float>> input_symbols(input_symbols_count);
    std::vector<std::complex<float>> channel_estimates(channel_estimates_count);
    
    generate_random_symbols(input_symbols);
    generate_channel_matrix(channel_estimates, params.num_users, params.num_antennas, params.num_subcarriers);
    
    // Add some noise
    add_noise_to_symbols(input_symbols, 20.0f); // 20 dB SNR
    
    std::cout << "\nFirst 4 input symbols (antenna 0): ";
    for (int i = 0; i < std::min(4, static_cast<int>(input_symbols.size())); ++i) {
        std::cout << "(" << std::fixed << std::setprecision(3)
                  << input_symbols[i].real() << "," 
                  << input_symbols[i].imag() << ") ";
    }
    std::cout << "\n";
    
    // Output buffers
    std::vector<std::complex<float>> output_symbols;
    std::vector<std::complex<float>> beamforming_weights;
    std::vector<float> performance_metrics;
    
    // Process beamforming
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = pipeline.process_beamforming(
        input_symbols, channel_estimates, output_symbols, 
        beamforming_weights, performance_metrics
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (!success) {
        std::cerr << "Beamforming processing failed!\n";
        return;
    }
    
    std::cout << "Processed beamforming in " << (duration.count() / 1000.0) << " ms\n";
    
    // Display results
    std::cout << "First 4 output symbols (user 0): ";
    for (int i = 0; i < std::min(4, static_cast<int>(output_symbols.size())); ++i) {
        std::cout << "(" << std::fixed << std::setprecision(3)
                  << output_symbols[i].real() << "," 
                  << output_symbols[i].imag() << ") ";
    }
    std::cout << "\n";
    
    // Calculate and display average SINR
    double avg_sinr = 0.0;
    for (float sinr : performance_metrics) {
        avg_sinr += sinr;
    }
    avg_sinr /= performance_metrics.size();
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "  Average SINR: " << std::fixed << std::setprecision(2) << avg_sinr << " dB\n";
    
    // Get pipeline performance metrics
    auto pipeline_metrics = pipeline.get_performance_metrics();
    std::cout << "  Beamforming gain: " << pipeline_metrics.beamforming_gain_db << " dB\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << pipeline_metrics.throughput_mbps << " Mbps\n";
    std::cout << "  Processing time: " << std::fixed << std::setprecision(3) << pipeline_metrics.avg_processing_time_ms << " ms\n";
}

int main(int argc, char* argv[]) {
    // Default parameters
    neural_beamforming::BeamformingAlgorithm algorithm = neural_beamforming::BeamformingAlgorithm::MVDR;
    int num_antennas = 64;
    int num_users = 4;
    int num_subcarriers = 1200;
    int num_ofdm_symbols = 14;
    bool test_gain = false;
    bool test_sinr = false;
    bool show_steering = false;
    std::string model_path;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--algorithm" && i + 1 < argc) {
            algorithm = parse_algorithm(argv[++i]);
        } else if (arg == "--antennas" && i + 1 < argc) {
            num_antennas = std::atoi(argv[++i]);
        } else if (arg == "--users" && i + 1 < argc) {
            num_users = std::atoi(argv[++i]);
        } else if (arg == "--subcarriers" && i + 1 < argc) {
            num_subcarriers = std::atoi(argv[++i]);
        } else if (arg == "--symbols" && i + 1 < argc) {
            num_ofdm_symbols = std::atoi(argv[++i]);
        } else if (arg == "--test-gain") {
            test_gain = true;
        } else if (arg == "--test-sinr") {
            test_sinr = true;
        } else if (arg == "--steering-vector") {
            show_steering = true;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "NVIDIA Aerial Framework - Neural Beamforming Example\n";
    std::cout << "=====================================================\n";
    
    try {
        // Initialize CUDA
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        
        // Create beamforming configuration
        neural_beamforming::BeamformingParams params;
        params.algorithm = algorithm;
        params.num_antennas = num_antennas;
        params.num_users = num_users;
        params.num_subcarriers = num_subcarriers;
        params.num_ofdm_symbols = num_ofdm_symbols;
        params.regularization_factor = 1e-6f;
        params.noise_power = 0.1f;
        params.enable_calibration = true;
        if (!model_path.empty()) {
            params.model_path = model_path;
        }
        
        neural_beamforming::NeuralBeamformingPipeline::PipelineConfig config;
        config.module_id = "1001";
        config.beamforming_params = params;
        config.enable_profiling = true;
        
        // Create and initialize pipeline
        neural_beamforming::NeuralBeamformingPipeline pipeline(config);
        if (!pipeline.initialize()) {
            std::cerr << "Failed to initialize neural beamforming pipeline\n";
            return 1;
        }
        
        std::cout << "Initialized neural beamforming pipeline successfully\n";
        
        // Show steering vector pattern if requested
        if (show_steering) {
            display_steering_vector_pattern(pipeline);
        }
        
        // Run basic test
        run_basic_beamforming_test(pipeline, params);
        
        // Run performance tests if requested
        if (test_gain) {
            run_beamforming_gain_test(pipeline, params);
        }
        
        if (test_sinr) {
            run_sinr_performance_test(pipeline, params);
        }
        
        std::cout << "\nExample completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}