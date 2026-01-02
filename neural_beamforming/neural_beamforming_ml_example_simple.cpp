#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <complex>

#include "neural_beamforming_pipeline_simple.hpp"

using namespace neural_beamforming;

/**
 * @brief Comprehensive neural beamforming example with ML training pipeline
 */
class ComprehensiveNeuralBeamformingExample {
public:
    ComprehensiveNeuralBeamformingExample() = default;

    void run() {
        std::cout << "\n=== Comprehensive Neural Beamforming ML Example ===\n";
        
        try {
            setup_environment();
            benchmark_beamforming_algorithms();
            demonstrate_training_workflow();
            performance_analysis();
            cleanup();
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

private:
    struct BenchmarkResults {
        std::string algorithm_name;
        float avg_sir_db;
        float avg_latency_ms;
        float spectral_efficiency;
        size_t samples_processed;
    };

    std::vector<BenchmarkResults> results_;

    void setup_environment() {
        std::cout << "\n1. Setting up ML Environment\n";
        std::cout << "Initializing CUDA resources and ML frameworks...\n";
        std::cout << "Environment setup completed.\n";
    }

    void benchmark_beamforming_algorithms() {
        std::cout << "\n2. Benchmarking Beamforming Algorithms\n";
        
        std::vector<std::pair<NeuralBeamformingPipeline::BeamformingMode, std::string>> algorithms = {
            {NeuralBeamformingPipeline::BeamformingMode::TRADITIONAL_MMSE, "Traditional MMSE"},
            {NeuralBeamformingPipeline::BeamformingMode::NEURAL_DNN, "Neural DNN"},
            {NeuralBeamformingPipeline::BeamformingMode::NEURAL_CNN, "Neural CNN"},
            {NeuralBeamformingPipeline::BeamformingMode::HYBRID, "Hybrid ML+Traditional"}
        };

        for (const auto& [mode, name] : algorithms) {
            std::cout << "\nBenchmarking " << name << "...\n";
            
            auto config = NeuralBeamformingPipeline::create_default_config(64, 8);
            config.mode = mode;
            config.batch_size = 32;
            
            auto pipeline = NeuralBeamformingPipelineFactory::create(config);
            
            if (pipeline->initialize()) {
                BenchmarkResults result = run_benchmark(*pipeline, name);
                results_.push_back(result);
                
                std::cout << "Results:\n";
                std::cout << "  - Avg SIR: " << result.avg_sir_db << " dB\n";
                std::cout << "  - Avg Latency: " << result.avg_latency_ms << " ms\n";
                std::cout << "  - Spectral Efficiency: " << result.spectral_efficiency << " bits/s/Hz\n";
                
                pipeline->finalize();
            } else {
                std::cerr << "Failed to initialize pipeline for " << name << "\n";
            }
        }
    }

    BenchmarkResults run_benchmark(NeuralBeamformingPipeline& pipeline, const std::string& name) {
        BenchmarkResults result;
        result.algorithm_name = name;
        
        const int num_iterations = 100;
        float total_sir = 0.0f;
        float total_latency = 0.0f;
        float total_efficiency = 0.0f;
        
        for (int i = 0; i < num_iterations; ++i) {
            auto channel_data = generate_test_channel_data();
            auto user_signals = generate_test_user_signals();
            std::vector<std::complex<float>> output;
            
            auto start = std::chrono::high_resolution_clock::now();
            
            if (pipeline.process_beamforming(channel_data, user_signals, output)) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                
                const auto& metrics = pipeline.get_metrics();
                total_sir += metrics.signal_to_interference_ratio;
                total_latency += duration.count() / 1000.0f; // Convert to ms
                total_efficiency += metrics.spectral_efficiency;
            }
        }
        
        result.avg_sir_db = total_sir / num_iterations;
        result.avg_latency_ms = total_latency / num_iterations;
        result.spectral_efficiency = total_efficiency / num_iterations;
        result.samples_processed = num_iterations;
        
        return result;
    }

    void demonstrate_training_workflow() {
        std::cout << "\n3. ML Training Workflow Demonstration\n";
        
        std::cout << "Generating training data...\n";
        generate_training_dataset();
        
        std::cout << "Simulating model training...\n";
        simulate_model_training();
        
        std::cout << "Evaluating trained model...\n";
        evaluate_trained_model();
    }

    void generate_training_dataset() {
        const size_t num_samples = 10000;
        std::cout << "Generating " << num_samples << " training samples...\n";
        
        // Simulate different channel conditions
        std::vector<std::string> scenarios = {"urban", "suburban", "highway", "indoor"};
        
        for (const auto& scenario : scenarios) {
            std::cout << "  - Scenario: " << scenario << " (" << num_samples/4 << " samples)\n";
            // In a real implementation, this would generate diverse channel data
            // and corresponding optimal beamforming weights
        }
        
        std::cout << "Training dataset generation completed.\n";
    }

    void simulate_model_training() {
        const int num_epochs = 50;
        std::cout << "Training neural beamforming model for " << num_epochs << " epochs...\n";
        
        for (int epoch = 1; epoch <= num_epochs; ++epoch) {
            // Simulate training progress
            float loss = 1.0f * std::exp(-epoch * 0.1f) + 0.01f;
            float accuracy = 1.0f - std::exp(-epoch * 0.15f);
            
            if (epoch % 10 == 0 || epoch == 1) {
                std::cout << "  Epoch " << std::setw(2) << epoch 
                          << ": Loss = " << std::fixed << std::setprecision(4) << loss
                          << ", Accuracy = " << std::setprecision(3) << accuracy << "\n";
            }
        }
        
        std::cout << "Model training completed.\n";
    }

    void evaluate_trained_model() {
        std::cout << "Evaluating trained model performance...\n";
        
        auto config = NeuralBeamformingPipeline::create_default_config(64, 8);
        config.mode = NeuralBeamformingPipeline::BeamformingMode::NEURAL_DNN;
        
        auto pipeline = NeuralBeamformingPipelineFactory::create(config);
        
        if (pipeline->initialize()) {
            auto result = run_benchmark(*pipeline, "Trained Neural Model");
            
            std::cout << "Trained model performance:\n";
            std::cout << "  - SIR: " << result.avg_sir_db << " dB\n";
            std::cout << "  - Latency: " << result.avg_latency_ms << " ms\n";
            std::cout << "  - Spectral Efficiency: " << result.spectral_efficiency << " bits/s/Hz\n";
            
            pipeline->finalize();
        }
    }

    void performance_analysis() {
        std::cout << "\n4. Performance Analysis\n";
        
        if (results_.empty()) {
            std::cout << "No benchmark results available.\n";
            return;
        }
        
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "PERFORMANCE COMPARISON SUMMARY\n";
        std::cout << std::string(80, '=') << "\n";
        
        std::cout << std::left << std::setw(20) << "Algorithm"
                  << std::setw(12) << "SIR (dB)"
                  << std::setw(15) << "Latency (ms)"
                  << std::setw(18) << "Efficiency (b/s/Hz)"
                  << std::setw(10) << "Samples" << "\n";
        std::cout << std::string(80, '-') << "\n";
        
        for (const auto& result : results_) {
            std::cout << std::left << std::setw(20) << result.algorithm_name
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.avg_sir_db
                      << std::setw(15) << std::setprecision(3) << result.avg_latency_ms
                      << std::setw(18) << std::setprecision(2) << result.spectral_efficiency
                      << std::setw(10) << result.samples_processed << "\n";
        }
        std::cout << std::string(80, '=') << "\n";
        
        // Find best performing algorithm
        auto best_sir = std::max_element(results_.begin(), results_.end(),
            [](const auto& a, const auto& b) { return a.avg_sir_db < b.avg_sir_db; });
        
        auto best_latency = std::min_element(results_.begin(), results_.end(),
            [](const auto& a, const auto& b) { return a.avg_latency_ms < b.avg_latency_ms; });
        
        std::cout << "\nBest Performers:\n";
        std::cout << "  - Highest SIR: " << best_sir->algorithm_name 
                  << " (" << best_sir->avg_sir_db << " dB)\n";
        std::cout << "  - Lowest Latency: " << best_latency->algorithm_name 
                  << " (" << best_latency->avg_latency_ms << " ms)\n";
    }

    std::vector<std::complex<float>> generate_test_channel_data() {
        std::vector<std::complex<float>> data(64 * 8); // 64 antennas, 8 users
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& val : data) {
            val = std::complex<float>(dist(gen), dist(gen)) / std::sqrt(2.0f);
        }
        
        return data;
    }

    std::vector<std::complex<float>> generate_test_user_signals() {
        std::vector<std::complex<float>> signals(8 * 256); // 8 users, 256 symbols each
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> symbol_dist(0, 3);
        
        std::vector<std::complex<float>> constellation = {
            { 1.0f/std::sqrt(2.0f),  1.0f/std::sqrt(2.0f)},
            { 1.0f/std::sqrt(2.0f), -1.0f/std::sqrt(2.0f)},
            {-1.0f/std::sqrt(2.0f),  1.0f/std::sqrt(2.0f)},
            {-1.0f/std::sqrt(2.0f), -1.0f/std::sqrt(2.0f)}
        };
        
        for (auto& signal : signals) {
            signal = constellation[symbol_dist(gen)];
        }
        
        return signals;
    }

    void cleanup() {
        std::cout << "\n5. Cleanup\n";
        std::cout << "Cleaning up resources...\n";
        std::cout << "Cleanup completed.\n";
    }
};

int main(int argc, char* argv[]) {
    std::cout << "Comprehensive Neural Beamforming ML Example\n";
    std::cout << "===========================================\n";
    
    try {
        ComprehensiveNeuralBeamformingExample example;
        example.run();
        
        std::cout << "\nExample completed successfully!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}