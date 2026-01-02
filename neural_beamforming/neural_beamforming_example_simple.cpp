#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <random>
#include <complex>

#include "neural_beamforming_pipeline_simple.hpp"

using namespace neural_beamforming;

/**
 * @brief Simple neural beamforming example demonstrating basic ML integration
 */
class NeuralBeamformingExample {
public:
    NeuralBeamformingExample() = default;

    void run() {
        std::cout << "\n=== Neural Beamforming Pipeline Example ===\n";
        
        try {
            create_pipeline();
            run_beamforming_demo();
            demonstrate_ml_features();
            cleanup();
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

private:
    std::unique_ptr<NeuralBeamformingPipeline> pipeline_;
    NeuralBeamformingPipeline::Config config_;

    void create_pipeline() {
        std::cout << "\n1. Creating Neural Beamforming Pipeline\n";
        
        // Use the simplified config creation
        config_ = NeuralBeamformingPipeline::create_default_config(64, 8);
        config_.mode = NeuralBeamformingPipeline::BeamformingMode::NEURAL_DNN;
        config_.precision = NeuralBeamformingPipeline::ModelPrecision::FP32;
        
        std::cout << "Configuration:\n";
        std::cout << "  - Antennas: " << config_.num_antennas << "\n";
        std::cout << "  - Users: " << config_.num_users << "\n";
        std::cout << "  - Batch size: " << config_.batch_size << "\n";
        
        // Create pipeline using simplified factory
        pipeline_ = NeuralBeamformingPipelineFactory::create(config_);
        
        if (!pipeline_->initialize()) {
            throw std::runtime_error("Failed to initialize pipeline");
        }
        
        std::cout << "Pipeline created and initialized successfully!\n";
    }

    void run_beamforming_demo() {
        std::cout << "\n2. Running Beamforming Demo\n";
        
        // Generate synthetic data
        auto channel_data = generate_channel_data();
        auto user_signals = generate_user_signals();
        std::vector<std::complex<float>> beamformed_output;
        
        // Run beamforming
        auto start = std::chrono::high_resolution_clock::now();
        
        bool success = pipeline_->process_beamforming(
            channel_data, user_signals, beamformed_output
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (success) {
            std::cout << "Beamforming completed successfully!\n";
            std::cout << "Processing time: " << duration.count() << " μs\n";
            std::cout << "Output size: " << beamformed_output.size() << " symbols\n";
            
            // Display metrics
            const auto& metrics = pipeline_->get_metrics();
            std::cout << "\nPerformance Metrics:\n";
            std::cout << "  - SIR: " << metrics.signal_to_interference_ratio << " dB\n";
            std::cout << "  - Spectral Efficiency: " << metrics.spectral_efficiency << " bits/s/Hz\n";
            std::cout << "  - Total Latency: " << metrics.total_latency_ms << " ms\n";
            std::cout << "  - Processed Samples: " << metrics.processed_samples << "\n";
        } else {
            std::cerr << "Beamforming failed!\n";
        }
    }

    void demonstrate_ml_features() {
        std::cout << "\n3. Demonstrating ML Features\n";
        
        // Demonstrate different beamforming modes
        std::vector<NeuralBeamformingPipeline::BeamformingMode> modes = {
            NeuralBeamformingPipeline::BeamformingMode::TRADITIONAL_MMSE,
            NeuralBeamformingPipeline::BeamformingMode::NEURAL_DNN,
            NeuralBeamformingPipeline::BeamformingMode::NEURAL_CNN,
            NeuralBeamformingPipeline::BeamformingMode::HYBRID
        };
        
        std::vector<std::string> mode_names = {
            "Traditional MMSE", "Neural DNN", "Neural CNN", "Hybrid"
        };
        
        for (size_t i = 0; i < modes.size(); ++i) {
            std::cout << "\nTesting " << mode_names[i] << " beamforming:\n";
            
            // Create new config for this mode
            auto test_config = config_;
            test_config.mode = modes[i];
            
            auto test_pipeline = NeuralBeamformingPipelineFactory::create(test_config);
            
            if (test_pipeline->initialize()) {
                auto channel_data = generate_channel_data();
                auto user_signals = generate_user_signals();
                std::vector<std::complex<float>> output;
                
                if (test_pipeline->process_beamforming(channel_data, user_signals, output)) {
                    const auto& metrics = test_pipeline->get_metrics();
                    std::cout << "  - SIR: " << metrics.signal_to_interference_ratio << " dB\n";
                    std::cout << "  - Latency: " << metrics.total_latency_ms << " ms\n";
                }
                
                test_pipeline->finalize();
            }
        }
    }

    std::vector<std::complex<float>> generate_channel_data() {
        std::vector<std::complex<float>> data(config_.num_antennas * config_.num_users);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& val : data) {
            val = std::complex<float>(dist(gen), dist(gen)) / std::sqrt(2.0f);
        }
        
        std::cout << "Generated channel data: " << data.size() << " elements\n";
        return data;
    }

    std::vector<std::complex<float>> generate_user_signals() {
        std::vector<std::complex<float>> signals(config_.num_users * 1024); // 1024 symbols per user
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> symbol_dist(0, 3);
        
        // QPSK constellation
        std::vector<std::complex<float>> constellation = {
            { 1.0f/std::sqrt(2.0f),  1.0f/std::sqrt(2.0f)},
            { 1.0f/std::sqrt(2.0f), -1.0f/std::sqrt(2.0f)},
            {-1.0f/std::sqrt(2.0f),  1.0f/std::sqrt(2.0f)},
            {-1.0f/std::sqrt(2.0f), -1.0f/std::sqrt(2.0f)}
        };
        
        for (auto& signal : signals) {
            signal = constellation[symbol_dist(gen)];
        }
        
        std::cout << "Generated user signals: " << signals.size() << " symbols\n";
        return signals;
    }

    void cleanup() {
        std::cout << "\n4. Cleanup\n";
        if (pipeline_) {
            pipeline_->finalize();
            pipeline_.reset();
        }
        std::cout << "Cleanup completed.\n";
    }
};

int main() {
    std::cout << "Neural Beamforming Pipeline Example\n";
    std::cout << "===================================\n";
    
    try {
        NeuralBeamformingExample example;
        example.run();
        
        std::cout << "\nExample completed successfully!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}