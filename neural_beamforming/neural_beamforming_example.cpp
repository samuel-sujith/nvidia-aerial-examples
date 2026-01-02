#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <random>

#include "neural_beamforming_pipeline_simple.hpp"

using namespace neural_beamforming;

/**
 * @brief Simple neural beamforming example demonstrating basic ML integration
 */
class NeuralBeamformingExample {
public:
    NeuralBeamformingExample() = default;

    /**
     * @brief Run the basic neural beamforming demonstration
     */
    void run() {
        std::cout << "=== Neural Beamforming Pipeline Example ===\n";
        std::cout << "Demonstrates ML model integration with Aerial framework\n\n";

        try {
            initialize_framework();
            create_pipeline();
            run_beamforming_demo();
            demonstrate_ml_features();
            cleanup();
            
            std::cout << "\n=== Example completed successfully! ===\n";
            
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

private:
    std::shared_ptr<memory::MemoryPool> memory_pool_;
    std::shared_ptr<cuda_utils::CudaContext> cuda_context_;
    std::unique_ptr<NeuralBeamformingPipeline> pipeline_;
    NeuralBeamformingPipeline::Config config_;
    
    // Demo parameters
    static constexpr std::size_t NUM_ANTENNAS = 32;
    static constexpr std::size_t NUM_USERS = 4;
    static constexpr std::size_t BATCH_SIZE = 8;

    void initialize_framework() {
        std::cout << "1. Initializing Aerial Framework components...\n";
        
        // Create memory pool for efficient GPU memory management
        memory_pool_ = memory::MemoryPoolFactory::create_cuda_pool(
            1024 * 1024 * 1024  // 1GB pool
        );
        
        // Initialize CUDA context
        cuda_context_ = std::make_shared<cuda_utils::CudaContext>(0); // GPU 0
        
        std::cout << "   ✓ Memory pool created (1GB)\n";
        std::cout << "   ✓ CUDA context initialized\n";
    }

    void create_pipeline() {
        std::cout << "\n2. Creating Neural Beamforming Pipeline...\n";
        
        // Configure pipeline for demo
        config_ = NeuralBeamformingPipelineFactory::create_default_config(
            NUM_ANTENNAS, NUM_USERS
        );
        config_.batch_size = BATCH_SIZE;
        config_.mode = NeuralBeamformingPipeline::BeamformingMode::TRADITIONAL_MMSE; // Start with baseline
        config_.enable_training_mode = false; // Disable training for this demo
        
        std::cout << "   Configuration:\n";
        std::cout << "   - Antennas: " << config_.num_antennas << "\n";
        std::cout << "   - Users: " << config_.num_users << "\n";
        std::cout << "   - Batch size: " << config_.batch_size << "\n";
        std::cout << "   - Mode: Traditional MMSE (baseline)\n";
        
        // Create pipeline using factory
        pipeline_ = NeuralBeamformingPipelineFactory::create(
            config_, memory_pool_, cuda_context_
        );
        
        std::cout << "   ✓ Neural beamforming pipeline created\n";
    }

    void run_beamforming_demo() {
        std::cout << "\n3. Running Beamforming Demonstration...\n";
        
        // Generate synthetic channel data
        auto channel_data = generate_channel_data();
        auto user_requirements = generate_user_requirements();
        
        // Initialize pipeline
        std::vector<tensor::TensorInfo> inputs = {channel_data, user_requirements};
        auto init_result = pipeline_->initialize(inputs);
        
        if (init_result.status != task::TaskStatus::Completed) {
            throw std::runtime_error("Pipeline initialization failed");
        }
        
        std::cout << "   ✓ Pipeline initialized successfully\n";
        
        // Create output tensor for beamforming weights
        auto beamforming_weights = create_output_tensor();
        std::vector<tensor::TensorInfo> outputs = {beamforming_weights};
        
        // Measure processing time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Execute beamforming processing
        auto result = pipeline_->execute(inputs, outputs);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        
        if (result.status == task::TaskStatus::Completed) {
            std::cout << "   ✓ Beamforming processing completed\n";
            std::cout << "   ✓ Processing time: " << duration / 1000.0 << " ms\n";
            
            // Display results
            display_beamforming_results();
        } else {
            throw std::runtime_error("Beamforming execution failed: " + result.error_message);
        }
    }

    void demonstrate_ml_features() {
        std::cout << "\n4. Demonstrating ML Integration Features...\n";
        
        // Show how to switch to neural mode (would require pre-trained model)
        std::cout << "   ML Pipeline Features:\n";
        std::cout << "   • TensorRT engine loading for optimized inference\n";
        std::cout << "   • Multiple model precision support (FP32/FP16/INT8)\n";
        std::cout << "   • Online training data collection\n";
        std::cout << "   • Hybrid traditional + neural algorithms\n";
        std::cout << "   • Channel history tracking for temporal models\n";
        
        // Demonstrate training data generation
        std::cout << "\n   Generating synthetic training data...\n";
        auto training_data = pipeline_->generate_training_data(1000, "urban");
        std::cout << "   ✓ Generated " << training_data.size() << " training samples\n";
        
        // Show configuration flexibility
        auto new_config = config_;
        new_config.mode = NeuralBeamformingPipeline::BeamformingMode::NEURAL_DNN;
        new_config.precision = NeuralBeamformingPipeline::ModelPrecision::FP16;
        new_config.enable_training_mode = true;
        
        pipeline_->update_config(new_config);
        std::cout << "   ✓ Runtime configuration updated for neural mode\n";
        
        // Note: Actual neural inference would require a trained TensorRT model
        std::cout << "\n   Note: For neural inference, provide a trained TensorRT model:\n";
        std::cout << "   config.model_path = \"beamforming_model.trt\";\n";
    }

    tensor::TensorInfo generate_channel_data() {
        // Create channel estimates tensor: [batch_size, num_antennas, num_users]
        std::vector<std::size_t> shape = {BATCH_SIZE, NUM_ANTENNAS, NUM_USERS};
        auto tensor = tensor::TensorFactory::create_tensor(
            shape, tensor::DataType::ComplexFloat32, memory_pool_
        );
        
        // Fill with synthetic channel data
        auto* data = static_cast<cuFloatComplex*>(tensor.data);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (std::size_t i = 0; i < tensor.total_elements(); ++i) {
            data[i] = make_cuFloatComplex(
                dist(gen) / sqrtf(2.0f),  // Real part
                dist(gen) / sqrtf(2.0f)   // Imaginary part
            );
        }
        
        return tensor;
    }

    tensor::TensorInfo generate_user_requirements() {
        // Create user QoS requirements tensor: [batch_size, num_users]
        std::vector<std::size_t> shape = {BATCH_SIZE, NUM_USERS};
        auto tensor = tensor::TensorFactory::create_tensor(
            shape, tensor::DataType::Float32, memory_pool_
        );
        
        // Fill with QoS requirements (normalized 0-1)
        auto* data = static_cast<float*>(tensor.data);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.5f, 1.0f);
        
        for (std::size_t i = 0; i < tensor.total_elements(); ++i) {
            data[i] = dist(gen);
        }
        
        return tensor;
    }

    tensor::TensorInfo create_output_tensor() {
        // Create beamforming weights tensor: [batch_size, num_antennas, num_users]
        std::vector<std::size_t> shape = {BATCH_SIZE, NUM_ANTENNAS, NUM_USERS};
        return tensor::TensorFactory::create_tensor(
            shape, tensor::DataType::ComplexFloat32, memory_pool_
        );
    }

    void display_beamforming_results() {
        const auto& metrics = pipeline_->get_metrics();
        
        std::cout << "\n   Performance Metrics:\n";
        std::cout << "   - Total latency: " << metrics.total_latency_ms << " ms\n";
        std::cout << "   - Inference latency: " << metrics.inference_latency_ms << " ms\n";
        std::cout << "   - Processed samples: " << metrics.processed_samples << "\n";
        std::cout << "   - Spectral efficiency: " << metrics.spectral_efficiency << " bits/s/Hz\n";
        std::cout << "   - SIR: " << metrics.signal_to_interference_ratio << " dB\n";
        
        // Calculate theoretical throughput
        float samples_per_sec = 1000.0f / metrics.total_latency_ms * BATCH_SIZE;
        std::cout << "   - Throughput: " << samples_per_sec << " samples/sec\n";
    }

    void cleanup() {
        std::cout << "\n5. Cleaning up resources...\n";
        
        if (pipeline_) {
            pipeline_->finalize();
            pipeline_.reset();
        }
        
        memory_pool_.reset();
        cuda_context_.reset();
        
        std::cout << "   ✓ All resources cleaned up\n";
    }
};

int main(int argc, char* argv[]) {
    try {
        NeuralBeamformingExample example;
        example.run();
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}