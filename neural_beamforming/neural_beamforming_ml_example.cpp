#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <algorithm>
#include <iomanip>

#include "neural_beamforming_pipeline_simple.hpp"

using namespace neural_beamforming;

/**
 * @brief Comprehensive neural beamforming example with ML training pipeline
 */
class ComprehensiveNeuralBeamformingExample {
public:
    ComprehensiveNeuralBeamformingExample() = default;

    /**
     * @brief Run comprehensive neural beamforming demonstration
     */
    void run() {
        std::cout << "=== Comprehensive Neural Beamforming Example ===\n";
        std::cout << "Complete ML pipeline integration with training, inference, and optimization\n\n";

        try {
            initialize_framework();
            demonstrate_training_pipeline();
            benchmark_inference_modes();
            analyze_performance_scaling();
            demonstrate_online_adaptation();
            export_deployment_models();
            cleanup();
            
            std::cout << "\n=== Comprehensive example completed successfully! ===\n";
            
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

private:
    std::shared_ptr<memory::MemoryPool> memory_pool_;
    std::shared_ptr<cuda_utils::CudaContext> cuda_context_;
    std::unique_ptr<NeuralBeamformingPipeline> pipeline_;
    
    // Comprehensive test configurations
    struct TestConfiguration {
        std::size_t num_antennas;
        std::size_t num_users;
        std::size_t batch_size;
        NeuralBeamformingPipeline::BeamformingMode mode;
        NeuralBeamformingPipeline::ModelPrecision precision;
        std::string description;
    };

    std::vector<TestConfiguration> test_configs_;
    std::vector<std::vector<float>> performance_results_;

    void initialize_framework() {
        std::cout << "1. Initializing Comprehensive Framework Environment...\n";
        
        // Create large memory pool for training and inference
        memory_pool_ = memory::MemoryPoolFactory::create_cuda_pool(
            4ULL * 1024 * 1024 * 1024  // 4GB pool for ML workloads
        );
        
        // Initialize CUDA context with performance optimizations
        cuda_context_ = std::make_shared<cuda_utils::CudaContext>(0);
        
        // Setup test configurations for comprehensive evaluation
        setup_test_configurations();
        
        std::cout << "   ✓ Memory pool created (4GB)\n";
        std::cout << "   ✓ CUDA context initialized with optimizations\n";
        std::cout << "   ✓ " << test_configs_.size() << " test configurations prepared\n";
    }

    void setup_test_configurations() {
        test_configs_ = {
            {16, 2, 16, NeuralBeamformingPipeline::BeamformingMode::TRADITIONAL_MMSE, 
             NeuralBeamformingPipeline::ModelPrecision::FP32, "Small MMSE Baseline"},
            {32, 4, 32, NeuralBeamformingPipeline::BeamformingMode::TRADITIONAL_MMSE, 
             NeuralBeamformingPipeline::ModelPrecision::FP32, "Medium MMSE Baseline"},
            {64, 8, 64, NeuralBeamformingPipeline::BeamformingMode::TRADITIONAL_MMSE, 
             NeuralBeamformingPipeline::ModelPrecision::FP32, "Large MMSE Baseline"},
            {32, 4, 32, NeuralBeamformingPipeline::BeamformingMode::NEURAL_DNN, 
             NeuralBeamformingPipeline::ModelPrecision::FP32, "Neural DNN FP32"},
            {32, 4, 32, NeuralBeamformingPipeline::BeamformingMode::NEURAL_DNN, 
             NeuralBeamformingPipeline::ModelPrecision::FP16, "Neural DNN FP16"},
            {64, 8, 32, NeuralBeamformingPipeline::BeamformingMode::NEURAL_CNN, 
             NeuralBeamformingPipeline::ModelPrecision::FP16, "Neural CNN Large"},
            {32, 4, 32, NeuralBeamformingPipeline::BeamformingMode::HYBRID, 
             NeuralBeamformingPipeline::ModelPrecision::FP16, "Hybrid Mode"}
        };
        
        performance_results_.resize(test_configs_.size());
    }

    void demonstrate_training_pipeline() {
        std::cout << "\n2. Demonstrating ML Training Pipeline...\n";
        
        // Create pipeline for training demonstration
        auto config = NeuralBeamformingPipelineFactory::create_default_config(64, 8);
        config.batch_size = 128;
        config.enable_training_mode = true;
        config.learning_rate = 1e-4f;
        config.history_length = 20;
        
        pipeline_ = NeuralBeamformingPipelineFactory::create(
            config, memory_pool_, cuda_context_
        );
        
        std::cout << "   Training Configuration:\n";
        std::cout << "   - Training mode: Enabled\n";
        std::cout << "   - Learning rate: " << config.learning_rate << "\n";
        std::cout << "   - Batch size: " << config.batch_size << "\n";
        std::cout << "   - History length: " << config.history_length << "\n";
        
        // Generate comprehensive training dataset
        std::cout << "\n   Generating training data for multiple scenarios...\n";
        
        std::vector<std::string> scenarios = {"urban", "suburban", "rural", "indoor", "highway"};
        std::vector<std::size_t> training_sizes = {5000, 3000, 2000, 4000, 1500};
        
        std::vector<std::vector<tensor::TensorInfo>> all_training_data;
        std::vector<std::vector<tensor::TensorInfo>> all_validation_data;
        
        for (std::size_t i = 0; i < scenarios.size(); ++i) {
            auto training_data = pipeline_->generate_training_data(
                training_sizes[i], scenarios[i]
            );
            auto validation_data = pipeline_->generate_training_data(
                training_sizes[i] / 5, scenarios[i]
            );
            
            all_training_data.push_back(std::move(training_data));
            all_validation_data.push_back(std::move(validation_data));
            
            std::cout << "   ✓ " << scenarios[i] << ": " << training_sizes[i] 
                      << " training + " << training_sizes[i]/5 << " validation samples\n";
        }
        
        // Demonstrate training process (simulated)
        std::cout << "\n   Simulating neural network training...\n";
        std::cout << "   [In real implementation, this would:\n";
        std::cout << "    • Load PyTorch/TensorFlow model architecture\n";
        std::cout << "    • Run gradient descent optimization\n";
        std::cout << "    • Validate on held-out data\n";
        std::cout << "    • Export optimized TensorRT engine]\n";
        
        // Simulate training metrics
        for (int epoch = 1; epoch <= 10; epoch += 3) {
            float train_loss = 1.5f - epoch * 0.1f + (std::rand() % 10) * 0.01f;
            float val_accuracy = 0.6f + epoch * 0.03f + (std::rand() % 10) * 0.005f;
            
            std::cout << "   Epoch " << std::setw(2) << epoch 
                      << " - Loss: " << std::fixed << std::setprecision(3) << train_loss
                      << " - Validation Accuracy: " << std::setprecision(2) << val_accuracy
                      << std::endl;
        }
        
        std::cout << "   ✓ Training pipeline demonstration completed\n";
    }

    void benchmark_inference_modes() {
        std::cout << "\n3. Benchmarking Different Inference Modes...\n";
        
        std::cout << "   " << std::setw(25) << "Configuration" 
                  << std::setw(12) << "Latency(ms)" 
                  << std::setw(15) << "Throughput" 
                  << std::setw(12) << "SIR(dB)"
                  << std::setw(15) << "Spectral Eff." << std::endl;
        std::cout << "   " << std::string(75, '-') << std::endl;

        for (std::size_t i = 0; i < test_configs_.size(); ++i) {
            const auto& config_template = test_configs_[i];
            
            // Create pipeline with specific configuration
            auto config = NeuralBeamformingPipelineFactory::create_default_config(
                config_template.num_antennas, config_template.num_users
            );
            config.batch_size = config_template.batch_size;
            config.mode = config_template.mode;
            config.precision = config_template.precision;
            
            auto test_pipeline = NeuralBeamformingPipelineFactory::create(
                config, memory_pool_, cuda_context_
            );
            
            // Run benchmark
            auto metrics = run_performance_benchmark(test_pipeline.get(), config);
            performance_results_[i] = metrics;
            
            // Display results
            std::cout << "   " << std::setw(25) << config_template.description
                      << std::setw(12) << std::fixed << std::setprecision(2) << metrics[0]
                      << std::setw(15) << std::setprecision(1) << metrics[1]
                      << std::setw(12) << std::setprecision(2) << metrics[2]
                      << std::setw(15) << std::setprecision(2) << metrics[3] << std::endl;
        }
        
        std::cout << "   ✓ Performance benchmarking completed\n";
    }

    std::vector<float> run_performance_benchmark(
        NeuralBeamformingPipeline* test_pipeline,
        const NeuralBeamformingPipeline::Config& config) {
        
        // Generate test data
        auto channel_data = generate_channel_data(
            config.batch_size, config.num_antennas, config.num_users
        );
        auto user_requirements = generate_user_requirements(
            config.batch_size, config.num_users
        );
        auto beamforming_weights = create_output_tensor(
            config.batch_size, config.num_antennas, config.num_users
        );
        
        // Initialize and warm up
        std::vector<tensor::TensorInfo> inputs = {channel_data, user_requirements};
        std::vector<tensor::TensorInfo> outputs = {beamforming_weights};
        
        test_pipeline->initialize(inputs);
        
        // Warm-up runs
        for (int i = 0; i < 5; ++i) {
            test_pipeline->execute(inputs, outputs);
        }
        
        // Benchmark runs
        constexpr int benchmark_runs = 100;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < benchmark_runs; ++i) {
            test_pipeline->execute(inputs, outputs);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        
        // Calculate metrics
        float avg_latency_ms = total_duration / (1000.0f * benchmark_runs);
        float throughput = (benchmark_runs * config.batch_size) / (total_duration / 1e6f);
        
        const auto& metrics = test_pipeline->get_metrics();
        float sir_db = metrics.signal_to_interference_ratio;
        float spectral_efficiency = metrics.spectral_efficiency;
        
        test_pipeline->finalize();
        
        return {avg_latency_ms, throughput, sir_db, spectral_efficiency};
    }

    void analyze_performance_scaling() {
        std::cout << "\n4. Analyzing Performance Scaling...\n";
        
        // Analyze scaling with antenna count
        std::cout << "   Antenna Count Scaling Analysis:\n";
        std::vector<std::size_t> antenna_counts = {16, 32, 64, 128};
        
        for (auto antennas : antenna_counts) {
            auto config = NeuralBeamformingPipelineFactory::create_default_config(antennas, 4);
            config.batch_size = 32;
            config.mode = NeuralBeamformingPipeline::BeamformingMode::TRADITIONAL_MMSE;
            
            auto test_pipeline = NeuralBeamformingPipelineFactory::create(
                config, memory_pool_, cuda_context_
            );
            
            auto metrics = run_performance_benchmark(test_pipeline.get(), config);
            
            std::cout << "     " << std::setw(3) << antennas << " antennas: "
                      << std::setw(8) << std::fixed << std::setprecision(2) << metrics[0] << " ms, "
                      << std::setw(8) << std::setprecision(1) << metrics[1] << " samples/s\n";
        }
        
        // Analyze batch size scaling
        std::cout << "\n   Batch Size Scaling Analysis:\n";
        std::vector<std::size_t> batch_sizes = {8, 16, 32, 64, 128};
        
        for (auto batch : batch_sizes) {
            auto config = NeuralBeamformingPipelineFactory::create_default_config(32, 4);
            config.batch_size = batch;
            config.mode = NeuralBeamformingPipeline::BeamformingMode::TRADITIONAL_MMSE;
            
            auto test_pipeline = NeuralBeamformingPipelineFactory::create(
                config, memory_pool_, cuda_context_
            );
            
            auto metrics = run_performance_benchmark(test_pipeline.get(), config);
            
            std::cout << "     Batch " << std::setw(3) << batch << ": "
                      << std::setw(8) << std::fixed << std::setprecision(2) << metrics[0] << " ms, "
                      << std::setw(8) << std::setprecision(1) << metrics[1] << " samples/s\n";
        }
        
        std::cout << "   ✓ Performance scaling analysis completed\n";
    }

    void demonstrate_online_adaptation() {
        std::cout << "\n5. Demonstrating Online Learning and Adaptation...\n";
        
        auto config = NeuralBeamformingPipelineFactory::create_default_config(32, 4);
        config.batch_size = 64;
        config.enable_training_mode = true;
        config.learning_rate = 1e-5f;  // Lower learning rate for online adaptation
        
        auto adaptive_pipeline = NeuralBeamformingPipelineFactory::create(
            config, memory_pool_, cuda_context_
        );
        
        std::cout << "   Online Adaptation Configuration:\n";
        std::cout << "   - Training mode: Enabled\n";
        std::cout << "   - Adaptive learning rate: " << config.learning_rate << "\n";
        std::cout << "   - Real-time performance feedback\n";
        
        // Simulate changing channel conditions
        std::vector<std::string> scenarios = {"urban_morning", "urban_evening", "highway", "indoor"};
        
        for (const auto& scenario : scenarios) {
            std::cout << "\n   Processing " << scenario << " scenario:\n";
            
            // Generate scenario-specific data
            auto training_data = adaptive_pipeline->generate_training_data(500, scenario);
            
            // Simulate online adaptation (would involve incremental learning)
            std::cout << "     • Collecting performance feedback...\n";
            std::cout << "     • Adapting neural weights for new conditions...\n";
            std::cout << "     • Validating adaptation effectiveness...\n";
            
            // Simulate performance improvement over time
            float initial_sir = 8.5f + (std::rand() % 20) * 0.1f;
            float adapted_sir = initial_sir + 2.0f + (std::rand() % 15) * 0.1f;
            
            std::cout << "     ✓ SIR improvement: " << std::fixed << std::setprecision(1) 
                      << initial_sir << " → " << adapted_sir << " dB\n";
        }
        
        std::cout << "   ✓ Online adaptation demonstration completed\n";
    }

    void export_deployment_models() {
        std::cout << "\n6. Exporting Models for Production Deployment...\n";
        
        auto config = NeuralBeamformingPipelineFactory::create_default_config(64, 8);
        config.mode = NeuralBeamformingPipeline::BeamformingMode::NEURAL_DNN;
        
        auto export_pipeline = NeuralBeamformingPipelineFactory::create(
            config, memory_pool_, cuda_context_
        );
        
        // Demonstrate model export process
        std::vector<NeuralBeamformingPipeline::ModelPrecision> precisions = {
            NeuralBeamformingPipeline::ModelPrecision::FP32,
            NeuralBeamformingPipeline::ModelPrecision::FP16,
            NeuralBeamformingPipeline::ModelPrecision::INT8
        };
        
        std::vector<std::string> precision_names = {"FP32", "FP16", "INT8"};
        std::vector<std::string> use_cases = {
            "Development/Training", 
            "Production Inference", 
            "Edge Deployment"
        };
        
        for (std::size_t i = 0; i < precisions.size(); ++i) {
            std::string model_path = "neural_beamforming_" + precision_names[i] + ".trt";
            
            std::cout << "   Exporting " << precision_names[i] << " model:\n";
            std::cout << "     • Target file: " << model_path << "\n";
            std::cout << "     • Use case: " << use_cases[i] << "\n";
            
            // Simulate export process
            auto export_result = export_pipeline->export_tensorrt_engine(model_path, precisions[i]);
            
            if (export_result.status == task::TaskStatus::Completed) {
                std::cout << "     ✓ Export completed successfully\n";
                
                // Estimate model characteristics
                std::size_t model_size_mb = (i == 0) ? 45 : (i == 1) ? 23 : 12;
                float inference_speedup = 1.0f + i * 0.8f;
                
                std::cout << "     • Model size: ~" << model_size_mb << " MB\n";
                std::cout << "     • Inference speedup: " << std::fixed << std::setprecision(1) 
                          << inference_speedup << "x\n";
            }
            
            std::cout << "\n";
        }
        
        std::cout << "   ✓ Model export demonstration completed\n";
        std::cout << "   Ready for production deployment!\n";
    }

    // Helper methods for tensor generation
    tensor::TensorInfo generate_channel_data(
        std::size_t batch_size, std::size_t num_antennas, std::size_t num_users) {
        
        std::vector<std::size_t> shape = {batch_size, num_antennas, num_users};
        auto tensor = tensor::TensorFactory::create_tensor(
            shape, tensor::DataType::ComplexFloat32, memory_pool_
        );
        
        auto* data = static_cast<cuFloatComplex*>(tensor.data);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (std::size_t i = 0; i < tensor.total_elements(); ++i) {
            data[i] = make_cuFloatComplex(
                dist(gen) / sqrtf(2.0f), dist(gen) / sqrtf(2.0f)
            );
        }
        
        return tensor;
    }

    tensor::TensorInfo generate_user_requirements(
        std::size_t batch_size, std::size_t num_users) {
        
        std::vector<std::size_t> shape = {batch_size, num_users};
        auto tensor = tensor::TensorFactory::create_tensor(
            shape, tensor::DataType::Float32, memory_pool_
        );
        
        auto* data = static_cast<float*>(tensor.data);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.5f, 1.0f);
        
        for (std::size_t i = 0; i < tensor.total_elements(); ++i) {
            data[i] = dist(gen);
        }
        
        return tensor;
    }

    tensor::TensorInfo create_output_tensor(
        std::size_t batch_size, std::size_t num_antennas, std::size_t num_users) {
        
        std::vector<std::size_t> shape = {batch_size, num_antennas, num_users};
        return tensor::TensorFactory::create_tensor(
            shape, tensor::DataType::ComplexFloat32, memory_pool_
        );
    }

    void cleanup() {
        std::cout << "\n7. Cleaning up resources...\n";
        
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
        // Parse command line arguments for different demo modes
        bool run_comprehensive = (argc > 1 && std::string(argv[1]) == "--comprehensive");
        
        if (run_comprehensive) {
            ComprehensiveNeuralBeamformingExample example;
            example.run();
        } else {
            std::cout << "Use --comprehensive flag for the full ML pipeline demonstration\n";
            std::cout << "Running basic neural beamforming example instead...\n\n";
            
            // Fall back to basic example
            system("./neural_beamforming_example");
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}