#pragma once

#include <pipeline/ipipeline.hpp>
#include <aerial_framework/tensor/tensor_info.hpp>
#include <aerial_framework/task/task_result.hpp>
#include <aerial_framework/memory/memory_pool.hpp>
#include <aerial_framework/cuda_utils/cuda_context.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <NvInfer.h>
#include <memory>
#include <vector>
#include <string>

namespace aerial::examples::neural_beamforming {

/**
 * @brief Neural network-based beamforming pipeline for 5G/6G antenna arrays
 * 
 * This pipeline demonstrates how to integrate ML models with the Aerial framework
 * for real-time beamforming optimization. It includes:
 * - Model training data generation
 * - TensorRT inference engine
 * - GPU-accelerated preprocessing/postprocessing
 * - Performance benchmarking and validation
 */
class NeuralBeamformingPipeline final : public pipeline::IPipeline {
public:
    /**
     * @brief Beamforming algorithms available
     */
    enum class BeamformingMode {
        TRADITIONAL_MMSE,    ///< Traditional MMSE beamforming (baseline)
        NEURAL_DNN,          ///< Deep Neural Network beamforming
        NEURAL_CNN,          ///< Convolutional Neural Network beamforming
        HYBRID              ///< Hybrid traditional + neural approach
    };

    /**
     * @brief Model precision for inference
     */
    enum class ModelPrecision {
        FP32,               ///< Full precision (32-bit float)
        FP16,               ///< Half precision (16-bit float)
        INT8                ///< Quantized (8-bit integer)
    };

    /**
     * @brief Configuration parameters for neural beamforming
     */
    struct Config {
        std::size_t num_antennas = 64;          ///< Number of antenna elements
        std::size_t num_users = 8;              ///< Number of simultaneous users
        std::size_t batch_size = 32;            ///< Batch size for inference
        BeamformingMode mode = BeamformingMode::NEURAL_DNN;
        ModelPrecision precision = ModelPrecision::FP16;
        std::string model_path;                 ///< Path to TensorRT engine file
        bool enable_training_mode = false;      ///< Enable training data collection
        float learning_rate = 1e-4f;           ///< Learning rate for online adaptation
        std::size_t history_length = 10;       ///< Channel history for temporal models
    };

    /**
     * @brief Performance metrics for beamforming
     */
    struct Metrics {
        float signal_to_interference_ratio = 0.0f;  ///< Average SIR (dB)
        float spectral_efficiency = 0.0f;           ///< Bits/s/Hz
        float inference_latency_ms = 0.0f;          ///< Neural network inference time
        float total_latency_ms = 0.0f;              ///< End-to-end processing time
        std::size_t processed_samples = 0;          ///< Total samples processed
        float model_accuracy = 0.0f;                ///< Model validation accuracy
    };

    /**
     * @brief Construct neural beamforming pipeline
     * @param config Configuration parameters
     * @param memory_pool Shared memory pool for tensor allocation
     * @param cuda_context CUDA execution context
     */
    NeuralBeamformingPipeline(
        const Config& config,
        std::shared_ptr<memory::MemoryPool> memory_pool,
        std::shared_ptr<cuda_utils::CudaContext> cuda_context
    );

    /**
     * @brief Destructor - cleanup TensorRT and CUDA resources
     */
    ~NeuralBeamformingPipeline() override;

    // IPipeline interface implementation
    std::string_view get_pipeline_id() const override {
        return "neural_beamforming_v1";
    }

    task::TaskResult initialize(const std::vector<tensor::TensorInfo>& inputs) override;
    task::TaskResult execute(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs,
        const task::CancellationToken& token = {}
    ) override;
    void finalize() override;

    /**
     * @brief Process beamforming with neural network
     * @param channel_estimates Channel state information (CSI) tensor
     * @param user_requirements QoS requirements per user
     * @param beamforming_weights Output beamforming weight matrix
     * @return Processing result with performance metrics
     */
    task::TaskResult process_beamforming(
        const tensor::TensorInfo& channel_estimates,
        const tensor::TensorInfo& user_requirements,
        tensor::TensorInfo& beamforming_weights
    );

    /**
     * @brief Train or fine-tune the neural network model
     * @param training_data Channel and performance data for training
     * @param validation_data Data for model validation
     * @param epochs Number of training epochs
     * @return Training result with final metrics
     */
    task::TaskResult train_model(
        const std::vector<tensor::TensorInfo>& training_data,
        const std::vector<tensor::TensorInfo>& validation_data,
        std::size_t epochs = 100
    );

    /**
     * @brief Load pre-trained TensorRT engine
     * @param engine_path Path to serialized TensorRT engine
     * @return Load operation result
     */
    task::TaskResult load_tensorrt_engine(const std::string& engine_path);

    /**
     * @brief Export current model to TensorRT for deployment
     * @param output_path Path to save TensorRT engine
     * @param precision Target inference precision
     * @return Export operation result
     */
    task::TaskResult export_tensorrt_engine(
        const std::string& output_path, 
        ModelPrecision precision = ModelPrecision::FP16
    );

    /**
     * @brief Update configuration at runtime
     * @param new_config New configuration parameters
     */
    void update_config(const Config& new_config);

    /**
     * @brief Get current performance metrics
     */
    const Metrics& get_metrics() const { return metrics_; }

    /**
     * @brief Reset performance counters
     */
    void reset_metrics();

    /**
     * @brief Generate synthetic training data for model development
     * @param num_samples Number of training samples to generate
     * @param scenario Channel scenario (urban, rural, indoor, etc.)
     * @return Generated training data tensors
     */
    std::vector<tensor::TensorInfo> generate_training_data(
        std::size_t num_samples,
        const std::string& scenario = "urban"
    );

private:
    // Configuration and state
    Config config_;
    Metrics metrics_;
    bool is_initialized_ = false;

    // Aerial framework components
    std::shared_ptr<memory::MemoryPool> memory_pool_;
    std::shared_ptr<cuda_utils::CudaContext> cuda_context_;

    // CUDA resources
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;

    // TensorRT inference engine
    std::unique_ptr<nvinfer1::IRuntime> tensorrt_runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> tensorrt_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> tensorrt_context_;

    // Model I/O tensors
    std::vector<tensor::TensorInfo> model_inputs_;
    std::vector<tensor::TensorInfo> model_outputs_;
    std::vector<void*> tensorrt_bindings_;

    // Channel simulation and preprocessing
    std::vector<cuFloatComplex> channel_history_;
    std::vector<float> user_qos_requirements_;
    std::vector<cuFloatComplex> traditional_weights_;

    // Performance monitoring
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;

    // Internal methods
    void initialize_cuda_resources();
    void cleanup_cuda_resources();
    void initialize_tensorrt();
    void preprocess_channel_data(
        const tensor::TensorInfo& raw_channel,
        tensor::TensorInfo& preprocessed
    );
    void run_neural_inference(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs
    );
    void postprocess_beamforming_weights(
        const tensor::TensorInfo& raw_weights,
        tensor::TensorInfo& final_weights
    );
    void compute_traditional_mmse_baseline(
        const tensor::TensorInfo& channel_estimates,
        tensor::TensorInfo& mmse_weights
    );
    void update_performance_metrics();
    void validate_inputs(const std::vector<tensor::TensorInfo>& inputs);
    void validate_outputs(const std::vector<tensor::TensorInfo>& outputs);

    // Training utilities
    void collect_training_sample(
        const tensor::TensorInfo& channel_input,
        const tensor::TensorInfo& optimal_weights
    );
    float evaluate_beamforming_performance(
        const tensor::TensorInfo& weights,
        const tensor::TensorInfo& channel_estimates
    );
};

/**
 * @brief Factory for creating neural beamforming pipelines
 */
class NeuralBeamformingPipelineFactory {
public:
    static std::unique_ptr<NeuralBeamformingPipeline> create(
        const NeuralBeamformingPipeline::Config& config,
        std::shared_ptr<memory::MemoryPool> memory_pool,
        std::shared_ptr<cuda_utils::CudaContext> cuda_context
    );

    static NeuralBeamformingPipeline::Config create_default_config(
        std::size_t num_antennas = 64,
        std::size_t num_users = 8
    );

    static NeuralBeamformingPipeline::Config load_config_from_file(
        const std::string& config_file
    );
};

} // namespace aerial::examples::neural_beamforming