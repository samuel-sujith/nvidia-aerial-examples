#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <vector>
#include <string>
#include <complex>

namespace neural_beamforming {

/**
 * @brief Simplified neural network-based beamforming pipeline
 */
class NeuralBeamformingPipeline {
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
        FP32,  ///< 32-bit floating point
        FP16,  ///< 16-bit floating point
        INT8   ///< 8-bit integer (quantized)
    };

    /**
     * @brief Configuration for neural beamforming pipeline
     */
    struct Config {
        std::size_t num_antennas = 64;
        std::size_t num_users = 8;
        std::size_t batch_size = 32;
        BeamformingMode mode = BeamformingMode::NEURAL_DNN;
        ModelPrecision precision = ModelPrecision::FP32;
        std::string model_path = "";
        float snr_threshold_db = 10.0f;
        std::size_t history_length = 10;
    };

    /**
     * @brief Performance metrics for beamforming
     */
    struct Metrics {
        float signal_to_interference_ratio = 0.0f;
        float spectral_efficiency = 0.0f;
        float inference_latency_ms = 0.0f;
        float total_latency_ms = 0.0f;
        std::size_t processed_samples = 0;
        float model_accuracy = 0.0f;
    };

    explicit NeuralBeamformingPipeline(const Config& config);
    ~NeuralBeamformingPipeline();

    // Core processing functions
    bool initialize();
    void finalize();
    
    bool process_beamforming(
        const std::vector<std::complex<float>>& channel_input,
        const std::vector<std::complex<float>>& user_signals,
        std::vector<std::complex<float>>& beamformed_output
    );

    // Metrics and configuration
    const Metrics& get_metrics() const { return metrics_; }
    std::string_view get_pipeline_id() const { return "neural_beamforming"; }

    // Factory methods
    static Config create_default_config(std::size_t num_antennas = 64, std::size_t num_users = 8);

private:
    Config config_;
    Metrics metrics_;
    
    // CUDA resources
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
    
    // GPU memory
    std::complex<float>* d_channel_matrix_ = nullptr;
    std::complex<float>* d_beamforming_weights_ = nullptr;
    std::complex<float>* d_temp_buffer_ = nullptr;
    
    bool is_initialized_ = false;
    
    // Helper functions
    void initialize_cuda_resources();
    void cleanup_cuda_resources();
    void compute_traditional_mmse();
    void run_neural_inference();
};

/**
 * @brief Factory for creating neural beamforming pipelines
 */
class NeuralBeamformingPipelineFactory {
public:
    static std::unique_ptr<NeuralBeamformingPipeline> create(const NeuralBeamformingPipeline::Config& config);
};

} // namespace neural_beamforming