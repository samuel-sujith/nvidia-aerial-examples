#pragma once

#include "neural_beamforming_module.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"
#include <memory>
#include <string>
#include <chrono>
#include <mutex>

namespace neural_beamforming {

/**
 * @brief Pipeline wrapper for Neural Beamforming module
 *
 * This class provides a complete pipeline for massive MIMO beamforming,
 * supporting both traditional algorithms and neural network-based approaches.
 * 
 * Pipeline Flow:
 * Input:  input_symbols + channel_estimates
 * Output: beamformed_symbols + beamforming_weights + performance_metrics
 *
 * Supported algorithms:
 * - Conventional delay-and-sum beamforming
 * - MVDR (Minimum Variance Distortionless Response)
 * - Zero-Forcing beamforming
 * - Neural Network based beamforming
 */
class NeuralBeamformingPipeline {
public:
    /**
     * @brief Configuration parameters for beamforming pipeline
     */
    struct PipelineConfig {
        BeamformingParams beamforming_params;  ///< Beamforming parameters
        std::string module_id;                 ///< Unique module identifier
        bool enable_profiling = false;         ///< Enable performance profiling
        size_t stream_buffer_size = 1024;      ///< Stream buffer size for batching
        
        PipelineConfig() {
            module_id = "neural_beamforming_pipeline";
        }
    };

    /**
     * @brief Performance metrics for the pipeline
     */
    struct PerformanceMetrics {
        size_t total_processed_frames = 0;
        double avg_processing_time_ms = 0.0;
        double peak_processing_time_ms = 0.0;
        double throughput_mbps = 0.0;
        double avg_sinr_db = 0.0;
        double beamforming_gain_db = 0.0;
        size_t total_beamformed_symbols = 0;
    };

    /**
     * @brief Constructor
     * @param config Pipeline configuration
     */
    explicit NeuralBeamformingPipeline(const PipelineConfig& config);
    
    /**
     * @brief Destructor
     */
    ~NeuralBeamformingPipeline();
    
    /**
     * @brief Initialize the pipeline
     * @return True if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Process beamforming for given input
     * @param input_symbols Input symbols from antenna array
     * @param channel_estimates Channel estimates for all users
     * @param output_symbols Beamformed output symbols
     * @param beamforming_weights Computed beamforming weights
     * @param performance_metrics SINR and other performance metrics
     * @param stream CUDA stream for processing
     * @return True if processing successful
     */
    bool process_beamforming(
        const std::vector<std::complex<float>>& input_symbols,
        const std::vector<std::complex<float>>& channel_estimates,
        std::vector<std::complex<float>>& output_symbols,
        std::vector<std::complex<float>>& beamforming_weights,
        std::vector<float>& performance_metrics,
        cudaStream_t stream = 0
    );
    
    /**
     * @brief Update beamforming parameters
     * @param new_params New parameters to apply
     * @return True if update successful
     */
    bool update_parameters(const BeamformingParams& new_params);
    
    /**
     * @brief Calculate theoretical beamforming gain
     * @param num_antennas Number of antenna elements
     * @param algorithm Beamforming algorithm
     * @return Expected beamforming gain in dB
     */
    double calculate_theoretical_gain(int num_antennas, BeamformingAlgorithm algorithm) const;
    
    /**
     * @brief Get current performance metrics
     * @return Current performance statistics
     */
    PerformanceMetrics get_performance_metrics() const;
    
    /**
     * @brief Reset performance metrics
     */
    void reset_metrics();
    
    /**
     * @brief Get steering vector for given angle
     * @param angle_degrees Angle in degrees (0-359)
     * @param steering_vector Output steering vector
     */
    void get_steering_vector(
        float angle_degrees,
        std::vector<std::complex<float>>& steering_vector
    ) const;
    
    /**
     * @brief Calculate total input symbols for current configuration
     * @return Total input symbol count
     */
    size_t calculate_input_symbols() const;
    
    /**
     * @brief Calculate total output symbols for current configuration
     * @return Total output symbol count
     */
    size_t calculate_output_symbols() const;

private:
    PipelineConfig config_;
    std::shared_ptr<NeuralBeamformer> beamformer_;
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics metrics_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
    // Memory management
    void* h_input_buffer_{nullptr};
    void* h_output_buffer_{nullptr};
    void* d_input_buffer_{nullptr};
    void* d_output_buffer_{nullptr};
    void* d_module_tensor_{nullptr};
    size_t module_tensor_bytes_{0};
    
    /**
     * @brief Process data on GPU device
     * @param d_input_symbols Device pointer to input symbols
     * @param d_channel_estimates Device pointer to channel estimates
     * @param d_output_symbols Device pointer to output symbols
     * @param d_beamforming_weights Device pointer to beamforming weights
     * @param d_performance_metrics Device pointer to performance metrics
     * @param stream CUDA stream
     * @return True if processing successful
     */
    bool process_device(
        const void* d_input_symbols,
        const void* d_channel_estimates,
        void* d_output_symbols,
        void* d_beamforming_weights,
        void* d_performance_metrics,
        cudaStream_t stream
    );
    
    /**
     * @brief Allocate internal memory buffers
     */
    void allocate_buffers();
    
    /**
     * @brief Deallocate internal memory buffers
     */
    void deallocate_buffers();
    
    /**
     * @brief Update performance metrics with new data
     * @param processing_time_ms Processing time in milliseconds
     * @param num_symbols Number of processed symbols
     * @param avg_sinr_db Average SINR in dB
     */
    void update_metrics(double processing_time_ms, size_t num_symbols, double avg_sinr_db = 0.0);
    
    /**
     * @brief Get size of input symbols buffer in bytes
     * @return Buffer size in bytes
     */
    size_t get_input_symbols_size() const;
    
    /**
     * @brief Get size of channel estimates buffer in bytes
     * @return Buffer size in bytes
     */
    size_t get_channel_estimates_size() const;
    
    /**
     * @brief Get size of output symbols buffer in bytes
     * @return Buffer size in bytes
     */
    size_t get_output_symbols_size() const;
    
    /**
     * @brief Get size of beamforming weights buffer in bytes
     * @return Buffer size in bytes
     */
    size_t get_beamforming_weights_size() const;
    
    /**
     * @brief Get size of performance metrics buffer in bytes
     * @return Buffer size in bytes
     */
    size_t get_performance_metrics_size() const;
    
    /**
     * @brief Validate input buffer sizes
     * @param input_symbols_size Size of input symbols buffer
     * @param channel_estimates_size Size of channel estimates buffer
     * @return True if sizes are valid
     */
    bool validate_input_sizes(size_t input_symbols_size, size_t channel_estimates_size) const;
};

} // namespace neural_beamforming