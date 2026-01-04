#pragma once

#include "mimo_detection_module.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"
#include <memory>
#include <string>
#include <chrono>
#include <mutex>

namespace mimo_detection {

/**
 * @brief Pipeline wrapper for MIMO Detection module
 *
 * This class provides a complete pipeline for MIMO symbol detection,
 * handling multiple antenna configurations and detection algorithms.
 * 
 * Pipeline Flow:
 * Input: received_symbols (RX antennas x subcarriers x OFDM symbols)
 *        channel_matrix (RX x TX antennas x subcarriers)
 * Output: detected_symbols (TX antennas x subcarriers x OFDM symbols)
 *
 * Supported algorithms:
 * - Zero Forcing (ZF)
 * - Minimum Mean Square Error (MMSE)
 * - Maximum Likelihood (ML) [future]
 * - Sphere Decoding [future]
 */
class MIMODetectionPipeline {
public:
    /**
     * @brief Configuration parameters for MIMO detection pipeline
     */
    struct PipelineConfig {
        MIMOParams mimo_params;              ///< MIMO detection parameters
        std::string module_id;               ///< Unique module identifier
        bool enable_profiling = false;       ///< Enable performance profiling
        size_t stream_buffer_size = 1024;    ///< Stream buffer size for batching
        
        PipelineConfig() {
            module_id = "mimo_detection_pipeline";
        }
    };

    /**
     * @brief Constructor
     * @param config Pipeline configuration
     */
    explicit MIMODetectionPipeline(const PipelineConfig& config);

    /**
     * @brief Destructor
     */
    ~MIMODetectionPipeline();

    /**
     * @brief Initialize the pipeline
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Execute MIMO detection on input data
     * @param received_symbols Input received symbols from multiple antennas
     * @param channel_matrix Channel matrix (H) for current subframe
     * @param detected_symbols Output buffer for detected symbols
     * @param stream CUDA stream for execution
     * @return True if execution successful
     */
    bool process(
        const std::vector<std::complex<float>>& received_symbols,
        const std::vector<std::complex<float>>& channel_matrix,
        std::vector<std::complex<float>>& detected_symbols,
        cudaStream_t stream = 0
    );

    /**
     * @brief Process GPU data directly (zero-copy when possible)
     * @param d_received_symbols Device pointer to received symbols
     * @param d_channel_matrix Device pointer to channel matrix
     * @param d_detected_symbols Device pointer to output buffer
     * @param stream CUDA stream for execution
     * @return True if execution successful
     */
    bool process_device(
        const void* d_received_symbols,
        const void* d_channel_matrix,
        void* d_detected_symbols,
        cudaStream_t stream = 0
    );

    /**
     * @brief Get the underlying MIMO detector module
     * @return Pointer to MIMODetector instance
     */
    std::shared_ptr<MIMODetector> get_detector() const { return detector_; }

    /**
     * @brief Update MIMO parameters dynamically
     * @param new_params New MIMO parameters
     * @return True if update successful
     */
    bool update_parameters(const MIMOParams& new_params);

    /**
     * @brief Get current performance metrics
     * @return Performance statistics
     */
    struct PerformanceMetrics {
        double avg_processing_time_ms = 0.0;
        double peak_processing_time_ms = 0.0;
        size_t total_processed_frames = 0;
        double throughput_mbps = 0.0;
    };
    PerformanceMetrics get_performance_metrics() const;

    /**
     * @brief Reset performance counters
     */
    void reset_metrics();

private:
    PipelineConfig config_;                          ///< Pipeline configuration
    std::shared_ptr<MIMODetector> detector_;        ///< MIMO detector module
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics metrics_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
    // Internal buffers for host-device transfers
    void* h_received_buffer_ = nullptr;
    void* h_channel_buffer_ = nullptr;
    void* h_detected_buffer_ = nullptr;
    void* d_received_buffer_ = nullptr;
    void* d_channel_buffer_ = nullptr;
    void* d_detected_buffer_ = nullptr;
    
    /**
     * @brief Allocate internal buffers
     */
    void allocate_buffers();

    /**
     * @brief Deallocate internal buffers
     */
    void deallocate_buffers();

    /**
     * @brief Update performance metrics
     * @param processing_time_ms Processing time in milliseconds
     */
    void update_metrics(double processing_time_ms);

    /**
     * @brief Calculate data size in bytes for received symbols
     */
    size_t get_received_symbols_size() const;

    /**
     * @brief Calculate data size in bytes for channel matrix
     */
    size_t get_channel_matrix_size() const;

    /**
     * @brief Calculate data size in bytes for detected symbols
     */
    size_t get_detected_symbols_size() const;
};

} // namespace mimo_detection