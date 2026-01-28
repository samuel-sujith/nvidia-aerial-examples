#pragma once

#include "modulation_mapping_module.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"
#include <memory>
#include <string>
#include <chrono>
#include <mutex>

namespace modulation_mapping {

/**
 * @brief Pipeline wrapper for Modulation Mapping module
 *
 * This class provides a complete pipeline for digital modulation and demodulation,
 * supporting multiple modulation schemes and constellation mappings.
 * 
 * Pipeline Flow:
 * Modulation:   input_bits -> output_symbols
 * Demodulation: input_symbols -> output_bits (+ soft_bits)
 * Both:         supports round-trip processing
 *
 * Supported modulation schemes:
 * - BPSK (Binary Phase Shift Keying)
 * - QPSK (Quadrature Phase Shift Keying)
 * - 16-QAM (16-Quadrature Amplitude Modulation)
 * - 64-QAM (64-Quadrature Amplitude Modulation)
 * - 256-QAM (256-Quadrature Amplitude Modulation)
 */
class ModulationPipeline {
public:
    /**
     * @brief Configuration parameters for modulation pipeline
     */
    struct PipelineConfig {
        ModulationParams modulation_params;  ///< Modulation parameters
        std::string module_id;               ///< Unique module identifier
        bool enable_profiling = false;       ///< Enable performance profiling
        size_t stream_buffer_size = 1024;    ///< Stream buffer size for batching
        
        PipelineConfig() {
            module_id = "modulation_pipeline";
        }
    };

    /**
     * @brief Constructor
     * @param config Pipeline configuration
     */
    explicit ModulationPipeline(const PipelineConfig& config);

    /**
     * @brief Destructor
     */
    ~ModulationPipeline();

    /**
     * @brief Initialize the pipeline
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Execute modulation (bits to symbols)
     * @param input_bits Input bit stream
     * @param output_symbols Output modulated symbols
     * @param stream CUDA stream for execution
     * @return True if execution successful
     */
    bool modulate(
        const std::vector<uint8_t>& input_bits,
        std::vector<std::complex<float>>& output_symbols,
        cudaStream_t stream = 0
    );

    /**
     * @brief Execute demodulation (symbols to bits)
     * @param input_symbols Input received symbols
     * @param output_bits Output demodulated bits
     * @param soft_bits Output soft bits (optional)
     * @param stream CUDA stream for execution
     * @return True if execution successful
     */
    bool demodulate(
        const std::vector<std::complex<float>>& input_symbols,
        std::vector<uint8_t>& output_bits,
        std::vector<float>* soft_bits = nullptr,
        cudaStream_t stream = 0
    );

    /**
     * @brief Process GPU data directly (zero-copy when possible)
     * @param d_input_data Device pointer to input data (bits or symbols)
     * @param d_output_data Device pointer to output data (symbols or bits)
     * @param is_modulation True for modulation, false for demodulation
     * @param stream CUDA stream for execution
     * @return True if execution successful
     */
    bool process_device(
        const void* d_input_data,
        void* d_output_data,
        bool is_modulation,
        cudaStream_t stream = 0
    );

    /**
     * @brief Get the underlying modulation mapper module
     * @return Pointer to ModulationMapper instance
     */
    std::shared_ptr<ModulationMapper> get_mapper() const { return mapper_; }

    /**
     * @brief Update modulation parameters dynamically
     * @param new_params New modulation parameters
     * @return True if update successful
     */
    bool update_parameters(const ModulationParams& new_params);

    /**
     * @brief Calculate theoretical Symbol Error Rate for given SNR
     * @param snr_db Signal-to-noise ratio in dB
     * @return Theoretical SER
     */
    double calculate_theoretical_ser(double snr_db) const;

    /**
     * @brief Get current performance metrics
     * @return Performance statistics
     */
    struct PerformanceMetrics {
        double avg_processing_time_ms = 0.0;
        double peak_processing_time_ms = 0.0;
        size_t total_processed_frames = 0;
        double throughput_mbps = 0.0;
        double avg_evm_percent = 0.0;
        size_t total_symbol_errors = 0;
        double symbol_error_rate = 0.0;
    };
    PerformanceMetrics get_performance_metrics() const;

    /**
     * @brief Reset performance counters
     */
    void reset_metrics();

    /**
     * @brief Generate test constellation diagram data
     * @param points Output constellation points
     * @param labels Output bit labels for each point
     */
    void get_constellation_diagram(
        std::vector<std::complex<float>>& points,
        std::vector<std::string>& labels
    ) const;
    
    /**
     * @brief Calculate total number of bits for current configuration
     * @return Total bit count
     */
    size_t calculate_total_bits() const;
    
    /**
     * @brief Calculate total number of symbols for current configuration
     * @return Total symbol count
     */
    size_t calculate_total_symbols() const;

private:
    PipelineConfig config_;                          ///< Pipeline configuration
    std::shared_ptr<ModulationMapper> mapper_;      ///< Modulation mapper module
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics metrics_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
    // Internal buffers for host-device transfers
    void* h_input_buffer_ = nullptr;
    void* h_output_buffer_ = nullptr;
    void* d_input_buffer_ = nullptr;
    void* d_output_buffer_ = nullptr;
    void* d_module_tensor_ = nullptr;
    size_t module_tensor_bytes_ = 0;
    std::byte* static_desc_cpu_ = nullptr;
    std::byte* static_desc_gpu_ = nullptr;
    std::byte* dynamic_desc_cpu_ = nullptr;
    std::byte* dynamic_desc_gpu_ = nullptr;
    size_t static_desc_bytes_ = 0;
    size_t dynamic_desc_bytes_ = 0;
    
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
     * @param num_symbols Number of symbols processed
     * @param num_errors Number of symbol errors (if available)
     */
    void update_metrics(double processing_time_ms, size_t num_symbols, size_t num_errors = 0);

    /**
     * @brief Calculate data size in bytes for bits
     */
    size_t get_bits_size() const;

    /**
     * @brief Calculate data size in bytes for symbols
     */
    size_t get_symbols_size() const;

    /**
     * @brief Validate input data sizes
     * @param input_size Input data size
     * @param expected_size Expected size
     * @return True if sizes match
     */
    bool validate_input_size(size_t input_size, size_t expected_size) const;
};

} // namespace modulation_mapping