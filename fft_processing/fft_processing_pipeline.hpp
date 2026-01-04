#ifndef FFT_PROCESSING_PIPELINE_HPP
#define FFT_PROCESSING_PIPELINE_HPP

#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"
#include "fft_processing_module.hpp"

#include <memory>
#include <string>
#include <span>
#include <vector>

namespace fft_processing {

/**
 * FFT Processing Pipeline
 * 
 * This pipeline demonstrates:
 * - GPU-based FFT/IFFT operations using CUFFT
 * - Windowing functions and normalization
 * - Batch processing of multiple antenna streams
 * - Integration with Aerial framework patterns
 */
class FFTProcessingPipeline final : public framework::pipeline::IPipeline {
public:
    /**
     * Constructor
     * @param pipeline_id Unique identifier for this pipeline
     * @param params FFT processing parameters
     */
    explicit FFTProcessingPipeline(
        std::string pipeline_id,
        const FFTParams& params = {}
    );
    
    ~FFTProcessingPipeline() override = default;

    // ========================================================================
    // IPipeline Interface - Identification  
    // ========================================================================
    [[nodiscard]] std::string_view get_pipeline_id() const override { return pipeline_id_; }
    
    [[nodiscard]] std::size_t get_num_external_inputs() const override {
        return 1; // input_signal
    }
    
    [[nodiscard]] std::size_t get_num_external_outputs() const override {
        return 1; // output_signal
    }

    // ========================================================================
    // IPipeline Interface - Setup Phase
    // ========================================================================
    void setup() override;
    
    void warmup(cudaStream_t stream) override;
    
    void configure_io(
        const framework::pipeline::DynamicParams& params,
        std::span<const framework::pipeline::PortInfo> external_inputs,
        std::span<framework::pipeline::PortInfo> external_outputs,
        cudaStream_t stream
    ) override;
    
    void execute_stream(cudaStream_t stream) override;
    void execute_graph(cudaStream_t stream) override;

private:
    std::string pipeline_id_;
    FFTParams fft_params_;
    std::unique_ptr<FFTProcessor> fft_processor_;
    
    // Internal tensor connections
    std::vector<framework::tensor::TensorInfo> internal_tensors_;
    
    // Memory management
    void* device_memory_{nullptr};
    size_t memory_size_{0};
    
    void allocate_pipeline_memory();
    void setup_tensor_connections();
};

} // namespace fft_processing

#endif // FFT_PROCESSING_PIPELINE_HPP