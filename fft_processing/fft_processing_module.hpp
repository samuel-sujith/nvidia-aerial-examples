#ifndef FFT_PROCESSING_MODULE_HPP
#define FFT_PROCESSING_MODULE_HPP

#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"
#include "tensor/data_types.hpp"

#include <cufft.h>
#include <cuComplex.h>
#include <string>
#include <vector>
#include <memory>
#include <span>

namespace fft_processing {

/// FFT processing types
enum class FFTDirection {
    FORWARD,    ///< Time to frequency domain (DFT)
    INVERSE     ///< Frequency to time domain (IDFT)
};

/// FFT processing parameters
struct FFTParams {
    int fft_size{1024};              ///< FFT size (power of 2)
    int num_antennas{4};             ///< Number of antennas/streams
    int batch_size{1};               ///< Number of FFTs to process in parallel
    FFTDirection direction{FFTDirection::FORWARD}; ///< FFT direction
    bool normalize{true};            ///< Apply normalization for inverse FFT
    bool enable_windowing{false};    ///< Apply windowing function
};

/// GPU kernel descriptor for FFT preprocessing/postprocessing
struct FFTDescriptor {
    const cuComplex* input_data;     ///< Input time/frequency domain data
    cuComplex* output_data;          ///< Output frequency/time domain data
    const float* window_function;    ///< Optional windowing function [fft_size]
    FFTParams* params;               ///< FFT parameters
    int total_samples;               ///< Total number of samples to process
};

/// FFT processor module implementing the framework interface
class FFTProcessor final : public framework::pipeline::IModule,
                           public framework::pipeline::IAllocationInfoProvider, 
                           public framework::pipeline::IStreamExecutor {
public:
    /**
     * Constructor
     * @param module_id Unique identifier for this module
     * @param params FFT processing parameters
     */
    explicit FFTProcessor(
        const std::string& module_id,
        const FFTParams& params
    );
    
    ~FFTProcessor() override;

    // IModule interface implementation
    [[nodiscard]] std::string_view get_type_id() const override { return "fft_processor"; }
    [[nodiscard]] std::string_view get_instance_id() const override { return module_id_; }
    
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;
    
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;
    
    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;
    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;
    
    // IAllocationInfoProvider interface
    [[nodiscard]] framework::pipeline::ModuleMemoryRequirements get_requirements() const override;
    
    [[nodiscard]] framework::pipeline::OutputPortMemoryCharacteristics
    get_output_memory_characteristics(std::string_view port_name) const override;
    
    // Setup phase
    void setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) override;
    void warmup(cudaStream_t stream) override;
    
    // Per-iteration configuration
    void configure_io(
        const framework::pipeline::DynamicParams& params,
        cudaStream_t stream
    ) override;
    
    // IModule casting methods
    framework::pipeline::IGraphNodeProvider* as_graph_node_provider() override { return nullptr; }
    framework::pipeline::IStreamExecutor* as_stream_executor() override { return this; }
    
    // IStreamExecutor interface
    void set_inputs(std::span<const framework::pipeline::PortInfo> inputs) override;
    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override;
    void execute(cudaStream_t stream) override;

private:
    std::string module_id_;
    FFTParams params_;
    
    // Port information
    std::vector<framework::pipeline::PortInfo> input_ports_;
    std::vector<framework::pipeline::PortInfo> output_ports_;
    
    // CUFFT resources
    cufftHandle fft_plan_;
    
    // GPU resources
    FFTDescriptor* d_descriptor_;
    FFTDescriptor h_descriptor_;
    
    // Device memory pointers
    cuComplex* d_input_data_{nullptr};
    cuComplex* d_output_data_{nullptr};
    float* d_window_function_{nullptr};
    
    // Current tensor pointers (set during set_inputs)
    const cuComplex* current_input_{nullptr};
    cuComplex* current_output_{nullptr};
    
    void allocate_gpu_memory();
    void deallocate_gpu_memory();
    void setup_port_info();
    void create_fft_plan();
    void destroy_fft_plan();
    void setup_windowing();
};

} // namespace fft_processing

#endif // FFT_PROCESSING_MODULE_HPP