#ifndef MIMO_DETECTION_MODULE_HPP
#define MIMO_DETECTION_MODULE_HPP

#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"
#include "tensor/data_types.hpp"

#include <cuComplex.h>
#include <string>
#include <vector>
#include <memory>
#include <span>

namespace mimo_detection {

/// MIMO detection algorithm types
enum class MIMODetectionAlgorithm {
    ZERO_FORCING,           ///< Zero forcing linear detection
    MMSE,                  ///< Minimum mean square error detection
    ML_EXHAUSTIVE,         ///< Maximum likelihood exhaustive search
    SPHERE_DECODER         ///< Sphere decoding algorithm
};

/// MIMO detection parameters
struct MIMOParams {
    int num_tx_antennas{4};        ///< Number of transmit antennas
    int num_rx_antennas{4};        ///< Number of receive antennas  
    int num_subcarriers{1024};     ///< Number of subcarriers (OFDM)
    int num_ofdm_symbols{14};      ///< Number of OFDM symbols per slot
    int constellation_size{16};     ///< Constellation size (4=QPSK, 16=16QAM, 64=64QAM)
    float noise_variance{0.01f};   ///< Noise variance estimate for MMSE
    MIMODetectionAlgorithm algorithm{MIMODetectionAlgorithm::ZERO_FORCING};
    bool soft_output{false};       ///< Generate soft bits (LLRs) vs hard decisions
};

/// MIMO detection descriptor for GPU kernels
struct MIMODescriptor {
    const cuComplex* received_symbols;    ///< Received symbols [num_rx x num_subcarriers x num_symbols]
    const cuComplex* channel_matrix;      ///< Channel matrix [num_rx x num_tx x num_subcarriers]
    cuComplex* detected_symbols;          ///< Output detected symbols [num_tx x num_subcarriers x num_symbols]
    float* soft_bits;                     ///< Output soft bits/LLRs (if enabled)
    MIMOParams* params;                   ///< Detection parameters
    int total_resource_elements;          ///< Total number of resource elements to process
};

/// MIMO detector module implementing the framework interface
class MIMODetector final : public framework::pipeline::IModule,
                          public framework::pipeline::IAllocationInfoProvider,
                          public framework::pipeline::IStreamExecutor {
public:
    /**
     * Constructor
     * @param module_id Unique identifier for this module
     * @param params MIMO detection parameters
     */
    explicit MIMODetector(
        const std::string& module_id,
        const MIMOParams& params
    );
    
    ~MIMODetector() override;

    // IModule interface implementation
    [[nodiscard]] std::string_view get_type_id() const override { return "mimo_detector"; }
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
    MIMOParams params_;
    framework::pipeline::ModuleMemorySlice mem_slice_;
    
    // Port information
    std::vector<framework::pipeline::PortInfo> input_ports_;
    std::vector<framework::pipeline::PortInfo> output_ports_;
    
    // GPU resources
    MIMODescriptor* d_descriptor_;
    MIMODescriptor h_descriptor_;
    
    // Device memory pointers
    cuComplex* d_received_symbols_{nullptr};
    cuComplex* d_channel_matrix_{nullptr};
    cuComplex* d_detected_symbols_{nullptr};
    float* d_soft_bits_{nullptr};
    cuComplex* d_temp_matrix_{nullptr};  // For matrix operations
    
    // Current tensor pointers (set during set_inputs)
    const cuComplex* current_received_{nullptr};
    const cuComplex* current_channel_{nullptr};
    
    void allocate_gpu_memory();
    void deallocate_gpu_memory();
    void setup_port_info();
    cudaError_t launch_mimo_detection_kernel(cudaStream_t stream);
    size_t calculate_total_elements() const;
};

} // namespace mimo_detection

#endif // MIMO_DETECTION_MODULE_HPP