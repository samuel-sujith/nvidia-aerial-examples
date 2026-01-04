#ifndef MODULATION_MAPPING_MODULE_HPP
#define MODULATION_MAPPING_MODULE_HPP

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

namespace modulation_mapping {

/// Supported modulation schemes
enum class ModulationScheme {
    BPSK,       ///< Binary Phase Shift Keying
    QPSK,       ///< Quadrature Phase Shift Keying  
    QAM16,      ///< 16-Quadrature Amplitude Modulation
    QAM64,      ///< 64-Quadrature Amplitude Modulation
    QAM256      ///< 256-Quadrature Amplitude Modulation
};

/// Processing direction
enum class ProcessingMode {
    MODULATION,      ///< Bits to symbols (modulation)
    DEMODULATION,    ///< Symbols to bits (demodulation)
    BOTH            ///< Both modulation and demodulation
};

/// Modulation mapping parameters
struct ModulationParams {
    ModulationScheme scheme{ModulationScheme::QPSK};
    ProcessingMode mode{ProcessingMode::BOTH};
    int num_subcarriers{1024};       ///< Number of subcarriers
    int num_ofdm_symbols{14};        ///< Number of OFDM symbols per slot
    bool soft_output{false};         ///< Generate soft bits for demodulation
    float noise_variance{0.1f};      ///< Noise power for LLR computation
    float constellation_scaling{1.0f}; ///< Constellation normalization factor
    bool enable_evm_calculation{true}; ///< Calculate Error Vector Magnitude
};

/// GPU kernel descriptor for modulation mapping
struct ModulationDescriptor {
    // Input/output data pointers
    const uint8_t* input_bits;        ///< Input bit stream [total_bits]
    const cuComplex* input_symbols;   ///< Input symbol stream [total_symbols]
    cuComplex* output_symbols;        ///< Output modulated symbols [total_symbols]
    uint8_t* output_bits;            ///< Output demodulated bits [total_bits]
    float* soft_bits;                ///< Output soft bits/LLRs [total_bits]
    float* evm_values;               ///< Error Vector Magnitude values [total_symbols]
    
    // Configuration
    ModulationParams* params;         ///< Modulation parameters
    cuComplex* constellation;         ///< Constellation points [constellation_size]
    
    // Dimensions
    int total_symbols;               ///< Total number of symbols
    int total_bits;                  ///< Total number of bits
    int bits_per_symbol;             ///< Bits per constellation symbol
    int constellation_size;          ///< Number of constellation points
};

/// Modulation mapper module implementing the framework interface
class ModulationMapper final : public framework::pipeline::IModule,
                              public framework::pipeline::IAllocationInfoProvider,
                              public framework::pipeline::IStreamExecutor {
public:
    /**
     * Constructor
     * @param module_id Unique identifier for this module
     * @param params Modulation mapping parameters
     */
    explicit ModulationMapper(
        const std::string& module_id,
        const ModulationParams& params
    );
    
    ~ModulationMapper() override;

    // IModule interface implementation
    [[nodiscard]] std::string_view get_type_id() const override { return "modulation_mapper"; }
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

    /**
     * @brief Calculate total number of symbols
     * @return Total symbol count
     */
    size_t calculate_total_symbols() const;
    
    /**
     * @brief Calculate total number of bits
     * @return Total bit count
     */
    size_t calculate_total_bits() const;
    
    /**
     * @brief Get bits per symbol for current modulation scheme
     * @return Bits per symbol
     */
    int get_bits_per_symbol() const;
    
    /**
     * @brief Set processing mode for modulation/demodulation
     * @param mode The processing mode to set
     */
    void set_processing_mode(ProcessingMode mode) { params_.mode = mode; }

private:
    std::string module_id_;
    ModulationParams params_;
    
    // Port information
    std::vector<framework::pipeline::PortInfo> input_ports_;
    std::vector<framework::pipeline::PortInfo> output_ports_;
    
    // GPU memory management
    ModulationDescriptor h_descriptor_;  ///< Host descriptor
    ModulationDescriptor* d_descriptor_{nullptr};  ///< Device descriptor
    
    // Device memory pointers
    uint8_t* d_input_bits_{nullptr};
    cuComplex* d_input_symbols_{nullptr};
    cuComplex* d_output_symbols_{nullptr};
    uint8_t* d_output_bits_{nullptr};
    float* d_soft_bits_{nullptr};
    float* d_evm_values_{nullptr};
    cuComplex* d_constellation_{nullptr};
    
    // Current input pointers from framework
    const void* current_bits_{nullptr};
    const void* current_symbols_{nullptr};
    
    /**
     * @brief Allocate GPU memory for modulation processing
     */
    void allocate_gpu_memory();
    
    /**
     * @brief Deallocate GPU memory
     */
    void deallocate_gpu_memory();
    
    /**
     * @brief Setup port information for input/output tensors
     */
    void setup_port_info();
    
    /**
     * @brief Launch CUDA kernel for modulation/demodulation
     * @param stream CUDA stream for execution
     * @return CUDA error code
     */
    cudaError_t launch_modulation_kernel(cudaStream_t stream);
    
    /**
     * @brief Initialize constellation points on GPU
     */
    void initialize_constellation();
    
    /**
     * @brief Get constellation size for current modulation scheme
     * @return Constellation size
     */
    int get_constellation_size() const;
};

} // namespace modulation_mapping

#endif // MODULATION_MAPPING_MODULE_HPP