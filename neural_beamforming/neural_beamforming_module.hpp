#ifndef NEURAL_BEAMFORMING_MODULE_HPP
#define NEURAL_BEAMFORMING_MODULE_HPP

#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"
#include "tensor/data_types.hpp"

#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <span>

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <fstream>
#include <iostream>
#endif

namespace neural_beamforming {

/**
 * @brief Beamforming algorithm types
 */
enum class BeamformingAlgorithm {
    CONVENTIONAL,     ///< Conventional delay-and-sum beamforming
    MVDR,            ///< Minimum Variance Distortionless Response
    ZERO_FORCING,    ///< Zero Forcing beamforming
    NEURAL_NETWORK   ///< Neural network based beamforming
};

/**
 * @brief Processing mode for beamforming operations
 */
enum class ProcessingMode {
    TRANSMIT,        ///< Transmit beamforming
    RECEIVE,         ///< Receive beamforming
    BOTH            ///< Both transmit and receive
};

/**
 * @brief Configuration parameters for neural beamforming
 */
struct BeamformingParams {
    BeamformingAlgorithm algorithm = BeamformingAlgorithm::MVDR;
    ProcessingMode mode = ProcessingMode::RECEIVE;
    
    // Array configuration
    int num_antennas = 64;           ///< Number of antenna elements
    int num_users = 4;               ///< Number of users/beams
    int num_subcarriers = 1200;      ///< Number of subcarriers
    int num_ofdm_symbols = 14;       ///< Number of OFDM symbols
    
    // Algorithm parameters
    float regularization_factor = 1e-6f;  ///< Regularization for matrix inversion
    float noise_power = 0.1f;             ///< Noise power estimate
    bool enable_calibration = true;       ///< Enable array calibration
    
    // Neural network parameters (for ML-based beamforming)
    std::string model_path;          ///< Path to TensorRT model file (.trt or .onnx)
    int max_batch_size = 32;         ///< Maximum batch size for inference
    bool use_fp16 = true;           ///< Use FP16 precision for inference
    int input_size = 256;           ///< Neural network input size (antennas * subcarriers)
    int output_size = 256;          ///< Neural network output size (beamforming weights)
};

/**
 * @brief Descriptor for beamforming operations on GPU
 */
struct BeamformingDescriptor {
    // Array configuration
    int num_antennas;
    int num_users;
    int num_subcarriers;
    int num_ofdm_symbols;
    
    // Algorithm parameters
    BeamformingAlgorithm algorithm;
    ProcessingMode mode;
    float regularization_factor;
    float noise_power;
    
    // Data dimensions
    int input_symbols_size;
    int output_symbols_size;
    int weights_size;
    int covariance_size;
    
    // Device memory pointers
    cuComplex* input_symbols;      ///< Input symbols [num_antennas x num_subcarriers x num_symbols]
    cuComplex* output_symbols;     ///< Output symbols [num_users x num_subcarriers x num_symbols]
    cuComplex* beamforming_weights; ///< Beamforming weights [num_users x num_antennas x num_subcarriers]
    cuComplex* channel_matrix;     ///< Channel matrix [num_users x num_antennas x num_subcarriers]
    cuComplex* covariance_matrix;  ///< Covariance matrix [num_antennas x num_antennas x num_subcarriers]
    cuComplex* steering_vectors;   ///< Steering vectors for DOA
    float* performance_metrics;    ///< Performance metrics (SINR, throughput, etc.)
};

/**
 * @brief Neural beamforming module implementing classical and ML-based algorithms
 *
 * This module provides comprehensive beamforming capabilities for massive MIMO systems,
 * supporting both traditional algorithms and neural network-based approaches.
 */
class NeuralBeamformer : public framework::pipeline::IModule,
                        public framework::pipeline::IAllocationInfoProvider,
                        public framework::pipeline::IStreamExecutor {
public:
    /**
     * @brief Constructor
     * @param module_id Unique identifier for the module
     * @param params Beamforming parameters
     */
    explicit NeuralBeamformer(const std::string& module_id, const BeamformingParams& params);
    
    /**
     * @brief Destructor
     */
    ~NeuralBeamformer() override;
    
    // IModule interface
    [[nodiscard]] std::string_view get_type_id() const override { return "neural_beamformer"; }
    [[nodiscard]] std::string_view get_instance_id() const override { return module_id_; }
    
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;

    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;
    
    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;
    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;
    
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
     * @brief Get the module identifier
     * @return Module ID string
     */
    [[nodiscard]] const std::string& get_module_id() const { return module_id_; }
    
    /**
     * @brief Calculate total input symbols
     * @return Total input symbol count
     */
    size_t calculate_input_symbols() const;
    
    /**
     * @brief Calculate total output symbols  
     * @return Total output symbol count
     */
    size_t calculate_output_symbols() const;
    
    /**
     * @brief Get current beamforming algorithm
     * @return Current algorithm
     */
    BeamformingAlgorithm get_algorithm() const { return params_.algorithm; }

private:
    std::string module_id_;
    BeamformingParams params_;
    framework::pipeline::ModuleMemorySlice mem_slice_;
    
    // Framework interface
    std::vector<framework::pipeline::PortInfo> input_ports_;
    std::vector<framework::pipeline::PortInfo> output_ports_;
    
    // GPU resources
    BeamformingDescriptor* d_descriptor_;
    BeamformingDescriptor h_descriptor_;
    
    // Device memory pointers
    cuComplex* d_input_symbols_{nullptr};
    cuComplex* d_output_symbols_{nullptr};
    cuComplex* d_beamforming_weights_{nullptr};
    cuComplex* d_channel_matrix_{nullptr};
    cuComplex* d_covariance_matrix_{nullptr};
    cuComplex* d_steering_vectors_{nullptr};
    float* d_performance_metrics_{nullptr};
    
    // Neural network resources (for ML-based beamforming)
#ifdef TENSORRT_AVAILABLE
    nvinfer1::IRuntime* trt_runtime_{nullptr};           ///< TensorRT runtime
    nvinfer1::ICudaEngine* trt_engine_{nullptr};         ///< TensorRT inference engine
    nvinfer1::IExecutionContext* trt_context_{nullptr};  ///< TensorRT execution context
#else
    void* trt_runtime_{nullptr};
    void* trt_engine_{nullptr};         
    void* trt_context_{nullptr};        
#endif
    float* d_ml_input_{nullptr};         ///< Device memory for ML input
    float* d_ml_output_{nullptr};        ///< Device memory for ML output
    
    // Current input pointers from framework
    const void* current_input_symbols_{nullptr};
    const void* current_channel_estimates_{nullptr};
    
    /**
     * @brief Allocate GPU memory for beamforming processing
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
     * @brief Launch CUDA kernel for beamforming computation
     * @param stream CUDA stream for execution
     * @return CUDA error code
     */
    cudaError_t launch_beamforming_kernel(cudaStream_t stream);
    
    /**
     * @brief Initialize TensorRT engine for neural network beamforming
     * @return True if initialization successful
     */
    bool initialize_tensorrt_engine();
    
    /**
     * @brief Cleanup TensorRT resources
     */
    void cleanup_tensorrt_resources();
    
    /**
     * @brief Load TensorRT engine from file
     * @param engine_path Path to .trt or .onnx file
     * @return Success status
     */
    bool load_engine_from_file(const std::string& engine_path);
    
    /**
     * @brief Run neural network inference
     * @param stream CUDA stream
     * @return Success status
     */
    bool run_neural_inference(cudaStream_t stream);
    
    /**
     * @brief Compute conventional beamforming weights
     * @param stream CUDA stream
     * @return Success status
     */
    bool compute_conventional_weights(cudaStream_t stream);
    
    /**
     * @brief Compute MVDR beamforming weights
     * @param stream CUDA stream
     * @return Success status
     */
    bool compute_mvdr_weights(cudaStream_t stream);
    
    /**
     * @brief Compute zero-forcing beamforming weights
     * @param stream CUDA stream
     * @return Success status
     */
    bool compute_zero_forcing_weights(cudaStream_t stream);
    
    /**
     * @brief Compute neural network beamforming weights
     * @param stream CUDA stream
     * @return Success status
     */
    bool compute_neural_weights(cudaStream_t stream);
    
    /**
     * @brief Apply beamforming weights to input symbols
     * @param stream CUDA stream
     * @return Success status
     */
    bool apply_beamforming_weights(cudaStream_t stream);
    
    /**
     * @brief Calculate performance metrics
     * @param stream CUDA stream
     * @return Success status
     */
    bool calculate_performance_metrics(cudaStream_t stream);
};

} // namespace neural_beamforming

#endif // NEURAL_BEAMFORMING_MODULE_HPP