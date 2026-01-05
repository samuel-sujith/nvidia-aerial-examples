#include "neural_beamforming_module.hpp"

#include <stdexcept>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <fstream>

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only print warnings and errors
        if (severity <= Severity::kWARNING) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};

static Logger gLogger;
#endif

namespace neural_beamforming {

/// CUDA kernel for conventional delay-and-sum beamforming
__global__ void conventional_beamforming_kernel(BeamformingDescriptor* desc) {
    int subcarrier_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int user_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int symbol_idx = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (subcarrier_idx >= desc->num_subcarriers || 
        user_idx >= desc->num_users || 
        symbol_idx >= desc->num_ofdm_symbols) return;
    
    // Calculate indices
    int weight_idx = user_idx * desc->num_antennas * desc->num_subcarriers + 
                    subcarrier_idx * desc->num_antennas;
    int output_idx = user_idx * desc->num_subcarriers * desc->num_ofdm_symbols +
                    symbol_idx * desc->num_subcarriers + subcarrier_idx;
    
    cuComplex output_symbol = make_cuComplex(0.0f, 0.0f);
    
    // Apply beamforming weights: y = w^H * x
    for (int ant = 0; ant < desc->num_antennas; ++ant) {
        int input_idx = ant * desc->num_subcarriers * desc->num_ofdm_symbols +
                       symbol_idx * desc->num_subcarriers + subcarrier_idx;
        
        cuComplex weight = desc->beamforming_weights[weight_idx + ant];
        cuComplex input_symbol = desc->input_symbols[input_idx];
        
        // Conjugate transpose: w^H
        weight = cuConjf(weight);
        output_symbol = cuCaddf(output_symbol, cuCmulf(weight, input_symbol));
    }
    
    desc->output_symbols[output_idx] = output_symbol;
}

/// CUDA kernel for MVDR beamforming weight computation
__global__ void mvdr_weight_computation_kernel(BeamformingDescriptor* desc) {
    int subcarrier_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int user_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (subcarrier_idx >= desc->num_subcarriers || user_idx >= desc->num_users) return;
    
    // MVDR: w = R^(-1) * h / (h^H * R^(-1) * h)
    // This is a simplified version - full implementation would use matrix inversion
    
    int weight_base_idx = user_idx * desc->num_antennas * desc->num_subcarriers + 
                         subcarrier_idx * desc->num_antennas;
    int channel_base_idx = user_idx * desc->num_antennas * desc->num_subcarriers + 
                          subcarrier_idx * desc->num_antennas;
    
    float normalization = 0.0f;
    
    // Calculate normalization factor
    for (int ant = 0; ant < desc->num_antennas; ++ant) {
        cuComplex h = desc->channel_matrix[channel_base_idx + ant];
        normalization += cuCrealf(cuCmulf(cuConjf(h), h));
    }
    
    normalization += desc->regularization_factor;
    normalization = 1.0f / sqrtf(normalization);
    
    // Compute MVDR weights (simplified)
    for (int ant = 0; ant < desc->num_antennas; ++ant) {
        cuComplex h = desc->channel_matrix[channel_base_idx + ant];
        desc->beamforming_weights[weight_base_idx + ant] = 
            make_cuComplex(cuCrealf(h) * normalization, cuCimagf(h) * normalization);
    }
}

/// CUDA kernel for Zero-Forcing beamforming
__global__ void zero_forcing_beamforming_kernel(BeamformingDescriptor* desc) {
    int subcarrier_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (subcarrier_idx >= desc->num_subcarriers) return;
    
    // Zero-Forcing: W = H^H * (H * H^H)^(-1)
    // Simplified implementation for demonstration
    
    for (int user_idx = 0; user_idx < desc->num_users; ++user_idx) {
        int weight_base_idx = user_idx * desc->num_antennas * desc->num_subcarriers + 
                             subcarrier_idx * desc->num_antennas;
        int channel_base_idx = user_idx * desc->num_antennas * desc->num_subcarriers + 
                              subcarrier_idx * desc->num_antennas;
        
        float normalization = 0.0f;
        for (int ant = 0; ant < desc->num_antennas; ++ant) {
            cuComplex h = desc->channel_matrix[channel_base_idx + ant];
            normalization += cuCrealf(cuCmulf(cuConjf(h), h));
        }
        
        normalization += desc->regularization_factor;
        normalization = 1.0f / sqrtf(normalization);
        
        for (int ant = 0; ant < desc->num_antennas; ++ant) {
            cuComplex h = desc->channel_matrix[channel_base_idx + ant];
            desc->beamforming_weights[weight_base_idx + ant] = 
                make_cuComplex(cuCrealf(cuConjf(h)) * normalization, 
                              cuCimagf(cuConjf(h)) * normalization);
        }
    }
}

/// CUDA kernel for SINR calculation
__global__ void calculate_sinr_kernel(BeamformingDescriptor* desc) {
    int subcarrier_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int user_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (subcarrier_idx >= desc->num_subcarriers || user_idx >= desc->num_users) return;
    
    int weight_base_idx = user_idx * desc->num_antennas * desc->num_subcarriers + 
                         subcarrier_idx * desc->num_antennas;
    int channel_base_idx = user_idx * desc->num_antennas * desc->num_subcarriers + 
                          subcarrier_idx * desc->num_antennas;
    
    // Calculate signal power: |w^H * h|^2
    cuComplex signal_term = make_cuComplex(0.0f, 0.0f);
    for (int ant = 0; ant < desc->num_antennas; ++ant) {
        cuComplex w = cuConjf(desc->beamforming_weights[weight_base_idx + ant]);
        cuComplex h = desc->channel_matrix[channel_base_idx + ant];
        signal_term = cuCaddf(signal_term, cuCmulf(w, h));
    }
    float signal_power = cuCrealf(cuCmulf(cuConjf(signal_term), signal_term));
    
    // Calculate interference + noise power
    float interference_power = 0.0f;
    for (int interferer = 0; interferer < desc->num_users; ++interferer) {
        if (interferer == user_idx) continue;
        
        int interferer_channel_base = interferer * desc->num_antennas * desc->num_subcarriers + 
                                     subcarrier_idx * desc->num_antennas;
        
        cuComplex interference_term = make_cuComplex(0.0f, 0.0f);
        for (int ant = 0; ant < desc->num_antennas; ++ant) {
            cuComplex w = cuConjf(desc->beamforming_weights[weight_base_idx + ant]);
            cuComplex h_interferer = desc->channel_matrix[interferer_channel_base + ant];
            interference_term = cuCaddf(interference_term, cuCmulf(w, h_interferer));
        }
        interference_power += cuCrealf(cuCmulf(cuConjf(interference_term), interference_term));
    }
    
    // Add noise power
    float noise_power = desc->noise_power;
    for (int ant = 0; ant < desc->num_antennas; ++ant) {
        cuComplex w = desc->beamforming_weights[weight_base_idx + ant];
        noise_power += cuCrealf(cuCmulf(cuConjf(w), w)) * desc->noise_power;
    }
    
    // Calculate SINR in dB
    float sinr_linear = signal_power / (interference_power + noise_power + 1e-10f);
    float sinr_db = 10.0f * log10f(sinr_linear + 1e-10f);
    
    // Store in performance metrics
    int metrics_idx = user_idx * desc->num_subcarriers + subcarrier_idx;
    desc->performance_metrics[metrics_idx] = sinr_db;
}

NeuralBeamformer::NeuralBeamformer(const std::string& module_id, const BeamformingParams& params)
    : module_id_(module_id), params_(params) {
    
    // Initialize descriptor
    h_descriptor_.num_antennas = params_.num_antennas;
    h_descriptor_.num_users = params_.num_users;
    h_descriptor_.num_subcarriers = params_.num_subcarriers;
    h_descriptor_.num_ofdm_symbols = params_.num_ofdm_symbols;
    h_descriptor_.algorithm = params_.algorithm;
    h_descriptor_.mode = params_.mode;
    h_descriptor_.regularization_factor = params_.regularization_factor;
    h_descriptor_.noise_power = params_.noise_power;
    
    // Calculate data sizes
    h_descriptor_.input_symbols_size = params_.num_antennas * params_.num_subcarriers * params_.num_ofdm_symbols;
    h_descriptor_.output_symbols_size = params_.num_users * params_.num_subcarriers * params_.num_ofdm_symbols;
    h_descriptor_.weights_size = params_.num_users * params_.num_antennas * params_.num_subcarriers;
    h_descriptor_.covariance_size = params_.num_antennas * params_.num_antennas * params_.num_subcarriers;
    
    // Setup port information
    setup_port_info();
    
    // Allocate GPU memory
    allocate_gpu_memory();
    
    // Initialize TensorRT engine if using neural network algorithm
    if (params_.algorithm == BeamformingAlgorithm::NEURAL_NETWORK && !params_.model_path.empty()) {
        initialize_tensorrt_engine();
    }
}

NeuralBeamformer::~NeuralBeamformer() {
    cleanup_tensorrt_resources();
    deallocate_gpu_memory();
}

std::vector<std::string> NeuralBeamformer::get_input_port_names() const {
    std::vector<std::string> names;
    for (const auto& port : input_ports_) {
        names.push_back(port.name);
    }
    return names;
}

std::vector<std::string> NeuralBeamformer::get_output_port_names() const {
    std::vector<std::string> names;
    for (const auto& port : output_ports_) {
        names.push_back(port.name);
    }
    return names;
}

framework::pipeline::ModuleMemoryRequirements NeuralBeamformer::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements requirements;
    
    // Calculate total memory needed
    size_t total_size = 0;
    
    // Input/output symbols
    total_size += h_descriptor_.input_symbols_size * sizeof(cuComplex);
    total_size += h_descriptor_.output_symbols_size * sizeof(cuComplex);
    
    // Beamforming weights and channel matrix
    total_size += h_descriptor_.weights_size * sizeof(cuComplex);
    total_size += h_descriptor_.weights_size * sizeof(cuComplex); // channel matrix
    
    // Covariance matrix
    total_size += h_descriptor_.covariance_size * sizeof(cuComplex);
    
    // Steering vectors (for DOA)
    total_size += h_descriptor_.num_antennas * 360 * sizeof(cuComplex); // 360 degrees
    
    // Performance metrics
    total_size += h_descriptor_.num_users * h_descriptor_.num_subcarriers * sizeof(float);
    
    // Descriptor
    total_size += sizeof(BeamformingDescriptor);
    
    requirements.device_tensor_bytes = total_size;
    requirements.alignment = 256; // CUDA memory alignment
    
    return requirements;
}

framework::pipeline::OutputPortMemoryCharacteristics 
NeuralBeamformer::get_output_memory_characteristics(std::string_view port_name) const {
    framework::pipeline::OutputPortMemoryCharacteristics characteristics{};
    characteristics.provides_fixed_address_for_zero_copy = true;
    return characteristics;
}

void NeuralBeamformer::setup_port_info() {
    using namespace framework::tensor;
    
    // Setup input ports
    input_ports_.resize(2);
    
    // Input port 0: input_symbols [num_antennas x num_subcarriers x num_ofdm_symbols]
    input_ports_[0].name = "input_symbols";
    input_ports_[0].tensors.resize(1);
    input_ports_[0].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(h_descriptor_.num_antennas),
            static_cast<std::size_t>(h_descriptor_.num_subcarriers),
            static_cast<std::size_t>(h_descriptor_.num_ofdm_symbols)
        }
    );
    
    // Input port 1: channel_estimates [num_users x num_antennas x num_subcarriers]  
    input_ports_[1].name = "channel_estimates";
    input_ports_[1].tensors.resize(1);
    input_ports_[1].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(h_descriptor_.num_users),
            static_cast<std::size_t>(h_descriptor_.num_antennas),
            static_cast<std::size_t>(h_descriptor_.num_subcarriers)
        }
    );
    
    // Setup output ports
    output_ports_.resize(3);
    
    // Output port 0: output_symbols [num_users x num_subcarriers x num_ofdm_symbols]
    output_ports_[0].name = "output_symbols";
    output_ports_[0].tensors.resize(1);
    output_ports_[0].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(h_descriptor_.num_users),
            static_cast<std::size_t>(h_descriptor_.num_subcarriers),
            static_cast<std::size_t>(h_descriptor_.num_ofdm_symbols)
        }
    );
    
    // Output port 1: beamforming_weights [num_users x num_antennas x num_subcarriers]
    output_ports_[1].name = "beamforming_weights";
    output_ports_[1].tensors.resize(1);
    output_ports_[1].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(h_descriptor_.num_users),
            static_cast<std::size_t>(h_descriptor_.num_antennas),
            static_cast<std::size_t>(h_descriptor_.num_subcarriers)
        }
    );
    
    // Output port 2: performance_metrics [num_users x num_subcarriers]
    output_ports_[2].name = "performance_metrics";
    output_ports_[2].tensors.resize(1);
    output_ports_[2].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorR32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(h_descriptor_.num_users),
            static_cast<std::size_t>(h_descriptor_.num_subcarriers)
        }
    );
}

void NeuralBeamformer::allocate_gpu_memory() {
    // Allocate device memory for all data structures
    cudaError_t err;
    
    // Input/output symbols
    err = cudaMalloc(&d_input_symbols_, h_descriptor_.input_symbols_size * sizeof(cuComplex));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate input symbols memory");
    
    err = cudaMalloc(&d_output_symbols_, h_descriptor_.output_symbols_size * sizeof(cuComplex));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate output symbols memory");
    
    // Beamforming weights and channel matrix
    err = cudaMalloc(&d_beamforming_weights_, h_descriptor_.weights_size * sizeof(cuComplex));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate beamforming weights memory");
    
    err = cudaMalloc(&d_channel_matrix_, h_descriptor_.weights_size * sizeof(cuComplex));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate channel matrix memory");
    
    // Covariance matrix
    err = cudaMalloc(&d_covariance_matrix_, h_descriptor_.covariance_size * sizeof(cuComplex));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate covariance matrix memory");
    
    // Steering vectors (for DOA algorithms)
    err = cudaMalloc(&d_steering_vectors_, h_descriptor_.num_antennas * 360 * sizeof(cuComplex));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate steering vectors memory");
    
    // Performance metrics
    err = cudaMalloc(&d_performance_metrics_, h_descriptor_.num_users * h_descriptor_.num_subcarriers * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate performance metrics memory");
    
    // Descriptor
    err = cudaMalloc(&d_descriptor_, sizeof(BeamformingDescriptor));
    if (err != cudaSuccess) throw std::runtime_error("Failed to allocate descriptor memory");
    
    // Update descriptor with device pointers
    h_descriptor_.input_symbols = d_input_symbols_;
    h_descriptor_.output_symbols = d_output_symbols_;
    h_descriptor_.beamforming_weights = d_beamforming_weights_;
    h_descriptor_.channel_matrix = d_channel_matrix_;
    h_descriptor_.covariance_matrix = d_covariance_matrix_;
    h_descriptor_.steering_vectors = d_steering_vectors_;
    h_descriptor_.performance_metrics = d_performance_metrics_;
    
    // Copy descriptor to GPU
    err = cudaMemcpy(d_descriptor_, &h_descriptor_, sizeof(BeamformingDescriptor), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("Failed to copy beamforming descriptor to GPU");
}

void NeuralBeamformer::deallocate_gpu_memory() {
    cudaFree(d_input_symbols_);
    cudaFree(d_output_symbols_);
    cudaFree(d_beamforming_weights_);
    cudaFree(d_channel_matrix_);
    cudaFree(d_covariance_matrix_);
    cudaFree(d_steering_vectors_);
    cudaFree(d_performance_metrics_);
    cudaFree(d_descriptor_);
    
    d_input_symbols_ = nullptr;
    d_output_symbols_ = nullptr;
    d_beamforming_weights_ = nullptr;
    d_channel_matrix_ = nullptr;
    d_covariance_matrix_ = nullptr;
    d_steering_vectors_ = nullptr;
    d_performance_metrics_ = nullptr;
    d_descriptor_ = nullptr;
}

void NeuralBeamformer::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    // Extract device pointers from input ports
    for (const auto& port : inputs) {
        if (port.name == "input_symbols" && !port.tensors.empty()) {
            current_input_symbols_ = port.tensors[0].device_ptr;
        } else if (port.name == "channel_estimates" && !port.tensors.empty()) {
            current_channel_estimates_ = port.tensors[0].device_ptr;
        }
    }
}

std::vector<framework::pipeline::PortInfo> NeuralBeamformer::get_outputs() const {
    // Return a copy of output_ports_ with updated device pointers
    auto outputs = output_ports_;
    
    if (!outputs.empty() && !outputs[0].tensors.empty()) {
        outputs[0].tensors[0].device_ptr = d_output_symbols_;
    }
    if (outputs.size() > 1 && !outputs[1].tensors.empty()) {
        outputs[1].tensors[0].device_ptr = d_beamforming_weights_;
    }
    if (outputs.size() > 2 && !outputs[2].tensors.empty()) {
        outputs[2].tensors[0].device_ptr = d_performance_metrics_;
    }
    
    return outputs;
}

void NeuralBeamformer::execute(cudaStream_t stream) {
    if (!current_input_symbols_ || !current_channel_estimates_) {
        throw std::runtime_error("Input symbols or channel estimates not set for beamforming");
    }
    
    // Copy input data to internal buffers
    cudaError_t err;
    err = cudaMemcpyAsync(d_input_symbols_, current_input_symbols_, 
                         h_descriptor_.input_symbols_size * sizeof(cuComplex), 
                         cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy input symbols");
    }
    
    err = cudaMemcpyAsync(d_channel_matrix_, current_channel_estimates_,
                         h_descriptor_.weights_size * sizeof(cuComplex),
                         cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy channel estimates");
    }
    
    // Launch appropriate beamforming algorithm
    bool success = false;
    switch (params_.algorithm) {
        case BeamformingAlgorithm::CONVENTIONAL:
            success = compute_conventional_weights(stream);
            break;
        case BeamformingAlgorithm::MVDR:
            success = compute_mvdr_weights(stream);
            break;
        case BeamformingAlgorithm::ZERO_FORCING:
            success = compute_zero_forcing_weights(stream);
            break;
        case BeamformingAlgorithm::NEURAL_NETWORK:
            success = compute_neural_weights(stream);
            break;
    }
    
    if (!success) {
        throw std::runtime_error("Beamforming weight computation failed");
    }
    
    // Apply beamforming weights to input symbols
    success = apply_beamforming_weights(stream);
    if (!success) {
        throw std::runtime_error("Failed to apply beamforming weights");
    }
    
    // Calculate performance metrics
    success = calculate_performance_metrics(stream);
    if (!success) {
        throw std::runtime_error("Failed to calculate performance metrics");
    }
}

bool NeuralBeamformer::compute_conventional_weights(cudaStream_t stream) {
    // For conventional beamforming, weights are just normalized channel responses
    dim3 blockSize(16, 4, 1);
    dim3 gridSize(
        (h_descriptor_.num_subcarriers + blockSize.x - 1) / blockSize.x,
        (h_descriptor_.num_users + blockSize.y - 1) / blockSize.y,
        1
    );
    
    // Simple weight computation: w = h / ||h||
    mvdr_weight_computation_kernel<<<gridSize, blockSize, 0, stream>>>(d_descriptor_);
    return cudaGetLastError() == cudaSuccess;
}

bool NeuralBeamformer::compute_mvdr_weights(cudaStream_t stream) {
    dim3 blockSize(16, 4);
    dim3 gridSize(
        (h_descriptor_.num_subcarriers + blockSize.x - 1) / blockSize.x,
        (h_descriptor_.num_users + blockSize.y - 1) / blockSize.y
    );
    
    mvdr_weight_computation_kernel<<<gridSize, blockSize, 0, stream>>>(d_descriptor_);
    return cudaGetLastError() == cudaSuccess;
}

bool NeuralBeamformer::compute_zero_forcing_weights(cudaStream_t stream) {
    dim3 blockSize(32, 1);
    dim3 gridSize((h_descriptor_.num_subcarriers + blockSize.x - 1) / blockSize.x, 1);
    
    zero_forcing_beamforming_kernel<<<gridSize, blockSize, 0, stream>>>(d_descriptor_);
    return cudaGetLastError() == cudaSuccess;
}

bool NeuralBeamformer::compute_neural_weights(cudaStream_t stream) {
    if (!trt_context_ || !trt_engine_) {
        fprintf(stderr, "TensorRT engine not initialized for neural beamforming - falling back to MVDR\n");
        return compute_mvdr_weights(stream);
    }
    
    // Run neural network inference to get beamforming weights
    if (!run_neural_inference(stream)) {
        fprintf(stderr, "Neural inference failed - falling back to MVDR\n");
        return compute_mvdr_weights(stream);
    }
    
    return true;
}

bool NeuralBeamformer::apply_beamforming_weights(cudaStream_t stream) {
    dim3 blockSize(8, 4, 4);
    dim3 gridSize(
        (h_descriptor_.num_subcarriers + blockSize.x - 1) / blockSize.x,
        (h_descriptor_.num_users + blockSize.y - 1) / blockSize.y,
        (h_descriptor_.num_ofdm_symbols + blockSize.z - 1) / blockSize.z
    );
    
    conventional_beamforming_kernel<<<gridSize, blockSize, 0, stream>>>(d_descriptor_);
    return cudaGetLastError() == cudaSuccess;
}

bool NeuralBeamformer::calculate_performance_metrics(cudaStream_t stream) {
    dim3 blockSize(16, 4);
    dim3 gridSize(
        (h_descriptor_.num_subcarriers + blockSize.x - 1) / blockSize.x,
        (h_descriptor_.num_users + blockSize.y - 1) / blockSize.y
    );
    
    calculate_sinr_kernel<<<gridSize, blockSize, 0, stream>>>(d_descriptor_);
    return cudaGetLastError() == cudaSuccess;
}

bool NeuralBeamformer::initialize_tensorrt_engine() {
#ifdef TENSORRT_AVAILABLE
    if (params_.model_path.empty()) {
        printf("Warning: No model path specified for neural beamforming\n");
        return false;
    }
    
    try {
        // Create TensorRT runtime
        trt_runtime_ = nvinfer1::createInferRuntime(gLogger);
        if (!trt_runtime_) {
            printf("Failed to create TensorRT runtime\n");
            return false;
        }
        
        // Load engine from file
        if (!load_engine_from_file(params_.model_path)) {
            printf("Failed to load TensorRT engine from %s\n", params_.model_path.c_str());
            return false;
        }
        
        // Create execution context
        trt_context_ = trt_engine_->createExecutionContext();
        if (!trt_context_) {
            printf("Failed to create TensorRT execution context\n");
            return false;
        }
        
        // Allocate device memory for ML input/output
        size_t input_size = params_.input_size * sizeof(float);
        size_t output_size = params_.output_size * sizeof(float);
        
        cudaError_t err = cudaMalloc(&d_ml_input_, input_size);
        if (err != cudaSuccess) {
            printf("Failed to allocate ML input memory: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        err = cudaMalloc(&d_ml_output_, output_size);
        if (err != cudaSuccess) {
            printf("Failed to allocate ML output memory: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        printf("TensorRT engine initialized successfully for neural beamforming\n");
        return true;
        
    } catch (const std::exception& e) {
        printf("Exception during TensorRT initialization: %s\n", e.what());
        return false;
    }
#else
    printf("TensorRT not available - neural beamforming will fall back to MVDR\n");
    return false;
#endif
}

void NeuralBeamformer::cleanup_tensorrt_resources() {
#ifdef TENSORRT_AVAILABLE
    if (trt_context_) {
        delete trt_context_;
        trt_context_ = nullptr;
    }
    
    if (trt_engine_) {
        delete trt_engine_;
        trt_engine_ = nullptr;
    }
    
    if (trt_runtime_) {
        delete trt_runtime_;
        trt_runtime_ = nullptr;
    }
#else
    // For stub version, just reset pointers
    trt_context_ = nullptr;
    trt_engine_ = nullptr;
    trt_runtime_ = nullptr;
#endif
    
    if (d_ml_input_) {
        cudaFree(d_ml_input_);
        d_ml_input_ = nullptr;
    }
    
    if (d_ml_output_) {
        cudaFree(d_ml_output_);
        d_ml_output_ = nullptr;
    }
}

size_t NeuralBeamformer::calculate_input_symbols() const {
    return static_cast<size_t>(h_descriptor_.input_symbols_size);
}

size_t NeuralBeamformer::calculate_output_symbols() const {
    return static_cast<size_t>(h_descriptor_.output_symbols_size);
}

cudaError_t NeuralBeamformer::launch_beamforming_kernel(cudaStream_t stream) {
    // This method coordinates the full beamforming pipeline
    execute(stream);
    return cudaGetLastError();
}

// IModule interface implementations
std::vector<framework::tensor::TensorInfo>
NeuralBeamformer::get_input_tensor_info(std::string_view port_name) const {
    std::vector<framework::tensor::TensorInfo> tensor_infos;
    
    if (port_name == "input_symbols") {
        tensor_infos.emplace_back(
            framework::tensor::NvDataType::TensorC32F,
            std::vector<std::size_t>{static_cast<std::size_t>(h_descriptor_.num_antennas),
                                     static_cast<std::size_t>(h_descriptor_.num_subcarriers),
                                     static_cast<std::size_t>(h_descriptor_.num_ofdm_symbols)}
        );
    } else if (port_name == "channel_estimates") {
        tensor_infos.emplace_back(
            framework::tensor::NvDataType::TensorC32F,
            std::vector<std::size_t>{static_cast<std::size_t>(h_descriptor_.num_users),
                                     static_cast<std::size_t>(h_descriptor_.num_antennas),
                                     static_cast<std::size_t>(h_descriptor_.num_subcarriers)}
        );
    }
    
    return tensor_infos;
}

std::vector<framework::tensor::TensorInfo>
NeuralBeamformer::get_output_tensor_info(std::string_view port_name) const {
    std::vector<framework::tensor::TensorInfo> tensor_infos;
    
    if (port_name == "output_symbols") {
        tensor_infos.emplace_back(
            framework::tensor::NvDataType::TensorC32F,
            std::vector<std::size_t>{static_cast<std::size_t>(h_descriptor_.num_users),
                                     static_cast<std::size_t>(h_descriptor_.num_subcarriers),
                                     static_cast<std::size_t>(h_descriptor_.num_ofdm_symbols)}
        );
    } else if (port_name == "beamforming_weights") {
        tensor_infos.emplace_back(
            framework::tensor::NvDataType::TensorC32F,
            std::vector<std::size_t>{static_cast<std::size_t>(h_descriptor_.num_users),
                                     static_cast<std::size_t>(h_descriptor_.num_antennas),
                                     static_cast<std::size_t>(h_descriptor_.num_subcarriers)}
        );
    } else if (port_name == "performance_metrics") {
        tensor_infos.emplace_back(
            framework::tensor::NvDataType::TensorR32F,
            std::vector<std::size_t>{static_cast<std::size_t>(h_descriptor_.num_users),
                                     static_cast<std::size_t>(h_descriptor_.num_subcarriers)}
        );
    }
    
    return tensor_infos;
}

void NeuralBeamformer::setup_memory(const framework::pipeline::ModuleMemorySlice& /*memory_slice*/) {
    // For now, continue using our own memory allocation
    // In a full implementation, this would use the provided memory slice
    allocate_gpu_memory();
}

void NeuralBeamformer::warmup(cudaStream_t stream) {
    // Perform any initialization kernels needed for warmup
    // This ensures all GPU resources are ready for processing
    if (!d_descriptor_) {
        return;
    }
    
    // Launch a dummy kernel to warm up the GPU
    dim3 blockSize(256);
    dim3 gridSize(1);
    
    // Just sync to ensure the stream is ready
    cudaStreamSynchronize(stream);
}

void NeuralBeamformer::configure_io(
    const framework::pipeline::DynamicParams& params,
    cudaStream_t stream
) {
    // Update any dynamic parameters if needed
    // For now, parameters are set during construction
    // In a full implementation, this would update descriptor based on dynamic params
    
    // Ensure descriptor is ready for execution
    if (d_descriptor_) {
        cudaMemcpyAsync(d_descriptor_, &h_descriptor_, sizeof(BeamformingDescriptor),
                       cudaMemcpyHostToDevice, stream);
    }
}

bool NeuralBeamformer::load_engine_from_file(const std::string& engine_path) {
#ifdef TENSORRT_AVAILABLE
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        printf("Failed to open TensorRT engine file: %s\n", engine_path.c_str());
        return false;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read engine data
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    
    // Deserialize engine
    trt_engine_ = trt_runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!trt_engine_) {
        printf("Failed to deserialize TensorRT engine\n");
        return false;
    }
    
    printf("Successfully loaded TensorRT engine from %s\n", engine_path.c_str());
    return true;
#else
    printf("TensorRT not available - cannot load engine file\n");
    return false;
#endif
}

bool NeuralBeamformer::run_neural_inference(cudaStream_t stream) {
    #ifdef TENSORRT_AVAILABLE
    if (!trt_context_ || !trt_engine_) {
        printf("TensorRT engine or context not initialized\n");
        return false;
    }

    // Prepare input data: convert complex channel matrix to real-valued input
    // Input format: [real(H), imag(H)] where H is channel matrix
    int total_elements = h_descriptor_.num_users * h_descriptor_.num_antennas * h_descriptor_.num_subcarriers;
    int input_floats = total_elements * 2; // real, imag interleaved

    // Launch kernel to convert cuComplex* to float* (real, imag interleaved)
    dim3 blockSize(256);
    dim3 gridSize((total_elements + blockSize.x - 1) / blockSize.x);
    neural_beamforming::cucomplex_to_float_interleaved<<<gridSize, blockSize, 0, stream>>>(d_channel_matrix_, d_ml_input_, total_elements);

    // Set up TensorRT explicit I/O bindings
    trt_context_->setTensorAddress("input", d_ml_input_);
    trt_context_->setTensorAddress("output", d_ml_output_);

    // Run inference
    bool success = trt_context_->enqueueV3(stream);
    if (!success) {
        printf("TensorRT inference failed\n");
        return false;
    }

    // Convert neural network output back to beamforming weights
    neural_beamforming::float_interleaved_to_cucomplex<<<gridSize, blockSize, 0, stream>>>(d_ml_output_, d_beamforming_weights_, total_elements);

    return true;
    #else
    printf("TensorRT not available - using fallback\n");
    return false;
    #endif
}

} // namespace neural_beamforming

namespace neural_beamforming {

// Kernel: Convert float* (real, imag interleaved) to cuComplex*
__global__ void float_interleaved_to_cucomplex(const float* in, cuComplex* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = make_cuComplex(in[2 * idx], in[2 * idx + 1]);
    }
}

// Kernel: Convert cuComplex* to float* (real, imag interleaved)
__global__ void cucomplex_to_float_interleaved(const cuComplex* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[2 * idx]     = cuCrealf(in[idx]);
        out[2 * idx + 1] = cuCimagf(in[idx]);
    }
}

} // namespace neural_beamforming