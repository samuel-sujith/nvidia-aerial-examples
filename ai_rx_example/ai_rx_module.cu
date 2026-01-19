#include "ai_rx_module.hpp"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cuComplex.h>

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#include <NvInferRuntime.h>

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};

static Logger gLogger;
#endif

namespace ai_rx_example {

bool AiRxModule::load_engine_from_file(const std::string& engine_path) {
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

void AiRxModule::cleanup_tensorrt_resources() {
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

    if (d_trt_input_) {
        cudaFree(d_trt_input_);
        d_trt_input_ = nullptr;
    }

    if (d_trt_output_) {
        cudaFree(d_trt_output_);
        d_trt_output_ = nullptr;
    }
}

bool AiRxModule::initialize_tensorrt_engine() {
#ifdef TENSORRT_AVAILABLE
    if (params_.model_path.empty()) {
        printf("Warning: No model path specified for TensorRT initialization\n");
        return false;
    }

    try {
        printf("Creating TensorRT runtime\n");
        trt_runtime_ = nvinfer1::createInferRuntime(gLogger);
        if (!trt_runtime_) {
            fprintf(stderr, "Failed to create TensorRT runtime\n");
            return false;
        }

        printf("Loading TensorRT engine from file: %s\n", params_.model_path.c_str());
        if (!load_engine_from_file(params_.model_path)) {
            fprintf(stderr, "Failed to load TensorRT engine from %s\n", params_.model_path.c_str());
            return false;
        }

        printf("Creating TensorRT execution context\n");
        trt_context_ = trt_engine_->createExecutionContext();
        if (!trt_context_) {
            fprintf(stderr, "Failed to create TensorRT execution context\n");
            return false;
        }

        printf("Allocating device memory for TensorRT input/output\n");
        size_t input_size = params_.num_symbols * sizeof(float);
        size_t output_size = params_.num_symbols * sizeof(float);

        cudaError_t err = cudaMalloc(&d_trt_input_, input_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate TensorRT input memory: %s\n", cudaGetErrorString(err));
            return false;
        }

        err = cudaMalloc(&d_trt_output_, output_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate TensorRT output memory: %s\n", cudaGetErrorString(err));
            return false;
        }

        printf("TensorRT engine initialized successfully\n");
        return true;

    } catch (const std::exception& e) {
        fprintf(stderr, "Exception during TensorRT initialization: %s\n", e.what());
        return false;
    }
#else
    printf("TensorRT not available - falling back to default implementation\n");
    return false;
#endif
}

AiRxModule::AiRxModule(const std::string& module_id, const AiRxParams& params)
    : module_id_(module_id), params_(params) {
    // Initialize port information
    setup_port_info();

    // Allocate GPU memory
    allocate_gpu_memory();

    // Initialize TensorRT engine if model path is provided
    if (!params_.model_path.empty()) {
        initialize_tensorrt_engine();
    }
}

AiRxModule::~AiRxModule() {
    // Clean up TensorRT resources
    cleanup_tensorrt_resources();

    // Deallocate GPU memory
    deallocate_gpu_memory();
}

void AiRxModule::setup_port_info() {
    printf("Setting up port information\n");
    // Setup input and output port information
    input_ports_.resize(1);
    input_ports_[0].name = "input";
    input_ports_[0].tensors.resize(1);
    input_ports_[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorR32F,
        {static_cast<size_t>(params_.num_symbols)}
    );

    output_ports_.resize(1);
    output_ports_[0].name = "output";
    output_ports_[0].tensors.resize(1);
    output_ports_[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorR32F,
        {static_cast<size_t>(params_.num_symbols)}
    );
    printf("Port information setup complete\n");
}

void AiRxModule::allocate_gpu_memory() {
    printf("Allocating GPU memory\n");
    cudaError_t err;

    // Allocate memory for input symbols
    err = cudaMalloc(&d_rx_symbols_, params_.num_symbols * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory for input symbols: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("Failed to allocate GPU memory for input symbols");
    }

    // Allocate memory for output symbols
    err = cudaMalloc(&d_rx_bits_, params_.num_symbols * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory for output symbols: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("Failed to allocate GPU memory for output symbols");
    }
    printf("GPU memory allocation complete\n");
}

void AiRxModule::deallocate_gpu_memory() {
    printf("Deallocating GPU memory\n");
    if (d_rx_symbols_) {
        cudaFree(d_rx_symbols_);
        d_rx_symbols_ = nullptr;
    }

    if (d_rx_bits_) {
        cudaFree(d_rx_bits_);
        d_rx_bits_ = nullptr;
    }
    printf("GPU memory deallocation complete\n");
}

void AiRxModule::setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) {
    // Implementation for setting up memory using the provided memory slice
    // For now, this is a placeholder
}

std::vector<framework::tensor::TensorInfo> AiRxModule::get_input_tensor_info(std::string_view port_name) const {
    if (port_name == "input") {
        return {input_ports_[0].tensors[0].tensor_info};
    }
    return {};
}

std::vector<framework::tensor::TensorInfo> AiRxModule::get_output_tensor_info(std::string_view port_name) const {
    if (port_name == "output") {
        return {output_ports_[0].tensors[0].tensor_info};
    }
    return {};
}

std::vector<std::string> AiRxModule::get_input_port_names() const {
    return {"input"};
}

std::vector<std::string> AiRxModule::get_output_port_names() const {
    return {"output"};
}

void AiRxModule::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    if (!inputs.empty()) {
        current_rx_symbols_ = inputs[0].tensors[0].device_ptr;
    }
}

std::vector<framework::pipeline::PortInfo> AiRxModule::get_outputs() const {
    auto outputs = output_ports_;
    if (!outputs.empty() && !outputs[0].tensors.empty()) {
        outputs[0].tensors[0].device_ptr = d_rx_bits_;
    }
    return outputs;
}

void AiRxModule::warmup(cudaStream_t stream) {
    // Perform any warmup operations if needed
    cudaMemsetAsync(d_rx_symbols_, 0, params_.num_symbols * sizeof(float), stream);
    cudaMemsetAsync(d_rx_bits_, 0, params_.num_symbols * sizeof(float), stream);
}

void AiRxModule::configure_io(const framework::pipeline::DynamicParams& params, cudaStream_t stream) {
    // Configure IO dynamically if needed
    // Placeholder implementation
}

framework::pipeline::ModuleMemoryRequirements AiRxModule::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements requirements;
    requirements.device_tensor_bytes = params_.num_symbols * sizeof(float) * 2; // Input and output
    requirements.alignment = 256; // CUDA memory alignment
    return requirements;
}

framework::pipeline::OutputPortMemoryCharacteristics AiRxModule::get_output_memory_characteristics(std::string_view port_name) const {
    framework::pipeline::OutputPortMemoryCharacteristics characteristics;
    characteristics.provides_fixed_address_for_zero_copy = true;
    return characteristics;
}

void AiRxModule::execute(cudaStream_t stream) {
    if (!current_rx_symbols_) {
        fprintf(stderr, "Input symbols not set for execution\n");
        throw std::runtime_error("Input symbols not set for execution");
    }

    printf("Executing AI Rx Module\n");

    // Perform computation (placeholder)
    cudaError_t err = cudaMemcpyAsync(d_rx_bits_, current_rx_symbols_, params_.num_symbols * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy input symbols to output buffer: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("Failed to copy input symbols to output buffer");
    }

    printf("Execution complete\n");
}
} // namespace ai_rx_example
