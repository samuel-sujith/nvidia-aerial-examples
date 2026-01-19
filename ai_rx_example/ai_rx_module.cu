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

        // Allocate device memory for input/output
        size_t input_size = params_.num_symbols * sizeof(float);
        size_t output_size = params_.num_symbols * sizeof(float);

        cudaError_t err = cudaMalloc(&d_trt_input_, input_size);
        if (err != cudaSuccess) {
            printf("Failed to allocate TensorRT input memory: %s\n", cudaGetErrorString(err));
            return false;
        }

        err = cudaMalloc(&d_trt_output_, output_size);
        if (err != cudaSuccess) {
            printf("Failed to allocate TensorRT output memory: %s\n", cudaGetErrorString(err));
            return false;
        }

        printf("TensorRT engine initialized successfully\n");
        return true;

    } catch (const std::exception& e) {
        printf("Exception during TensorRT initialization: %s\n", e.what());
        return false;
    }
#else
    printf("TensorRT not available - falling back to default implementation\n");
    return false;
#endif
}

} // namespace ai_rx_example
