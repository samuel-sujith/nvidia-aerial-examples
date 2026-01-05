// ml_channel_estimator_tensorrt.cu
// Implementation of the infer method for MLChannelEstimatorTRT

#include "ml_channel_estimator_tensorrt.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace channel_estimation {

std::vector<float> MLChannelEstimatorTRT::infer(const std::vector<float>& input) {
#ifdef TENSORRT_AVAILABLE
    int input_size = input.size();
    int output_size = input_size; // Adjust if model output size differs
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_input, input_size * sizeof(float));
    if (err != cudaSuccess) return {};
    err = cudaMalloc(&d_output, output_size * sizeof(float));
    if (err != cudaSuccess) { cudaFree(d_input); return {}; }
    err = cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_input); cudaFree(d_output); return {}; }

    // Set up TensorRT explicit I/O bindings (assumes input/output names are "input" and "output")
    context_->setTensorAddress("input", d_input);
    context_->setTensorAddress("output", d_output);

    // Run inference (no batch, default stream)
    bool success = context_->enqueueV3(0); // 0 = default stream
    std::vector<float> output;
    if (success) {
        output.resize(output_size);
        err = cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            output.clear();
        }
    }
    cudaFree(d_input);
    cudaFree(d_output);
    return output;
#else
    // TensorRT not available, return empty result
    return {};
#endif
}

} // namespace channel_estimation