// ml_channel_estimator_tensorrt.cu
// Implementation of the infer method for MLChannelEstimatorTRT

#include "ml_channel_estimator_tensorrt.hpp"
#include <cuda_runtime.h>
#include <vector>

std::vector<float> MLChannelEstimatorTRT::infer(const std::vector<float>& input) {
    int input_size = input.size();
    int output_size = input_size;
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    void* bindings[2] = {d_input, d_output};
    context_->enqueueV2(bindings, 0, nullptr);
    std::vector<float> output(output_size);
    cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}