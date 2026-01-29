/*
 * Path Loss Prediction Example (Module API)
 */

#include "path_loss_prediction_module.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

std::vector<float> generate_features(int batch_size, int num_features) {
    std::vector<float> features(static_cast<size_t>(batch_size) * num_features, 0.0f);
    for (int i = 0; i < batch_size; ++i) {
        float d_km = 0.1f + 0.01f * static_cast<float>(i);
        float logd = std::log10(d_km);
        float rsrp = -70.0f - 20.0f * logd;
        float sinr = 15.0f - 10.0f * d_km;

        size_t base = static_cast<size_t>(i) * num_features;
        features[base + 0] = rsrp;
        features[base + 1] = sinr;
        features[base + 2] = d_km * 1000.0f;
        features[base + 3] = 3.5f;
        features[base + 4] = 2.0f;
        features[base + 5] = 1.0f;
        features[base + 6] = 0.2f;
        features[base + 7] = 0.1f;
    }
    return features;
}

bool check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main() {
    path_loss_prediction::PathLossModelParams params;
    params.batch_size = 16;
    params.num_features = 8;
    params.hidden_size = 16;
    params.model_type = path_loss_prediction::ModelType::XGBoostStyle;

    path_loss_prediction::PathLossPredictionModule module("path_loss_module_example", params);

    auto requirements = module.get_requirements();
    framework::pipeline::ModuleMemorySlice slice{};

    if (requirements.device_tensor_bytes > 0) {
        if (!check_cuda(cudaMalloc(&slice.device_tensor_ptr, requirements.device_tensor_bytes),
                        "Failed to allocate module tensor")) {
            return 1;
        }
        slice.device_tensor_bytes = requirements.device_tensor_bytes;
    }

    module.setup_memory(slice);

    cudaStream_t stream{};
    if (!check_cuda(cudaStreamCreate(&stream), "Failed to create CUDA stream")) {
        return 1;
    }

    auto features = generate_features(params.batch_size, params.num_features);

    float* d_features = nullptr;
    size_t feature_bytes =
        static_cast<size_t>(params.batch_size) *
        static_cast<size_t>(params.num_features) * sizeof(float);

    if (!check_cuda(cudaMalloc(&d_features, feature_bytes), "Failed to allocate feature buffer")) {
        return 1;
    }

    if (!check_cuda(cudaMemcpy(d_features, features.data(), feature_bytes, cudaMemcpyHostToDevice),
                    "Failed to copy features")) {
        return 1;
    }

    std::vector<framework::pipeline::PortInfo> inputs(1);
    inputs[0].name = "features";
    inputs[0].tensors.resize(1);
    inputs[0].tensors[0].device_ptr = d_features;
    inputs[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorR32F,
        {static_cast<size_t>(params.batch_size),
         static_cast<size_t>(params.num_features)});

    module.set_inputs(inputs);
    module.configure_io({}, stream);
    module.execute(stream);

    if (!check_cuda(cudaStreamSynchronize(stream), "Failed to synchronize stream")) {
        return 1;
    }

    auto outputs = module.get_outputs();
    if (outputs.empty() || outputs[0].tensors.empty()) {
        std::cerr << "Module produced no outputs\n";
        return 1;
    }

    std::vector<float> path_loss(params.batch_size, 0.0f);
    if (!check_cuda(cudaMemcpy(path_loss.data(), outputs[0].tensors[0].device_ptr,
                               params.batch_size * sizeof(float), cudaMemcpyDeviceToHost),
                    "Failed to copy outputs")) {
        return 1;
    }

    std::cout << "Module API predicted path loss (dB):\n";
    for (size_t i = 0; i < std::min<size_t>(5, path_loss.size()); ++i) {
        std::cout << "  UE " << i << ": " << path_loss[i] << " dB\n";
    }

    cudaFree(d_features);
    cudaFree(slice.device_tensor_ptr);
    cudaStreamDestroy(stream);

    return 0;
}
