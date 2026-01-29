/*
 * Canonical Module API Example
 *
 * Demonstrates IModule lifecycle directly:
 * get_requirements() -> setup_memory() -> set_inputs() -> configure_io() ->
 * warmup() -> execute()
 */

#include "channel_estimation_module.hpp"

#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <iostream>
#include <vector>

namespace {

bool check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main() {
    try {
        if (!check_cuda(cudaSetDevice(0), "Failed to set CUDA device")) {
            return 1;
        }

        channel_estimation::ChannelEstParams params;
        params.algorithm = channel_estimation::ChannelEstAlgorithm::LEAST_SQUARES;
        params.num_rx_antennas = 1;
        params.num_tx_layers = 1;
        params.num_resource_blocks = 12;
        params.num_ofdm_symbols = 14;
        params.pilot_spacing = 4;

        channel_estimation::ChannelEstimator module("canonical_channel_estimator", params);

        auto requirements = module.get_requirements();
        framework::pipeline::ModuleMemorySlice slice{};

        if (requirements.device_tensor_bytes > 0) {
            if (!check_cuda(cudaMalloc(&slice.device_tensor_ptr, requirements.device_tensor_bytes),
                            "Failed to allocate device tensor")) {
                return 1;
            }
            slice.device_tensor_bytes = requirements.device_tensor_bytes;
        }

        if (requirements.static_kernel_descriptor_bytes > 0) {
            if (!check_cuda(cudaMallocHost(&slice.static_kernel_descriptor_cpu_ptr,
                                           requirements.static_kernel_descriptor_bytes),
                            "Failed to allocate static descriptor host memory")) {
                return 1;
            }
            if (!check_cuda(cudaMalloc(&slice.static_kernel_descriptor_gpu_ptr,
                                       requirements.static_kernel_descriptor_bytes),
                            "Failed to allocate static descriptor device memory")) {
                return 1;
            }
            slice.static_kernel_descriptor_bytes = requirements.static_kernel_descriptor_bytes;
        }

        if (requirements.dynamic_kernel_descriptor_bytes > 0) {
            if (!check_cuda(cudaMallocHost(&slice.dynamic_kernel_descriptor_cpu_ptr,
                                           requirements.dynamic_kernel_descriptor_bytes),
                            "Failed to allocate dynamic descriptor host memory")) {
                return 1;
            }
            if (!check_cuda(cudaMalloc(&slice.dynamic_kernel_descriptor_gpu_ptr,
                                       requirements.dynamic_kernel_descriptor_bytes),
                            "Failed to allocate dynamic descriptor device memory")) {
                return 1;
            }
            slice.dynamic_kernel_descriptor_bytes = requirements.dynamic_kernel_descriptor_bytes;
        }

        module.setup_memory(slice);

        cudaStream_t stream{};
        if (!check_cuda(cudaStreamCreate(&stream), "Failed to create CUDA stream")) {
            return 1;
        }

        const int num_pilots = params.num_resource_blocks * 12 / params.pilot_spacing;
        const int num_data_subcarriers = params.num_resource_blocks * 12 * params.num_ofdm_symbols;

        std::vector<cuComplex> tx_pilots(num_pilots);
        std::vector<cuComplex> rx_pilots(num_pilots);
        for (int i = 0; i < num_pilots; ++i) {
            const float real = 0.5f + static_cast<float>(i) * 0.01f;
            const float imag = -0.25f + static_cast<float>(i) * 0.005f;
            tx_pilots[i] = make_cuComplex(real, imag);
            rx_pilots[i] = make_cuComplex(real * 0.9f, imag * 1.1f);
        }

        cuComplex* d_rx_pilots = nullptr;
        cuComplex* d_tx_pilots = nullptr;
        if (!check_cuda(cudaMalloc(&d_rx_pilots, num_pilots * sizeof(cuComplex)),
                        "Failed to allocate rx pilot buffer")) {
            return 1;
        }
        if (!check_cuda(cudaMalloc(&d_tx_pilots, num_pilots * sizeof(cuComplex)),
                        "Failed to allocate tx pilot buffer")) {
            return 1;
        }

        if (!check_cuda(cudaMemcpy(d_rx_pilots, rx_pilots.data(),
                                   num_pilots * sizeof(cuComplex), cudaMemcpyHostToDevice),
                        "Failed to copy rx pilots")) {
            return 1;
        }
        if (!check_cuda(cudaMemcpy(d_tx_pilots, tx_pilots.data(),
                                   num_pilots * sizeof(cuComplex), cudaMemcpyHostToDevice),
                        "Failed to copy tx pilots")) {
            return 1;
        }

        std::vector<framework::pipeline::PortInfo> inputs(2);
        inputs[0].name = "rx_pilots";
        inputs[0].tensors.resize(1);
        inputs[0].tensors[0].device_ptr = d_rx_pilots;
        inputs[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            {static_cast<std::size_t>(num_pilots)});

        inputs[1].name = "tx_pilots";
        inputs[1].tensors.resize(1);
        inputs[1].tensors[0].device_ptr = d_tx_pilots;
        inputs[1].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            {static_cast<std::size_t>(num_pilots)});

        module.set_inputs(inputs);
        module.configure_io({}, stream);
        module.warmup(stream);
        module.execute(stream);

        if (!check_cuda(cudaStreamSynchronize(stream), "Failed to synchronize stream")) {
            return 1;
        }

        auto outputs = module.get_outputs();
        if (outputs.empty() || outputs[0].tensors.empty()) {
            std::cerr << "Module produced no output" << std::endl;
            return 1;
        }

        std::vector<cuComplex> channel_estimates(num_data_subcarriers);
        if (!check_cuda(cudaMemcpy(channel_estimates.data(), outputs[0].tensors[0].device_ptr,
                                   num_data_subcarriers * sizeof(cuComplex), cudaMemcpyDeviceToHost),
                        "Failed to copy channel estimates")) {
            return 1;
        }

        std::cout << "Canonical module API example completed. "
                  << "First estimate: (" << cuCrealf(channel_estimates[0])
                  << ", " << cuCimagf(channel_estimates[0]) << ")\n";

        cudaFree(d_rx_pilots);
        cudaFree(d_tx_pilots);
        cudaStreamDestroy(stream);

        if (slice.dynamic_kernel_descriptor_cpu_ptr) {
            cudaFreeHost(slice.dynamic_kernel_descriptor_cpu_ptr);
        }
        if (slice.dynamic_kernel_descriptor_gpu_ptr) {
            cudaFree(slice.dynamic_kernel_descriptor_gpu_ptr);
        }
        if (slice.static_kernel_descriptor_cpu_ptr) {
            cudaFreeHost(slice.static_kernel_descriptor_cpu_ptr);
        }
        if (slice.static_kernel_descriptor_gpu_ptr) {
            cudaFree(slice.static_kernel_descriptor_gpu_ptr);
        }
        if (slice.device_tensor_ptr) {
            cudaFree(slice.device_tensor_ptr);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
