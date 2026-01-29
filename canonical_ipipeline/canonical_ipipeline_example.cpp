/*
 * Canonical IPipeline Usage Example
 *
 * Demonstrates the expected lifecycle:
 * setup() -> configure_io() -> warmup() -> execute_stream()
 */

#include "channel_estimation_pipeline.hpp"

#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <iostream>
#include <memory>
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

        std::unique_ptr<framework::pipeline::IPipeline> pipeline =
            std::make_unique<channel_estimation::ChannelEstimationPipeline>(
                "canonical_channel_estimation",
                params);

        pipeline->setup();

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
        cuComplex* d_channel_estimates = nullptr;

        if (!check_cuda(cudaMalloc(&d_rx_pilots, num_pilots * sizeof(cuComplex)),
                        "Failed to allocate rx pilot buffer")) {
            return 1;
        }
        if (!check_cuda(cudaMalloc(&d_tx_pilots, num_pilots * sizeof(cuComplex)),
                        "Failed to allocate tx pilot buffer")) {
            return 1;
        }
        if (!check_cuda(cudaMalloc(&d_channel_estimates, num_data_subcarriers * sizeof(cuComplex)),
                        "Failed to allocate channel estimates buffer")) {
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

        framework::pipeline::PortInfo rx_port;
        rx_port.name = "rx_pilots";
        rx_port.tensors.resize(1);
        rx_port.tensors[0].device_ptr = d_rx_pilots;
        rx_port.tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            {static_cast<std::size_t>(num_pilots)});

        framework::pipeline::PortInfo tx_port;
        tx_port.name = "tx_pilots";
        tx_port.tensors.resize(1);
        tx_port.tensors[0].device_ptr = d_tx_pilots;
        tx_port.tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            {static_cast<std::size_t>(num_pilots)});

        framework::pipeline::PortInfo out_port;
        out_port.name = "channel_estimates";
        out_port.tensors.resize(1);
        out_port.tensors[0].device_ptr = d_channel_estimates;
        out_port.tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            {static_cast<std::size_t>(num_data_subcarriers)});

        std::vector<framework::pipeline::PortInfo> inputs = {rx_port, tx_port};
        std::vector<framework::pipeline::PortInfo> outputs = {out_port};

        framework::pipeline::DynamicParams dyn_params{};
        pipeline->configure_io(dyn_params, inputs, outputs, stream);

        pipeline->warmup(stream);
        pipeline->execute_stream(stream);

        if (!check_cuda(cudaStreamSynchronize(stream), "Failed to synchronize stream")) {
            return 1;
        }

        std::vector<cuComplex> channel_estimates(num_data_subcarriers);
        if (!check_cuda(cudaMemcpy(channel_estimates.data(), d_channel_estimates,
                                   num_data_subcarriers * sizeof(cuComplex), cudaMemcpyDeviceToHost),
                        "Failed to copy channel estimates")) {
            return 1;
        }

        std::cout << "Canonical IPipeline example completed. "
                  << "First estimate: (" << cuCrealf(channel_estimates[0])
                  << ", " << cuCimagf(channel_estimates[0]) << ")\n";

        cudaFree(d_rx_pilots);
        cudaFree(d_tx_pilots);
        cudaFree(d_channel_estimates);
        cudaStreamDestroy(stream);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
