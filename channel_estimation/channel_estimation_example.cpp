/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_estimation_pipeline.hpp"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <random>

using namespace channel_estimation;

int main(int argc, char** argv) {
    try {
        // Initialize CUDA
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        // Parse command-line for ML option (very basic)
        bool use_ml = false;
        std::string model_path;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--ml" && i + 1 < argc) {
                use_ml = true;
                model_path = argv[++i];
            }
        }

        // Create channel estimation parameters
        ChannelEstParams params;
        params.algorithm = use_ml ? ChannelEstAlgorithm::ML_TENSORRT : ChannelEstAlgorithm::LEAST_SQUARES;
        params.num_rx_antennas = 1;
        params.num_tx_layers = 1;
        params.num_resource_blocks = 25;  // Smaller for testing
        params.num_ofdm_symbols = 14;
        params.pilot_spacing = 4;
        params.beta_scaling = 1.0f;
        if (use_ml) {
            params.model_path = model_path;
            std::cout << "[INFO] ML model path: " << model_path << std::endl;
        }

        std::cout << "Creating channel estimation pipeline..." << std::endl;

        // Create the pipeline
        auto pipeline = std::make_unique<ChannelEstimationPipeline>(
            "test_channel_estimation",
            params
        );

        std::cout << "Pipeline ID: " << pipeline->get_pipeline_id() << std::endl;

        // Setup the pipeline
        std::cout << "Setting up pipeline..." << std::endl;
        pipeline->setup();

        // Create CUDA stream for operations
        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        // Warmup the pipeline
        std::cout << "Warming up pipeline..." << std::endl;
        pipeline->warmup(stream);
        
        // Generate test data
        int num_pilots = params.num_resource_blocks * 12 / params.pilot_spacing;
        int num_subcarriers = params.num_resource_blocks * 12;
        
        std::cout << "Generating test data..." << std::endl;
        std::cout << "Number of pilots: " << num_pilots << std::endl;
        std::cout << "Number of subcarriers: " << num_subcarriers << std::endl;
        
        // Create random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Generate pilot symbols (transmitted)
        std::vector<cuComplex> tx_pilots(num_pilots);
        for (int i = 0; i < num_pilots; ++i) {
            tx_pilots[i] = make_cuComplex(dis(gen), dis(gen));
        }
        
        // Generate received pilots (with channel applied)
        std::vector<cuComplex> rx_pilots(num_pilots);
        for (int i = 0; i < num_pilots; ++i) {
            // Simulate channel effect: H = 0.8 + 0.3j
            cuComplex channel = make_cuComplex(0.8f, 0.3f);
            rx_pilots[i] = cuCmulf(tx_pilots[i], channel);
            
            // Add some noise
            cuComplex noise = make_cuComplex(dis(gen) * 0.1f, dis(gen) * 0.1f);
            rx_pilots[i] = cuCaddf(rx_pilots[i], noise);
        }
        
        // Allocate device memory for test
        cuComplex* d_rx_pilots;
        cuComplex* d_tx_pilots;
        cuComplex* d_channel_estimates;
        
        err = cudaMalloc(&d_rx_pilots, num_pilots * sizeof(cuComplex));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for rx pilots" << std::endl;
            return -1;
        }
        
        err = cudaMalloc(&d_tx_pilots, num_pilots * sizeof(cuComplex));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for tx pilots" << std::endl;
            return -1;
        }
        
        err = cudaMalloc(&d_channel_estimates, num_subcarriers * params.num_ofdm_symbols * sizeof(cuComplex));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate device memory for channel estimates" << std::endl;
            return -1;
        }
        
        // Sanity check: ensure device pointers are different
        if (d_rx_pilots == d_tx_pilots) {
            std::cerr << "[ERROR] d_rx_pilots and d_tx_pilots point to the same device memory!" << std::endl;
            return -1;
        }
        
        // Copy data to device
        err = cudaMemcpy(d_rx_pilots, rx_pilots.data(), num_pilots * sizeof(cuComplex), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy rx pilots to device" << std::endl;
            return -1;
        }
        
        err = cudaMemcpy(d_tx_pilots, tx_pilots.data(), num_pilots * sizeof(cuComplex), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy tx pilots to device" << std::endl;
            return -1;
        }
        
        // Set up PortInfo structures for pipeline
        framework::pipeline::PortInfo rx_port, tx_port, out_port;
        rx_port.name = "rx_pilots";
        rx_port.tensors.resize(1);
        rx_port.tensors[0].device_ptr = d_rx_pilots;
        tx_port.name = "tx_pilots";
        tx_port.tensors.resize(1);
        tx_port.tensors[0].device_ptr = d_tx_pilots;
        out_port.name = "channel_estimates";
        out_port.tensors.resize(1);
        out_port.tensors[0].device_ptr = d_channel_estimates;

        std::cout << "[DEBUG] Example device ptrs: d_rx_pilots=" << d_rx_pilots
              << ", d_tx_pilots=" << d_tx_pilots
              << ", d_channel_estimates=" << d_channel_estimates << std::endl;

        std::vector<framework::pipeline::PortInfo> inputs = {rx_port, tx_port};
        std::vector<framework::pipeline::PortInfo> outputs = {out_port};

        // Configure pipeline I/O
        framework::pipeline::DynamicParams dyn_params; // Use default/empty for now
        pipeline->configure_io(dyn_params, inputs, outputs, stream);

        // Execute the pipeline
        pipeline->execute_stream(stream);

        // Copy results back to host
        std::vector<cuComplex> channel_estimates(num_subcarriers * params.num_ofdm_symbols);
        err = cudaMemcpy(channel_estimates.data(), d_channel_estimates, channel_estimates.size() * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy channel estimates to host: " << cudaGetErrorString(err) << std::endl;
            std::cerr << "  host ptr: " << static_cast<const void*>(channel_estimates.data())
                      << ", device ptr: " << static_cast<const void*>(d_channel_estimates)
                      << ", size: " << (channel_estimates.size() * sizeof(cuComplex)) << std::endl;
            return -1;
        }

        std::cout << "Channel estimation pipeline executed successfully!" << std::endl;
        std::cout << "First 5 channel estimates:" << std::endl;
        for (size_t i = 0; i < 5 && i < channel_estimates.size(); ++i) {
            std::cout << "  [" << i << "]: (" << cuCrealf(channel_estimates[i]) << ", " << cuCimagf(channel_estimates[i]) << ")" << std::endl;
        }
        
        // Cleanup
        cudaFree(d_rx_pilots);
        cudaFree(d_tx_pilots);
        cudaFree(d_channel_estimates);
        cudaStreamDestroy(stream);
        
        std::cout << "Test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}