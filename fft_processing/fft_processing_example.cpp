#include "fft_processing_pipeline.hpp"
#include "common/test_utils.hpp"

#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace {

void generate_test_signal(std::vector<cuComplex>& signal, int fft_size, int num_antennas, int batch_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    int total_samples = fft_size * num_antennas * batch_size;
    signal.resize(total_samples);
    
    // Generate complex Gaussian noise with some sinusoidal components
    for (int i = 0; i < total_samples; ++i) {
        int sample_idx = i % fft_size;
        
        // Add sinusoidal components at specific frequencies
        float freq1 = 2.0f * M_PI * 10.0f / fft_size; // Low frequency component
        float freq2 = 2.0f * M_PI * 100.0f / fft_size; // High frequency component
        
        float real_part = dist(gen) + 0.5f * std::cos(freq1 * sample_idx) + 0.3f * std::cos(freq2 * sample_idx);
        float imag_part = dist(gen) + 0.5f * std::sin(freq1 * sample_idx) + 0.3f * std::sin(freq2 * sample_idx);
        
        signal[i] = make_cuComplex(real_part, imag_part);
    }
}

void print_signal_stats(const std::vector<cuComplex>& signal, const std::string& name) {
    float mean_power = 0.0f;
    float max_magnitude = 0.0f;
    
    for (const auto& sample : signal) {
        float magnitude = cuCabsf(sample);
        mean_power += magnitude * magnitude;
        max_magnitude = std::max(max_magnitude, magnitude);
    }
    
    mean_power /= signal.size();
    
    std::cout << name << " - Samples: " << signal.size() 
              << ", Mean Power: " << mean_power 
              << ", Max Magnitude: " << max_magnitude << std::endl;
}

} // anonymous namespace

int main() {
    try {
        std::cout << "Creating FFT processing pipeline..." << std::endl;
        
        // Configure FFT parameters
        fft_processing::FFTParams params;
        params.fft_size = 1024;
        params.num_antennas = 4;
        params.batch_size = 2;
        params.direction = fft_processing::FFTDirection::FORWARD;
        params.normalize = true;
        params.enable_windowing = true;
        
        // Create pipeline
        auto pipeline = std::make_unique<fft_processing::FFTProcessingPipeline>(
            "test_fft_processing", params
        );
        
        std::cout << "Pipeline ID: " << pipeline->get_pipeline_id() << std::endl;
        std::cout << "External inputs: " << pipeline->get_num_external_inputs() << std::endl;
        std::cout << "External outputs: " << pipeline->get_num_external_outputs() << std::endl;
        
        // Setup pipeline
        std::cout << "Setting up pipeline..." << std::endl;
        pipeline->setup();
        
        // Create CUDA stream
        cudaStream_t stream;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        // Generate test data
        std::cout << "Generating test data..." << std::endl;
        int total_samples = params.fft_size * params.num_antennas * params.batch_size;
        
        std::vector<cuComplex> input_signal;
        generate_test_signal(input_signal, params.fft_size, params.num_antennas, params.batch_size);
        print_signal_stats(input_signal, "Input Signal");
        
        std::vector<cuComplex> output_signal(total_samples);
        
        // Allocate GPU memory for input/output
        cuComplex* d_input;
        cuComplex* d_output;
        size_t data_size = total_samples * sizeof(cuComplex);
        
        err = cudaMalloc(&d_input, data_size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate input memory: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        err = cudaMalloc(&d_output, data_size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate output memory: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        // Copy input data to GPU
        err = cudaMemcpy(d_input, input_signal.data(), data_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy input data to GPU: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        // Setup PortInfo for pipeline execution
        std::vector<framework::pipeline::PortInfo> input_ports(1);
        std::vector<framework::pipeline::PortInfo> output_ports(1);
        
        // Configure input port
        input_ports[0].name = "input_signal";
        input_ports[0].tensors.resize(1);
        input_ports[0].tensors[0].device_ptr = d_input;
        input_ports[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            std::vector<std::size_t>{
                static_cast<std::size_t>(params.batch_size),
                static_cast<std::size_t>(params.num_antennas),
                static_cast<std::size_t>(params.fft_size)
            }
        );
        
        // Configure output port
        output_ports[0].name = "output_signal";
        output_ports[0].tensors.resize(1);
        output_ports[0].tensors[0].device_ptr = d_output;
        output_ports[0].tensors[0].tensor_info = input_ports[0].tensors[0].tensor_info;
        
        // Configure pipeline I/O
        framework::pipeline::DynamicParams dynamic_params{};
        pipeline->configure_io(dynamic_params, input_ports, output_ports, stream);

        // Warmup pipeline
        std::cout << "Warming up pipeline..." << std::endl;
        pipeline->warmup(stream);
        
        // Execute pipeline
        std::cout << "Executing FFT processing..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        pipeline->execute_stream(stream);
        
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "Failed to synchronize stream: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Copy results back to host
        err = cudaMemcpy(output_signal.data(), d_output, data_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy results from GPU: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        // Print results
        print_signal_stats(output_signal, "Output Signal (FFT)");
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Processing time: " << duration.count() << " microseconds" << std::endl;
        
        // Test inverse FFT
        std::cout << "\\nTesting inverse FFT..." << std::endl;
        params.direction = fft_processing::FFTDirection::INVERSE;
        
        auto inverse_pipeline = std::make_unique<fft_processing::FFTProcessingPipeline>(
            "test_ifft_processing", params
        );

        inverse_pipeline->setup();
        
        // Use FFT output as IFFT input
        err = cudaMemcpy(d_input, d_output, data_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy data on GPU: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        inverse_pipeline->configure_io(dynamic_params, input_ports, output_ports, stream);
        inverse_pipeline->warmup(stream);
        inverse_pipeline->execute_stream(stream);
        
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            std::cerr << "Failed to synchronize IFFT stream: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        // Copy results back
        std::vector<cuComplex> reconstructed_signal(total_samples);
        err = cudaMemcpy(reconstructed_signal.data(), d_output, data_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy IFFT results from GPU: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        print_signal_stats(reconstructed_signal, "Reconstructed Signal (IFFT)");
        
        // Calculate reconstruction error
        float total_error = 0.0f;
        for (int i = 0; i < total_samples; ++i) {
            cuComplex diff = cuCsubf(input_signal[i], reconstructed_signal[i]);
            total_error += cuCabsf(diff);
        }
        float mean_error = total_error / total_samples;
        std::cout << "Mean reconstruction error: " << mean_error << std::endl;
        
        // Cleanup
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
        
        std::cout << "\\nFFT processing pipeline created successfully!" << std::endl;
        std::cout << "Forward FFT size: " << params.fft_size << std::endl;
        std::cout << "Number of antennas: " << params.num_antennas << std::endl;
        std::cout << "Batch size: " << params.batch_size << std::endl;
        std::cout << "Windowing enabled: " << (params.enable_windowing ? "Yes" : "No") << std::endl;
        std::cout << "Note: Full execution completed with CUFFT integration" << std::endl;
        std::cout << "Test completed successfully!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}