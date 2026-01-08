// ml_channel_estimator_tensorrt.cu
// Implementation of the infer method for MLChannelEstimatorTRT

#include "ml_channel_estimator_tensorrt.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace channel_estimation {


MLChannelEstimatorTRT::MLChannelEstimatorTRT(const std::string& module_id, const ChannelEstParams& params)
    : module_id_(module_id), params_(params),
      model_path_(params.model_path),
      input_size_(params.ml_input_size),
      output_size_(params.ml_output_size),
      use_fp16_(params.use_fp16),
      max_batch_size_(params.max_batch_size)
{
    setup_port_info();
    // TODO: Load TensorRT engine/model here
}

MLChannelEstimatorTRT::~MLChannelEstimatorTRT() {
    deallocate_gpu_memory();
    // TODO: Release TensorRT resources
}

void MLChannelEstimatorTRT::setup_port_info() {
    using namespace framework::tensor;
    input_ports_.resize(2);
    input_ports_[0].name = "rx_pilots";
    input_ports_[0].tensors.resize(1);
    input_ports_[0].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{static_cast<std::size_t>(params_.num_resource_blocks * 12 / params_.pilot_spacing)}
    );
    input_ports_[1].name = "tx_pilots";
    input_ports_[1].tensors.resize(1);
    input_ports_[1].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{static_cast<std::size_t>(params_.num_resource_blocks * 12 / params_.pilot_spacing)}
    );
    output_ports_.resize(1);
    output_ports_[0].name = "channel_estimates";
    output_ports_[0].tensors.resize(1);
    output_ports_[0].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(params_.num_resource_blocks * 12),
            static_cast<std::size_t>(params_.num_ofdm_symbols)
        }
    );
}

void MLChannelEstimatorTRT::setup_memory(const framework::pipeline::ModuleMemorySlice& /*memory_slice*/) {
    allocate_gpu_memory();
}

void MLChannelEstimatorTRT::warmup(cudaStream_t /*stream*/) {
    // Optionally run a dummy inference for warmup
}

void MLChannelEstimatorTRT::configure_io(
    const framework::pipeline::DynamicParams& /*params*/,
    cudaStream_t /*stream*/
) {
    // Update host descriptor for ML (if needed)
    // For ML, you may need to set up input/output device pointers for inference
}

void MLChannelEstimatorTRT::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    for (const auto& port : inputs) {
        if (port.name == "rx_pilots" && !port.tensors.empty()) {
            current_rx_pilots_ = static_cast<const cuComplex*>(port.tensors[0].device_ptr);
        } else if (port.name == "tx_pilots" && !port.tensors.empty()) {
            current_tx_pilots_ = static_cast<const cuComplex*>(port.tensors[0].device_ptr);
        } else if (port.name == "channel_estimates" && !port.tensors.empty()) {
            current_channel_estimates_ = static_cast<cuComplex*>(port.tensors[0].device_ptr);
        }
    }
}

std::vector<framework::pipeline::PortInfo> MLChannelEstimatorTRT::get_outputs() const {
    auto outputs = output_ports_;
    if (!outputs.empty() && !outputs[0].tensors.empty() && current_channel_estimates_) {
        outputs[0].tensors[0].device_ptr = current_channel_estimates_;
    }
    return outputs;
}

void MLChannelEstimatorTRT::execute(cudaStream_t stream) {
    // Validate pointers
    if (!current_rx_pilots_ || !current_tx_pilots_ || !current_channel_estimates_) {
        throw std::runtime_error("Missing input/output pointers");
    }

    // Calculate input sizes
    int num_pilots = params_.num_resource_blocks * 12 / params_.pilot_spacing;
    int num_data_subcarriers = params_.num_resource_blocks * 12 * params_.num_ofdm_symbols;

    // Copy device rx_pilots to host
    std::vector<float> rx_pilots_host(num_pilots * 2); // cuComplex: real+imag
    cudaError_t err = cudaMemcpyAsync(rx_pilots_host.data(), current_rx_pilots_, num_pilots * sizeof(cuComplex), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync rx_pilots failed");
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Stream sync failed after rx_pilots copy");
    }

    // Call infer
    std::vector<float> ml_output = infer(rx_pilots_host);
    if (ml_output.size() != num_data_subcarriers * 2) {
        throw std::runtime_error("ML inference output size mismatch");
    }

    // Copy output back to device (as cuComplex)
    std::vector<cuComplex> channel_estimates_host(num_data_subcarriers);
    for (int i = 0; i < num_data_subcarriers; ++i) {
        channel_estimates_host[i] = make_cuComplex(ml_output[2*i], ml_output[2*i+1]);
    }
    err = cudaMemcpyAsync(current_channel_estimates_, channel_estimates_host.data(), num_data_subcarriers * sizeof(cuComplex), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync channel_estimates failed");
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Stream sync failed after channel_estimates copy");
    }
}

void MLChannelEstimatorTRT::allocate_gpu_memory() {
    cudaError_t err;
    // Example: allocate output buffer (if needed for ML)
    size_t estimates_size = params_.num_resource_blocks * 12ULL * params_.num_ofdm_symbols * sizeof(cuComplex);
    err = cudaMalloc(&d_channel_estimates_, estimates_size);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_channel_estimates_ failed");
}

void MLChannelEstimatorTRT::deallocate_gpu_memory() {
    if (d_channel_estimates_) { cudaFree(d_channel_estimates_); d_channel_estimates_ = nullptr; }
}

framework::pipeline::ModuleMemoryRequirements MLChannelEstimatorTRT::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements reqs{};
    size_t total_bytes = 0;
    total_bytes += params_.num_rx_antennas * params_.num_ofdm_symbols * params_.num_resource_blocks * sizeof(cuComplex);
    total_bytes += params_.num_rx_antennas * params_.num_ofdm_symbols * params_.num_resource_blocks * 12 * sizeof(cuComplex);
    reqs.device_tensor_bytes = total_bytes;
    reqs.alignment = 256;
    return reqs;
}

framework::pipeline::OutputPortMemoryCharacteristics MLChannelEstimatorTRT::get_output_memory_characteristics(std::string_view port_name) const {
    framework::pipeline::OutputPortMemoryCharacteristics chars{};
    if (port_name == "channel_estimates") {
        chars.provides_fixed_address_for_zero_copy = true;
    }
    return chars;
}

std::vector<std::string> MLChannelEstimatorTRT::get_input_port_names() const {
    std::vector<std::string> names;
    for (const auto& port : input_ports_) {
        names.push_back(port.name);
    }
    return names;
}

std::vector<std::string> MLChannelEstimatorTRT::get_output_port_names() const {
    std::vector<std::string> names;
    for (const auto& port : output_ports_) {
        names.push_back(port.name);
    }
    return names;
}

std::vector<framework::tensor::TensorInfo> MLChannelEstimatorTRT::get_input_tensor_info(std::string_view port_name) const {
    std::vector<framework::tensor::TensorInfo> tensor_infos;
    for (const auto& port : input_ports_) {
        if (port.name == port_name) {
            for (const auto& tensor : port.tensors) {
                tensor_infos.push_back(tensor.tensor_info);
            }
        }
    }
    return tensor_infos;
}

std::vector<framework::tensor::TensorInfo> MLChannelEstimatorTRT::get_output_tensor_info(std::string_view port_name) const {
    std::vector<framework::tensor::TensorInfo> tensor_infos;
    for (const auto& port : output_ports_) {
        if (port.name == port_name) {
            for (const auto& tensor : port.tensors) {
                tensor_infos.push_back(tensor.tensor_info);
            }
        }
    }
    return tensor_infos;
}

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