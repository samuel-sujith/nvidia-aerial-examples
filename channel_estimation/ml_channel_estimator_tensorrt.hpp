
#ifndef ML_CHANNEL_ESTIMATOR_TENSORRT_HPP
#define ML_CHANNEL_ESTIMATOR_TENSORRT_HPP

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <memory>
#include <string>
#include <vector>
#include <span>

#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"
#include "tensor/data_types.hpp"
#include "channel_estimation_module.hpp" // For ChannelEstParams, IChannelEstimator

namespace channel_estimation {

class MLChannelEstimatorTRT final : public IChannelEstimator, public framework::pipeline::IModule,
                                    public framework::pipeline::IAllocationInfoProvider,
                                    public framework::pipeline::IStreamExecutor {
public:
    MLChannelEstimatorTRT(const std::string& module_id, const ChannelEstParams& params);
    ~MLChannelEstimatorTRT() override;

    // IModule interface implementation
    [[nodiscard]] std::string_view get_type_id() const override { return "ml_channel_estimator_trt"; }
    [[nodiscard]] std::string_view get_instance_id() const override { return module_id_; }

    void setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) override;
    void warmup(cudaStream_t stream) override;

    void configure_io(
        const framework::pipeline::DynamicParams& params,
        cudaStream_t stream
    ) override;

    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;

    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;

    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;
    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;

    [[nodiscard]] framework::pipeline::ModuleMemoryRequirements get_requirements() const override;

    [[nodiscard]] framework::pipeline::OutputPortMemoryCharacteristics
    get_output_memory_characteristics(std::string_view port_name) const override;

    void set_inputs(std::span<const framework::pipeline::PortInfo> inputs) override;
    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override;

    // IModule casting methods
    framework::pipeline::IGraphNodeProvider* as_graph_node_provider() override { return nullptr; }
    framework::pipeline::IStreamExecutor* as_stream_executor() override { return this; }

    // IStreamExecutor interface
    void execute(cudaStream_t stream) override;

private:
    std::string module_id_;
    ChannelEstParams params_;

    // Port information
    std::vector<framework::pipeline::PortInfo> input_ports_;
    std::vector<framework::pipeline::PortInfo> output_ports_;

    // ML resources (TensorRT, etc.)
    std::string model_path_;
    int input_size_;
    int output_size_;
    bool use_fp16_;
    int max_batch_size_;
#ifdef TENSORRT_AVAILABLE
    nvinfer1::IExecutionContext* context_{nullptr};
#endif

    // Device memory pointers (for output, etc.)
    cuComplex* d_channel_estimates_{nullptr};

    // Current tensor pointers (set during set_inputs)
    const cuComplex* current_rx_pilots_{nullptr};
    const cuComplex* current_tx_pilots_{nullptr};
    cuComplex* current_channel_estimates_{nullptr};

    void allocate_gpu_memory();
    void deallocate_gpu_memory();
    void setup_port_info();
};

} // namespace channel_estimation

#endif // ML_CHANNEL_ESTIMATOR_TENSORRT_HPP
