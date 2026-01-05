#ifndef ML_CHANNEL_ESTIMATOR_TENSORRT_HPP
#define ML_CHANNEL_ESTIMATOR_TENSORRT_HPP

#include <string>
#include <vector>
#include <memory>

#include <cuda_runtime.h>
#include <cuComplex.h>

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#endif

namespace channel_estimation {

class MLChannelEstimatorTRT {
public:
    MLChannelEstimatorTRT(const std::string& model_path, int input_size, int output_size, bool use_fp16, int max_batch_size);
    ~MLChannelEstimatorTRT();

    // Run inference on input data (host or device pointer)
    std::vector<float> infer(const std::vector<float>& input);
    // Optionally, add device pointer version if needed

private:
    std::string model_path_;
    int input_size_;
    int output_size_;
    bool use_fp16_;
    int max_batch_size_;
#ifdef TENSORRT_AVAILABLE
    nvinfer1::IExecutionContext* context_{nullptr};
#endif
    // TensorRT engine/context handles would go here
};

} // namespace channel_estimation

#endif // ML_CHANNEL_ESTIMATOR_TENSORRT_HPP
