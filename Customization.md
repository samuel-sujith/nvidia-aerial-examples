# Creating Custom Signal Processing Modules

This guide provides step-by-step instructions for creating new signal processing modules following the patterns established in the examples. The examples use framework stubs to demonstrate algorithms without requiring full Aerial Framework installation.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Patterns](#architecture-patterns)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Module Implementation](#module-implementation)
5. [Pipeline Implementation](#pipeline-implementation)
6. [Framework Stubs Integration](#framework-stubs-integration)
7. [Build System Integration](#build-system-integration)
8. [Testing and Validation](#testing-and-validation)
9. [Documentation Guidelines](#documentation-guidelines)
10. [Best Practices](#best-practices)

## Overview

The examples follow a simplified modular architecture where signal processing functionality is implemented as:

- **Modules**: Core processing units that execute specific algorithms
- **Pipelines**: Orchestration layer that manages module execution and data flow
- **Framework Stubs**: Compatibility layer that provides framework-like interfaces
- **Memory Management**: Efficient GPU memory handling using CUDA APIs
- **Factory Patterns**: Flexible instantiation and configuration

## Architecture Patterns

### Core Includes

```cpp
// Standard includes for signal processing modules
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <complex>
#include <vector>
#include <memory>
#include <chrono>

// Framework stubs for compatibility (if needed)
// Copy framework_stubs.cpp from any existing module
```

### Typical Module Structure

```
your_module/
├── your_module.hpp              # Module interface and declarations
├── your_module.cu              # CUDA kernel implementations (if needed)
├── your_pipeline.hpp           # Pipeline interface and factory
├── your_pipeline.cpp           # Pipeline implementation
├── your_example.cpp            # Simple usage example
├── your_comprehensive_example.cpp  # Advanced example with benchmarks
├── framework_stubs.cpp         # Framework compatibility layer (if needed)
├── README.md                   # Module documentation
└── CMakeLists.txt              # Build configuration
```
├── CMakeLists.txt              # Build configuration
└── README.md                   # Module documentation
```

## Step-by-Step Implementation

### Step 1: Project Setup

Create the directory structure for your new module:

```bash
# Create module directory
mkdir your_custom_module
cd your_custom_module

# Create basic files
touch your_custom_module.hpp
touch your_custom_module.cu          # If CUDA kernels needed
touch your_custom_pipeline.hpp
touch your_custom_pipeline.cpp
touch your_custom_example.cpp
touch your_comprehensive_example.cpp
touch CMakeLists.txt
touch README.md
```

### Step 2: Define Module Interface

Create the header file (`your_custom_module.hpp`):

```cpp
#pragma once

#include <aerial_framework/module/imodule.hpp>
#include <aerial_framework/tensor/tensor_info.hpp>
#include <aerial_framework/task/task_result.hpp>
#include <cuda_runtime.h>
#include <memory>

namespace aerial::examples::your_module {

/**
 * @brief Custom processing module for [describe your functionality]
 */
class YourCustomModule final : public module::IModule {
public:
    /**
     * @brief Configuration parameters for your module
     */
    struct Config {
        std::size_t parameter1 = 64;        // Example parameter
        std::size_t parameter2 = 8;         // Example parameter
        float threshold = 0.5f;             // Example threshold
        bool enable_optimization = true;    // Enable optimizations
    };

    /**
     * @brief Construct custom module
     * @param config Configuration parameters
     * @param stream CUDA stream for operations
     */
    YourCustomModule(const Config& config, cudaStream_t stream = nullptr);

    /**
     * @brief Destructor - cleanup resources
     */
    ~YourCustomModule() override;

    // IModule interface implementation
    std::string_view get_module_id() const override {
        return "your_custom_module_v1";
    }

    task::TaskResult execute(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs,
        const task::CancellationToken& token = {}
    ) override;

    bool is_input_ready(std::size_t input_index) const override;
    bool is_output_ready(std::size_t output_index) const override;

    /**
     * @brief Process your custom algorithm
     * @param input_data Input tensor data
     * @param output_data Output tensor data
     * @return Processing result
     */
    task::TaskResult process_algorithm(
        const tensor::TensorInfo& input_data,
        tensor::TensorInfo& output_data
    );

    /**
     * @brief Update module configuration
     * @param new_config New configuration parameters
     */
    void update_config(const Config& new_config);

    /**
     * @brief Get current configuration
     */
    const Config& get_config() const { return config_; }

private:
    Config config_;
    cudaStream_t stream_;
    bool is_initialized_ = false;

    // Internal methods
    void initialize_cuda_resources();
    void cleanup_cuda_resources();
    void validate_inputs(const std::vector<tensor::TensorInfo>& inputs);
    void validate_outputs(const std::vector<tensor::TensorInfo>& outputs);
};

/**
 * @brief Factory for creating custom modules
 */
class YourCustomModuleFactory {
public:
    static std::unique_ptr<YourCustomModule> create(
        const YourCustomModule::Config& config,
        cudaStream_t stream = nullptr
    );

    static YourCustomModule::Config create_default_config();
};

} // namespace aerial::examples::your_module
```

### Step 3: Implement CUDA Kernels (if needed)

Create CUDA implementation (`your_custom_module.cu`):

```cpp
#include "your_custom_module.hpp"
#include <aerial_framework/cuda_utils/cuda_error.hpp>
#include <aerial_framework/logging/logger.hpp>

namespace aerial::examples::your_module {

namespace {
    // CUDA kernel for your custom processing
    __global__ void your_custom_kernel(
        const float* input_data,
        float* output_data,
        int num_elements,
        float threshold
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < num_elements) {
            // Your custom processing logic here
            float value = input_data[idx];
            
            // Example: Apply threshold and transformation
            if (value > threshold) {
                output_data[idx] = value * 2.0f;  // Example transformation
            } else {
                output_data[idx] = value * 0.5f;  // Example transformation
            }
        }
    }

    // Helper function for optimal block size calculation
    dim3 calculate_grid_size(int num_elements, int block_size = 256) {
        int grid_size = (num_elements + block_size - 1) / block_size;
        return dim3(grid_size, 1, 1);
    }

} // anonymous namespace

YourCustomModule::YourCustomModule(const Config& config, cudaStream_t stream)
    : config_(config), stream_(stream) {
    
    if (stream_ == nullptr) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    
    initialize_cuda_resources();
    AERIAL_LOG_INFO("Custom module created with parameters: {}, {}", 
                   config_.parameter1, config_.parameter2);
}

YourCustomModule::~YourCustomModule() {
    cleanup_cuda_resources();
    
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

task::TaskResult YourCustomModule::execute(
    const std::vector<tensor::TensorInfo>& inputs,
    std::vector<tensor::TensorInfo>& outputs,
    const task::CancellationToken& token) {
    
    if (token.is_cancellation_requested()) {
        return task::TaskResult(task::TaskStatus::Cancelled);
    }

    try {
        validate_inputs(inputs);
        validate_outputs(outputs);
        
        // Process your algorithm
        auto result = process_algorithm(inputs[0], outputs[0]);
        
        if (result.status != task::TaskStatus::Completed) {
            return result;
        }
        
        return task::TaskResult(task::TaskStatus::Completed);
        
    } catch (const std::exception& e) {
        AERIAL_LOG_ERROR("Custom module execution failed: {}", e.what());
        return task::TaskResult(task::TaskStatus::Failed, e.what());
    }
}

task::TaskResult YourCustomModule::process_algorithm(
    const tensor::TensorInfo& input_data,
    tensor::TensorInfo& output_data) {
    
    const auto* input_ptr = static_cast<const float*>(input_data.data);
    auto* output_ptr = static_cast<float*>(output_data.data);
    
    std::size_t num_elements = input_data.total_elements();
    
    // Launch CUDA kernel
    dim3 block_size(256);
    dim3 grid_size = calculate_grid_size(num_elements, 256);
    
    your_custom_kernel<<<grid_size, block_size, 0, stream_>>>(
        input_ptr, output_ptr, num_elements, config_.threshold
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    if (stream_ != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    
    return task::TaskResult(task::TaskStatus::Completed);
}

void YourCustomModule::initialize_cuda_resources() {
    // Initialize any additional CUDA resources here
    // e.g., cuBLAS handles, memory allocations, etc.
    is_initialized_ = true;
}

void YourCustomModule::cleanup_cuda_resources() {
    // Cleanup CUDA resources
    is_initialized_ = false;
}

void YourCustomModule::validate_inputs(const std::vector<tensor::TensorInfo>& inputs) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("Expected exactly 1 input tensor");
    }
    
    if (inputs[0].data_type != tensor::DataType::Float32) {
        throw std::invalid_argument("Expected Float32 input data type");
    }
}

void YourCustomModule::validate_outputs(const std::vector<tensor::TensorInfo>& outputs) {
    if (outputs.size() != 1) {
        throw std::invalid_argument("Expected exactly 1 output tensor");
    }
}

bool YourCustomModule::is_input_ready(std::size_t input_index) const {
    return input_index == 0 && is_initialized_;
}

bool YourCustomModule::is_output_ready(std::size_t output_index) const {
    return output_index == 0 && is_initialized_;
}

// Factory implementation
std::unique_ptr<YourCustomModule> YourCustomModuleFactory::create(
    const YourCustomModule::Config& config,
    cudaStream_t stream) {
    
    return std::make_unique<YourCustomModule>(config, stream);
}

YourCustomModule::Config YourCustomModuleFactory::create_default_config() {
    YourCustomModule::Config config;
    config.parameter1 = 64;
    config.parameter2 = 8;
    config.threshold = 0.5f;
    config.enable_optimization = true;
    return config;
}

} // namespace aerial::examples::your_module
```

## Pipeline Implementation

### Step 4: Create Pipeline Interface

Create pipeline header (`your_custom_pipeline.hpp`):

```cpp
#pragma once

#include <aerial_framework/pipeline/ipipeline.hpp>
#include <aerial_framework/memory/memory_pool.hpp>
#include <aerial_framework/cuda_utils/cuda_context.hpp>
#include "your_custom_module.hpp"

namespace aerial::examples::your_module {

/**
 * @brief Pipeline for orchestrating custom processing modules
 */
class YourCustomPipeline final : public pipeline::IPipeline {
public:
    /**
     * @brief Pipeline configuration
     */
    struct Config {
        YourCustomModule::Config module_config;
        std::size_t batch_size = 32;
        bool enable_batching = true;
        std::string processing_mode = "standard";
    };

    /**
     * @brief Performance metrics
     */
    struct Metrics {
        float average_latency_ms = 0.0f;
        float throughput_ops_per_sec = 0.0f;
        std::size_t processed_batches = 0;
        float total_processing_time_ms = 0.0f;
    };

    /**
     * @brief Construct custom pipeline
     */
    YourCustomPipeline(
        const Config& config,
        std::shared_ptr<memory::MemoryPool> memory_pool,
        std::shared_ptr<cuda_utils::CudaContext> cuda_context
    );

    /**
     * @brief Destructor
     */
    ~YourCustomPipeline() override;

    // IPipeline interface
    std::string_view get_pipeline_id() const override {
        return "your_custom_pipeline_v1";
    }

    task::TaskResult initialize(const std::vector<tensor::TensorInfo>& inputs) override;
    task::TaskResult execute(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs,
        const task::CancellationToken& token = {}
    ) override;
    void finalize() override;

    /**
     * @brief Process data with your custom algorithm
     */
    task::TaskResult process_data(
        const tensor::TensorInfo& input_data,
        tensor::TensorInfo& output_data
    );

    /**
     * @brief Update pipeline configuration at runtime
     */
    void update_config(const Config& new_config);

    /**
     * @brief Get current performance metrics
     */
    const Metrics& get_metrics() const { return metrics_; }

    /**
     * @brief Reset performance metrics
     */
    void reset_metrics();

private:
    Config config_;
    Metrics metrics_;
    bool is_initialized_ = false;

    std::shared_ptr<memory::MemoryPool> memory_pool_;
    std::shared_ptr<cuda_utils::CudaContext> cuda_context_;
    std::unique_ptr<YourCustomModule> custom_module_;

    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;

    void initialize_cuda_resources();
    void cleanup_cuda_resources();
    void update_performance_metrics();
};

/**
 * @brief Factory for creating custom pipelines
 */
class YourCustomPipelineFactory {
public:
    static std::unique_ptr<YourCustomPipeline> create(
        const YourCustomPipeline::Config& config,
        std::shared_ptr<memory::MemoryPool> memory_pool,
        std::shared_ptr<cuda_utils::CudaContext> cuda_context
    );

    static YourCustomPipeline::Config create_default_config();
};

} // namespace aerial::examples::your_module
```

### Step 5: Implement Pipeline Logic

Create pipeline implementation (`your_custom_pipeline.cpp`):

```cpp
#include "your_custom_pipeline.hpp"
#include <aerial_framework/logging/logger.hpp>
#include <aerial_framework/cuda_utils/cuda_error.hpp>
#include <chrono>

namespace aerial::examples::your_module {

YourCustomPipeline::YourCustomPipeline(
    const Config& config,
    std::shared_ptr<memory::MemoryPool> memory_pool,
    std::shared_ptr<cuda_utils::CudaContext> cuda_context)
    : config_(config),
      memory_pool_(std::move(memory_pool)),
      cuda_context_(std::move(cuda_context)) {
    
    AERIAL_LOG_INFO("Creating custom pipeline with batch size: {}", config_.batch_size);
}

YourCustomPipeline::~YourCustomPipeline() {
    cleanup_cuda_resources();
    AERIAL_LOG_INFO("Custom pipeline destroyed");
}

task::TaskResult YourCustomPipeline::initialize(
    const std::vector<tensor::TensorInfo>& inputs) {
    
    try {
        // Initialize CUDA resources
        initialize_cuda_resources();
        
        // Create custom module
        custom_module_ = YourCustomModuleFactory::create(
            config_.module_config, stream_
        );
        
        // Validate inputs
        if (inputs.empty()) {
            throw std::invalid_argument("Pipeline requires at least one input");
        }
        
        is_initialized_ = true;
        AERIAL_LOG_INFO("Custom pipeline initialized successfully");
        
        return task::TaskResult(task::TaskStatus::Completed);
        
    } catch (const std::exception& e) {
        AERIAL_LOG_ERROR("Failed to initialize custom pipeline: {}", e.what());
        return task::TaskResult(task::TaskStatus::Failed, e.what());
    }
}

task::TaskResult YourCustomPipeline::execute(
    const std::vector<tensor::TensorInfo>& inputs,
    std::vector<tensor::TensorInfo>& outputs,
    const task::CancellationToken& token) {
    
    if (!is_initialized_) {
        return task::TaskResult(task::TaskStatus::Failed, "Pipeline not initialized");
    }

    if (token.is_cancellation_requested()) {
        return task::TaskResult(task::TaskStatus::Cancelled);
    }

    try {
        // Start timing
        CUDA_CHECK(cudaEventRecord(start_event_, stream_));
        
        // Process data
        auto result = process_data(inputs[0], outputs[0]);
        
        // End timing
        CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        
        // Update performance metrics
        update_performance_metrics();
        
        return result;
        
    } catch (const std::exception& e) {
        AERIAL_LOG_ERROR("Custom pipeline execution failed: {}", e.what());
        return task::TaskResult(task::TaskStatus::Failed, e.what());
    }
}

task::TaskResult YourCustomPipeline::process_data(
    const tensor::TensorInfo& input_data,
    tensor::TensorInfo& output_data) {
    
    // Use the custom module to process data
    std::vector<tensor::TensorInfo> inputs = {input_data};
    std::vector<tensor::TensorInfo> outputs = {output_data};
    
    return custom_module_->execute(inputs, outputs);
}

void YourCustomPipeline::initialize_cuda_resources() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
    
    AERIAL_LOG_INFO("CUDA resources initialized for custom pipeline");
}

void YourCustomPipeline::cleanup_cuda_resources() {
    if (start_event_) { cudaEventDestroy(start_event_); start_event_ = nullptr; }
    if (stop_event_) { cudaEventDestroy(stop_event_); stop_event_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

void YourCustomPipeline::finalize() {
    custom_module_.reset();
    cleanup_cuda_resources();
    is_initialized_ = false;
    AERIAL_LOG_INFO("Custom pipeline finalized");
}

void YourCustomPipeline::update_performance_metrics() {
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_));
    
    metrics_.processed_batches++;
    metrics_.total_processing_time_ms += elapsed_ms;
    metrics_.average_latency_ms = metrics_.total_processing_time_ms / metrics_.processed_batches;
    
    if (elapsed_ms > 0.0f) {
        metrics_.throughput_ops_per_sec = (config_.batch_size / elapsed_ms) * 1000.0f;
    }
}

void YourCustomPipeline::reset_metrics() {
    metrics_ = Metrics{};
}

void YourCustomPipeline::update_config(const Config& new_config) {
    config_ = new_config;
    if (custom_module_) {
        custom_module_->update_config(config_.module_config);
    }
}

// Factory implementation
std::unique_ptr<YourCustomPipeline> YourCustomPipelineFactory::create(
    const YourCustomPipeline::Config& config,
    std::shared_ptr<memory::MemoryPool> memory_pool,
    std::shared_ptr<cuda_utils::CudaContext> cuda_context) {
    
    return std::make_unique<YourCustomPipeline>(
        config, std::move(memory_pool), std::move(cuda_context)
    );
}

YourCustomPipeline::Config YourCustomPipelineFactory::create_default_config() {
    YourCustomPipeline::Config config;
    config.module_config = YourCustomModuleFactory::create_default_config();
    config.batch_size = 32;
    config.enable_batching = true;
    config.processing_mode = "standard";
    return config;
}

} // namespace aerial::examples::your_module
```

## Build System Integration

### Step 6: Create CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)

# Your Custom Module Pipeline
add_library(your_custom_module_pipeline STATIC
    your_custom_module.cu
    your_custom_pipeline.cpp
)

target_include_directories(your_custom_module_pipeline
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${AERIAL_FRAMEWORK_INCLUDE_DIRS}
)

target_link_libraries(your_custom_module_pipeline
    PRIVATE
        ${AERIAL_FRAMEWORK_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}  # If using cuBLAS
        ${CUDA_CURAND_LIBRARIES}  # If using cuRAND
        ${CUDA_CUFFT_LIBRARIES}   # If using cuFFT
)

target_compile_features(your_custom_module_pipeline PRIVATE cxx_std_20)

# Set CUDA properties
set_property(TARGET your_custom_module_pipeline PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
set_property(TARGET your_custom_module_pipeline PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# CUDA compilation flags
target_compile_options(your_custom_module_pipeline PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        -Xptxas -v
        --use_fast_math
        -lineinfo
    >
)

# Simple Example
add_executable(your_custom_example
    your_custom_example.cpp
)

target_link_libraries(your_custom_example
    PRIVATE your_custom_module_pipeline
)

target_compile_features(your_custom_example PRIVATE cxx_std_20)

# Comprehensive Example
add_executable(your_comprehensive_example
    your_comprehensive_example.cpp
)

target_link_libraries(your_comprehensive_example
    PRIVATE your_custom_module_pipeline
)

target_compile_features(your_comprehensive_example PRIVATE cxx_std_20)

# Installation rules
install(TARGETS your_custom_module_pipeline your_custom_example your_comprehensive_example
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES your_custom_module.hpp your_custom_pipeline.hpp
    DESTINATION include/your_custom_module
)

# Add custom build targets
add_custom_target(your_custom_benchmark
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/your_comprehensive_example --benchmark
    DEPENDS your_comprehensive_example
    COMMENT "Running custom module benchmarks"
)

add_custom_target(your_custom_test
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/your_custom_example --test
    DEPENDS your_custom_example
    COMMENT "Running custom module tests"
)
```

### Step 7: Update Main CMakeLists.txt

Add your module to the main project CMakeLists.txt:

```cmake
# Example modules
add_subdirectory(channel_estimation)
add_subdirectory(modulation_mapping)
add_subdirectory(fft_processing)
add_subdirectory(mimo_detection)
add_subdirectory(neural_beamforming)
add_subdirectory(your_custom_module)    # Add your module here
```

## Task Integration

### Advanced Task Integration

For complex workflows, you can integrate with the Aerial task system:

```cpp
#include <aerial_framework/task/task_scheduler.hpp>
#include <aerial_framework/task/task_graph.hpp>

class YourAdvancedPipeline {
public:
    task::TaskResult execute_with_task_graph(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs) {
        
        // Create task graph for complex workflows
        auto task_graph = task::TaskGraph::create();
        
        // Add preprocessing task
        auto preprocess_task = task_graph->add_task(
            "preprocess",
            [this](const auto& inputs, auto& outputs) {
                return this->preprocess_data(inputs[0], outputs[0]);
            }
        );
        
        // Add main processing task
        auto process_task = task_graph->add_task(
            "main_process",
            [this](const auto& inputs, auto& outputs) {
                return this->process_data(inputs[0], outputs[0]);
            }
        );
        
        // Add postprocessing task
        auto postprocess_task = task_graph->add_task(
            "postprocess",
            [this](const auto& inputs, auto& outputs) {
                return this->postprocess_data(inputs[0], outputs[0]);
            }
        );
        
        // Define dependencies
        task_graph->add_dependency(preprocess_task, process_task);
        task_graph->add_dependency(process_task, postprocess_task);
        
        // Execute task graph
        return task_graph->execute(inputs, outputs);
    }
};
```

## Testing and Validation

### Step 8: Create Simple Example

Create `your_custom_example.cpp`:

```cpp
#include <iostream>
#include <memory>
#include <vector>

#include "your_custom_pipeline.hpp"
#include <aerial_framework/memory/memory_pool_factory.hpp>
#include <aerial_framework/cuda_utils/cuda_context.hpp>
#include <aerial_framework/tensor/tensor_factory.hpp>

using namespace aerial::examples::your_module;

class YourCustomExample {
public:
    void run() {
        std::cout << "=== Custom Module Pipeline Example ===\n";
        
        try {
            initialize_framework();
            create_pipeline();
            run_processing_demo();
            cleanup();
            
            std::cout << "=== Example completed successfully! ===\n";
            
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

private:
    std::shared_ptr<memory::MemoryPool> memory_pool_;
    std::shared_ptr<cuda_utils::CudaContext> cuda_context_;
    std::unique_ptr<YourCustomPipeline> pipeline_;

    void initialize_framework() {
        std::cout << "1. Initializing Aerial Framework...\n";
        
        memory_pool_ = memory::MemoryPoolFactory::create_cuda_pool(256 * 1024 * 1024);
        cuda_context_ = std::make_shared<cuda_utils::CudaContext>(0);
        
        std::cout << "   ✓ Framework initialized\n";
    }

    void create_pipeline() {
        std::cout << "2. Creating Custom Pipeline...\n";
        
        auto config = YourCustomPipelineFactory::create_default_config();
        config.batch_size = 16;
        
        pipeline_ = YourCustomPipelineFactory::create(
            config, memory_pool_, cuda_context_
        );
        
        std::cout << "   ✓ Pipeline created\n";
    }

    void run_processing_demo() {
        std::cout << "3. Running Processing Demo...\n";
        
        // Create input data
        auto input_data = create_input_tensor();
        auto output_data = create_output_tensor();
        
        // Initialize pipeline
        std::vector<tensor::TensorInfo> inputs = {input_data};
        auto init_result = pipeline_->initialize(inputs);
        
        if (init_result.status != task::TaskStatus::Completed) {
            throw std::runtime_error("Pipeline initialization failed");
        }
        
        // Execute processing
        std::vector<tensor::TensorInfo> outputs = {output_data};
        auto result = pipeline_->execute(inputs, outputs);
        
        if (result.status == task::TaskStatus::Completed) {
            std::cout << "   ✓ Processing completed successfully\n";
            display_results();
        } else {
            throw std::runtime_error("Processing failed: " + result.error_message);
        }
    }

    tensor::TensorInfo create_input_tensor() {
        std::vector<std::size_t> shape = {16, 64};  // Example shape
        return tensor::TensorFactory::create_tensor(
            shape, tensor::DataType::Float32, memory_pool_
        );
    }

    tensor::TensorInfo create_output_tensor() {
        std::vector<std::size_t> shape = {16, 64};  // Example shape
        return tensor::TensorFactory::create_tensor(
            shape, tensor::DataType::Float32, memory_pool_
        );
    }

    void display_results() {
        const auto& metrics = pipeline_->get_metrics();
        std::cout << "   Performance Metrics:\n";
        std::cout << "   - Average latency: " << metrics.average_latency_ms << " ms\n";
        std::cout << "   - Throughput: " << metrics.throughput_ops_per_sec << " ops/sec\n";
    }

    void cleanup() {
        std::cout << "4. Cleaning up...\n";
        
        if (pipeline_) {
            pipeline_->finalize();
            pipeline_.reset();
        }
        
        std::cout << "   ✓ Cleanup completed\n";
    }
};

int main() {
    YourCustomExample example;
    example.run();
    return 0;
}
```

## Documentation Guidelines

### Step 9: Create README.md

Create comprehensive documentation for your module:

```markdown
# Your Custom Module

Brief description of what your module does and its purpose in the Aerial framework.

## Overview

- **Purpose**: [Describe the main functionality]
- **Algorithms**: [List supported algorithms]
- **Performance**: [Key performance characteristics]
- **Use Cases**: [When to use this module]

## Features

### Core Capabilities
- Feature 1 with GPU acceleration
- Feature 2 with optimized memory usage
- Feature 3 with real-time processing

### Performance Optimizations
- CUDA kernel optimizations
- Memory coalescing
- Batch processing support

## Usage Examples

### Basic Usage
```cpp
// Example code showing basic usage
auto config = YourCustomModuleFactory::create_default_config();
auto module = YourCustomModuleFactory::create(config);

// Process data
module->process_algorithm(input_data, output_data);
```

### Advanced Configuration
```cpp
// Example showing advanced configuration
YourCustomModule::Config config;
config.parameter1 = 128;
config.parameter2 = 16;
config.threshold = 0.8f;
config.enable_optimization = true;
```

## API Reference

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| parameter1 | size_t | 64 | Description of parameter1 |
| parameter2 | size_t | 8 | Description of parameter2 |
| threshold | float | 0.5 | Description of threshold |

### Performance Characteristics

| Metric | Value | Conditions |
|--------|-------|------------|
| Latency | X.X ms | Batch size Y |
| Throughput | X ops/sec | GPU model Z |
| Memory Usage | X MB | Configuration ABC |

## Build and Test

```bash
# Build the module
make your_custom_example

# Run simple example
./your_custom_example

# Run comprehensive tests
./your_comprehensive_example --benchmark
```

## Integration Guide

To integrate this module into your own project:

1. Include headers: `#include "your_custom_pipeline.hpp"`
2. Link libraries: `your_custom_module_pipeline`
3. Follow usage examples above

## Contributing

When extending this module:
- Follow existing coding patterns
- Add comprehensive tests
- Update documentation
- Ensure performance benchmarks pass
```

## Best Practices

### Performance Optimization

1. **Memory Management**
   - Use framework memory pools for tensor allocation
   - Minimize CPU-GPU memory transfers
   - Implement proper memory alignment

2. **CUDA Optimization**
   - Optimize block/grid dimensions for your workload
   - Use shared memory when beneficial
   - Implement proper error handling

3. **Pipeline Design**
   - Keep modules stateless when possible
   - Implement proper input/output validation
   - Use factory patterns for flexible configuration

### Error Handling

```cpp
// Always use proper error handling
task::TaskResult process_with_error_handling() {
    try {
        // Your processing logic
        
        return task::TaskResult(task::TaskStatus::Completed);
        
    } catch (const cuda_utils::CudaException& e) {
        AERIAL_LOG_ERROR("CUDA error: {}", e.what());
        return task::TaskResult(task::TaskStatus::Failed, e.what());
    } catch (const std::exception& e) {
        AERIAL_LOG_ERROR("Processing error: {}", e.what());
        return task::TaskResult(task::TaskStatus::Failed, e.what());
    }
}
```

### Testing Guidelines

1. **Unit Tests**: Test individual module functionality
2. **Integration Tests**: Test pipeline orchestration
3. **Performance Tests**: Validate latency and throughput requirements
4. **Stress Tests**: Test with maximum expected loads

### Documentation Standards

1. **Code Documentation**: Use Doxygen-style comments
2. **API Documentation**: Document all public interfaces
3. **Usage Examples**: Provide working code examples
4. **Performance Data**: Include benchmark results

## Next Steps

After implementing your custom module:

1. **Validation**: Run all tests and benchmarks
2. **Integration**: Add to main project build system
3. **Documentation**: Create comprehensive README
4. **Examples**: Create simple and comprehensive examples
5. **Performance**: Profile and optimize critical paths

This template provides a complete foundation for creating new Aerial framework modules while following established patterns and best practices. Modify the specifics based on your particular use case and requirements.