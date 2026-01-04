# Aerial Framework API Reference

## Core Interfaces

### pipeline::IModule

The base interface for all processing modules in the Aerial framework.

```cpp
class IModule {
public:
    virtual ~IModule() = default;
    
    /// Get unique identifier for this module
    virtual std::string_view get_module_id() const = 0;
    
    /// Execute module with input/output tensors
    virtual task::TaskResult execute(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs,
        const task::CancellationToken& token
    ) = 0;
    
    /// Check if input is ready for processing
    virtual bool is_input_ready(std::size_t input_index) const = 0;
    
    /// Check if output buffer is ready
    virtual bool is_output_ready(std::size_t output_index) const = 0;
};
```

**Usage Example:**
```cpp
class MyModule final : public pipeline::IModule {
public:
    std::string_view get_module_id() const override {
        return "my_module_v1";
    }
    
    task::TaskResult execute(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs,
        const task::CancellationToken& token
    ) override {
        // Implementation here
        return task::TaskResult(task::TaskStatus::Completed);
    }
};
```

### pipeline::IPipeline

Interface for pipeline orchestration and multi-module workflows.

```cpp
class IPipeline {
public:
    virtual ~IPipeline() = default;
    
    /// Get pipeline identifier
    virtual std::string_view get_pipeline_id() const = 0;
    
    /// Get number of external inputs
    virtual std::size_t get_num_external_inputs() const = 0;
    
    /// Get number of external outputs  
    virtual std::size_t get_num_external_outputs() const = 0;
    
    /// Execute pipeline with external data
    virtual task::TaskResult execute_pipeline(
        std::span<const tensor::TensorInfo> inputs,
        std::span<tensor::TensorInfo> outputs,
        const task::CancellationToken& token
    ) = 0;
    
    /// Execute with CUDA graph optimization
    virtual task::TaskResult execute_pipeline_graph(
        std::span<const tensor::TensorInfo> inputs,
        std::span<tensor::TensorInfo> outputs,
        const task::CancellationToken& token
    ) = 0;
    
    /// Setup pipeline resources
    virtual bool setup(const PipelineSpec& spec) = 0;
    
    /// Cleanup pipeline resources
    virtual void teardown() = 0;
    
    /// Check if pipeline is ready
    virtual bool is_ready() const = 0;
    
    /// Get performance statistics
    virtual PipelineStats get_stats() const = 0;
};
```

### Factory Interfaces

#### pipeline::IModuleFactory

Factory interface for creating modules.

```cpp
class IModuleFactory {
public:
    virtual ~IModuleFactory() = default;
    
    /// Create module instance
    virtual std::unique_ptr<IModule> create_module(
        const std::string& module_id,
        const ModuleSpec& spec
    ) = 0;
    
    /// Get supported module types
    virtual std::vector<std::string> get_supported_types() const = 0;
};
```

## Task System

### task::TaskResult

Result type for all framework operations.

```cpp
struct TaskResult {
    TaskStatus status{TaskStatus::Completed};
    std::string message;
    
    /// Default successful result
    TaskResult() = default;
    
    /// Create result with status and message
    TaskResult(TaskStatus s, std::string_view msg = "");
    
    /// Check if operation succeeded
    bool is_success() const noexcept;
};

enum class TaskStatus {
    NotStarted,  ///< Task not started
    Running,     ///< Task executing
    Completed,   ///< Task completed successfully
    Failed,      ///< Task failed
    Cancelled    ///< Task was cancelled
};
```

### task::CancellationToken

Cooperative cancellation mechanism.

```cpp
class CancellationToken {
public:
    /// Check if cancellation was requested
    bool is_cancellation_requested() const noexcept;
    
    /// Request cancellation
    void request_cancellation() noexcept;
    
    /// Reset cancellation state
    void reset() noexcept;
};
```

## Tensor System

### tensor::TensorInfo

Core tensor abstraction for data handling.

```cpp
class TensorInfo {
public:
    /// Set tensor data pointer
    void set_data(void* data);
    
    /// Get tensor data pointer
    void* data() const;
    
    /// Set tensor dimensions
    void set_dimensions(const std::vector<std::size_t>& dims);
    
    /// Get tensor dimensions
    const std::vector<std::size_t>& dimensions() const;
    
    /// Set element data type
    void set_element_type(ElementType type);
    
    /// Get element data type
    ElementType element_type() const;
    
    /// Get total size in bytes
    std::size_t size_bytes() const;
    
    /// Get number of elements
    std::size_t num_elements() const;
};

enum class ElementType {
    FLOAT32,
    FLOAT16, 
    INT32,
    INT16,
    INT8,
    UINT32,
    UINT16,
    UINT8,
    COMPLEX_FLOAT32,
    COMPLEX_FLOAT16
};
```

## Memory Management

### memory::MemoryPool

Framework memory pool for efficient allocation.

```cpp
class MemoryPool {
public:
    /// Create memory pool with specified size
    explicit MemoryPool(std::size_t pool_size);
    
    /// Allocate memory from pool
    void* allocate(std::size_t size, std::size_t alignment = 256);
    
    /// Deallocate memory back to pool
    void deallocate(void* ptr);
    
    /// Get total pool size
    std::size_t total_size() const;
    
    /// Get available memory
    std::size_t available_size() const;
    
    /// Reset pool (deallocate all)
    void reset();
};
```

## Configuration Types

### pipeline::PipelineSpec

Pipeline configuration specification.

```cpp
struct PipelineSpec {
    std::string pipeline_type;
    std::vector<ModuleSpec> modules;
    std::map<std::string, std::any> parameters;
    
    /// Memory pool configuration
    struct MemoryConfig {
        std::size_t pool_size{1024 * 1024 * 1024}; // 1GB default
        std::size_t alignment{256};
        bool use_pinned_memory{false};
    } memory_config;
    
    /// Performance configuration
    struct PerformanceConfig {
        bool enable_cuda_graphs{true};
        int max_batch_size{64};
        int num_streams{1};
        bool enable_profiling{false};
    } performance_config;
};
```

### pipeline::ModuleSpec

Module configuration specification.

```cpp
struct ModuleSpec {
    std::string module_type;
    std::string module_id;
    std::map<std::string, std::any> parameters;
    
    /// Input/output specifications
    std::vector<TensorSpec> input_specs;
    std::vector<TensorSpec> output_specs;
};

struct TensorSpec {
    std::vector<std::size_t> dimensions;
    ElementType element_type;
    std::string name;
    bool optional{false};
};
```

## Performance Monitoring

### pipeline::PipelineStats

Performance statistics collection.

```cpp
struct PipelineStats {
    std::string pipeline_id;
    
    uint64_t total_executions{0};
    uint64_t successful_executions{0}; 
    uint64_t failed_executions{0};
    
    uint64_t total_execution_time_us{0};
    uint64_t last_execution_time_us{0};
    uint64_t min_execution_time_us{UINT64_MAX};
    uint64_t max_execution_time_us{0};
    
    /// Calculate average execution time
    double average_execution_time_us() const;
    
    /// Calculate throughput (executions/second)
    double throughput_per_second() const;
    
    /// Calculate success rate (0.0 to 1.0)
    double success_rate() const;
};
```

## CUDA Integration Utilities

### Kernel Launch Helpers

```cpp
namespace cuda_utils {

/// Calculate optimal block size for kernel
template<typename KernelFunc>
cudaError_t get_optimal_block_size(
    KernelFunc kernel,
    int* min_grid_size,
    int* block_size,
    std::size_t dynamic_shared_mem = 0
);

/// Launch kernel with optimal configuration
template<typename KernelFunc, typename... Args>
cudaError_t launch_kernel_optimal(
    KernelFunc kernel,
    std::size_t num_elements,
    cudaStream_t stream,
    Args... args
);

/// CUDA graph helper
class CudaGraphHelper {
public:
    /// Begin graph capture
    cudaError_t begin_capture(cudaStream_t stream);
    
    /// End capture and create executable
    cudaError_t end_capture_and_instantiate();
    
    /// Launch captured graph
    cudaError_t launch(cudaStream_t stream);
    
    /// Cleanup resources
    void destroy();
};

} // namespace cuda_utils
```

## Error Handling

### Common Error Codes

```cpp
enum class FrameworkError {
    Success = 0,
    InvalidArgument,
    OutOfMemory,
    CudaError,
    InvalidTensorDimensions,
    ModuleNotFound,
    PipelineNotReady,
    ExecutionFailed,
    Cancelled
};

/// Convert CUDA error to framework error
FrameworkError cuda_to_framework_error(cudaError_t cuda_error);

/// Get error message string
const char* get_error_string(FrameworkError error);
```

### Exception Types

```cpp
/// Base framework exception
class FrameworkException : public std::exception {
public:
    FrameworkException(FrameworkError error, std::string_view message);
    const char* what() const noexcept override;
    FrameworkError error_code() const noexcept;
};

/// CUDA-specific exception
class CudaException : public FrameworkException {
public:
    CudaException(cudaError_t cuda_error, std::string_view context);
    cudaError_t cuda_error() const noexcept;
};
```

## Logging and Debugging

### Logging Interface

```cpp
namespace log {

enum class LogLevel {
    Trace = 0,
    Debug,
    Info, 
    Warning,
    Error,
    Critical
};

/// Log message with specified level
void log(LogLevel level, std::string_view message);

/// Convenience macros
#define AERIAL_LOG_DEBUG(msg) log(LogLevel::Debug, msg)
#define AERIAL_LOG_INFO(msg) log(LogLevel::Info, msg)
#define AERIAL_LOG_WARNING(msg) log(LogLevel::Warning, msg)
#define AERIAL_LOG_ERROR(msg) log(LogLevel::Error, msg)

} // namespace log
```

### Profiling Utilities

```cpp
namespace profiling {

/// NVTX range for detailed profiling
class NvtxRange {
public:
    explicit NvtxRange(const char* name);
    ~NvtxRange();
};

/// RAII profiling scope
#define AERIAL_PROFILE_SCOPE(name) profiling::NvtxRange _nvtx_range(name)

/// Performance timer
class PerformanceTimer {
public:
    void start();
    void stop();
    double elapsed_microseconds() const;
    void reset();
};

} // namespace profiling
```

## Usage Examples

### Complete Module Implementation

```cpp
class ExampleModule final : public pipeline::IModule {
private:
    std::string module_id_;
    ExampleParams params_;
    CudaResources cuda_resources_;
    
public:
    ExampleModule(const std::string& id, const ExampleParams& params)
        : module_id_(id), params_(params) {
        setup_cuda_resources();
    }
    
    ~ExampleModule() {
        cleanup_cuda_resources();
    }
    
    std::string_view get_module_id() const override {
        return module_id_;
    }
    
    task::TaskResult execute(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs,
        const task::CancellationToken& token
    ) override {
        AERIAL_PROFILE_SCOPE("ExampleModule::execute");
        
        if (token.is_cancellation_requested()) {
            return task::TaskResult(task::TaskStatus::Cancelled);
        }
        
        try {
            validate_inputs(inputs);
            setup_kernel_parameters(inputs, outputs);
            
            cudaError_t err = launch_processing_kernel();
            if (err != cudaSuccess) {
                return task::TaskResult(task::TaskStatus::Failed, 
                    cuda_utils::get_error_string(err));
            }
            
            return task::TaskResult(task::TaskStatus::Completed);
            
        } catch (const FrameworkException& e) {
            AERIAL_LOG_ERROR(e.what());
            return task::TaskResult(task::TaskStatus::Failed, e.what());
        }
    }
    
    bool is_input_ready(std::size_t input_index) const override {
        return input_index < expected_num_inputs();
    }
    
    bool is_output_ready(std::size_t output_index) const override {
        return output_index < expected_num_outputs();
    }
};
```

## Pipeline Implementations

### ModulationPipeline

QAM modulation and demodulation pipeline implementation.

```cpp
class ModulationPipeline final : public pipeline::IPipeline {
public:
    enum class ModulationType {
        QPSK = 4,
        QAM16 = 16,
        QAM64 = 64,
        QAM256 = 256
    };
    
    ModulationPipeline(ModulationType type, std::size_t batch_size);
    
    task::TaskResult process_symbols(
        const tensor::TensorInfo& input_bits,
        tensor::TensorInfo& output_symbols
    );
    
    float get_symbol_error_rate() const;
};
```

### FFTPipeline

cuFFT-based FFT/IFFT processing pipeline.

```cpp
class FFTPipeline final : public pipeline::IPipeline {
public:
    FFTPipeline(std::size_t fft_size, std::size_t batch_size);
    
    task::TaskResult process_fft(
        const tensor::TensorInfo& input_data,
        tensor::TensorInfo& output_data,
        bool forward = true
    );
    
    void configure_ofdm_mode(bool enabled);
};
```

### MIMOPipeline

Multi-antenna MIMO detection pipeline.

```cpp
class MIMOPipeline final : public pipeline::IPipeline {
public:
    enum class DetectorType {
        ZeroForcing,
        MMSE,
        MaximumLikelihood
    };
    
    MIMOPipeline(std::size_t tx_antennas, std::size_t rx_antennas);
    
    task::TaskResult detect_symbols(
        const tensor::TensorInfo& received_signal,
        const tensor::TensorInfo& channel_matrix,
        tensor::TensorInfo& detected_symbols,
        DetectorType detector = DetectorType::MMSE
    );
};
```

This API reference provides the complete interface for developing with the NVIDIA Aerial Framework. All examples in this repository follow these patterns and interfaces.