# Getting Started with NVIDIA Aerial Framework Examples

## Overview

This guide will walk you through setting up your development environment and running your first Aerial framework example. We'll start with the channel estimation example, which demonstrates core framework concepts and GPU acceleration patterns.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with compute capability 7.0+ (RTX 20 series or newer)
- 8GB+ GPU memory recommended
- 16GB+ system RAM

### Software Requirements
- Ubuntu 20.04+ or CentOS 8+
- CUDA Toolkit 11.8+
- CMake 3.20+
- GCC 9+ or Clang 12+
- Git

## Installation Steps

### 1. Install CUDA Toolkit

```bash
# Download and install CUDA 12.3 (adjust version as needed)
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
sudo sh cuda_12.3.2_545.23.08_linux.run

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
```

### 2. Install Aerial Framework

```bash
# Clone the Aerial framework (adjust URL/path as provided)
git clone <aerial-framework-repo-url>
cd aerial-framework

# Build framework
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Set environment variable
export AERIAL_FRAMEWORK_ROOT=$(pwd)/..
```

### 3. Clone Examples Repository

```bash
git clone https://github.com/nvidia/nvidia-aerial-examples.git
cd nvidia-aerial-examples
```

## Building Your First Example

### 1. Configure Build

```bash
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86" \
  -DAERIAL_FRAMEWORK_ROOT=${AERIAL_FRAMEWORK_ROOT}
```

### 2. Build Examples

```bash
# Build specific example
make channel_estimation_example -j$(nproc)

# Build all pipeline examples
make -j$(nproc)
```

### 3. Run the Examples

```bash
# Channel estimation example
./channel_estimation/channel_estimation_example

# Modulation pipeline examples
./modulation_mapping/modulation_example                    # Simple example
./modulation_mapping/modulation_mapping_example           # Comprehensive example

# FFT processing examples
./fft_processing/fft_example                              # Simple example
./fft_processing/fft_processing_example                   # Comprehensive example

# MIMO detection examples
./mimo_detection/mimo_example                             # Simple example  
./mimo_detection/mimo_detection_example                   # Comprehensive example

# Neural beamforming examples
./neural_beamforming/neural_beamforming_example           # Simple ML integration
./neural_beamforming/neural_beamforming_ml_example        # Complete ML pipeline

# With custom parameters
./channel_estimation/channel_estimation_example \
  --num-rbs 100 \
  --algorithm mmse \
  --iterations 1000

./modulation_mapping/modulation_mapping_example \
  --modulation 256QAM \
  --batch-size 1024 \
  --config high-performance

./fft_processing/fft_processing_example \
  --fft-size 4096 \
  --batch-size 128 \
  --ofdm-mode

./mimo_detection/mimo_detection_example \
  --tx-antennas 8 \
  --rx-antennas 8 \
  --detector mmse \
  --benchmark

./neural_beamforming/neural_beamforming_ml_example \
  --antennas 64 \
  --users 8 \
  --mode neural-dnn \
  --precision fp16 \
  --comprehensive
```

## Understanding the Output

The examples will output:
- Configuration parameters
- Performance metrics (throughput, latency)
- Sample processing results
- Validation and accuracy metrics
- Benchmark comparisons

Expected output:
```
=== Channel Estimation Pipeline Example ===
Configuration:
  Resource Blocks: 100
  OFDM Symbols: 14
  Algorithm: MMSE
  Iterations: 1000

Performing warm-up execution...
Running performance benchmark...
  Completed 100/1000 iterations
  ...

Results validation: PASSED
Sample channel estimates:
  H[0] = 0.85 + 0.23j
  H[1] = 0.92 - 0.15j
  ...

=== Pipeline Statistics ===
Total executions: 1000
Success rate: 100%
Average execution time: 85 μs

=== Performance Results ===
Pipeline Execution (Stream):
  Average: 85.2 μs
  Throughput: 11,737 executions/second
```

## Common Build Issues

### CUDA Not Found
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Set CUDA paths explicitly
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### Aerial Framework Not Found
```bash
# Verify framework path
ls $AERIAL_FRAMEWORK_ROOT/framework

# Build framework if needed
cd $AERIAL_FRAMEWORK_ROOT
mkdir build && cd build
cmake .. && make -j$(nproc)
```

### Compilation Errors
```bash
# Check compiler version
gcc --version  # Should be 9+

# Install required packages (Ubuntu)
sudo apt update
sudo apt install build-essential cmake git

# Install required packages (CentOS)
sudo yum groupinstall "Development Tools"
sudo yum install cmake3 git
```

## Next Steps

### Explore More Examples

1. **Modulation Mapping Pipeline**: GPU-accelerated QAM constellation mapping
   ```bash
   ./modulation_mapping/modulation_example
   ```
   - Demonstrates QPSK, 16QAM, 64QAM, 256QAM processing
   - Shows batch processing and performance optimization
   - Includes factory pattern usage

2. **FFT Processing Pipeline**: cuFFT-based OFDM processing  
   ```bash
   ./fft_processing/fft_example
   ```
   - Forward and inverse FFT operations
   - OFDM symbol processing with cyclic prefix
   - Mixed batch size processing

3. **MIMO Detection Pipeline**: Multi-antenna signal detection
   ```bash
   ./mimo_detection/mimo_example
   ```
   - Zero Forcing, MMSE, and Maximum Likelihood algorithms
   - Real-time streaming demonstrations  
   - SNR performance analysis
   ```bash
   ./fft_processing/fft_example --size 2048 --batch 16
   ```

### Modify Examples

1. **Change Parameters**: Edit the configuration in `example.cpp`
2. **Add Algorithms**: Implement new detection/estimation methods
3. **Optimize Performance**: Experiment with CUDA graphs, memory pools

### Create Your Own Module

1. **Copy Template**: Start with the channel estimation example
2. **Implement Interface**: Derive from `IModule` and implement required methods
3. **Add GPU Kernels**: Implement CUDA kernels for your algorithm
4. **Build Pipeline**: Create pipeline class if needed

## Development Environment

### Recommended IDE Setup

**Visual Studio Code**:
```bash
# Install VS Code extensions
code --install-extension ms-vscode.cpptools
code --install-extension nvidia.nsight-vscode-edition
code --install-extension ms-vscode.cmake-tools
```

**CLion**:
- Enable CUDA support in settings
- Configure CMake toolchain
- Set up remote development for GPU servers

### Debugging

**CPU Debugging**:
```bash
# Build debug version
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Run with GDB
gdb ./channel_estimation_example
```

**GPU Debugging**:
```bash
# Use compute-sanitizer for memory errors
compute-sanitizer ./channel_estimation_example

# Use Nsight Compute for profiling
ncu --set full ./channel_estimation_example
```

## Performance Optimization

### GPU Memory Management
- Use framework memory pools for large allocations
- Implement proper RAII patterns for cleanup
- Monitor memory usage with `nvidia-smi`

### CUDA Optimization
- Enable CUDA graphs for repeated workloads
- Use appropriate block/grid dimensions
- Optimize memory access patterns

### Framework Best Practices
- Implement proper error handling with `TaskResult`
- Use cancellation tokens for responsive applications
- Leverage framework's performance monitoring

## Getting Help

- **Documentation**: Check the `docs/` directory for detailed guides
- **Examples**: Study existing implementations for patterns
- **Issues**: Report problems on GitHub issues
- **Community**: Join discussions on GitHub Discussions

## Quick Reference

### Build Commands
```bash
# Clean build
rm -rf build && mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build specific target
make <target_name> -j$(nproc)

# Run tests
make test
```

### Environment Variables
```bash
export AERIAL_FRAMEWORK_ROOT=/path/to/aerial-framework
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

You're now ready to start developing with the NVIDIA Aerial Framework! Begin with the channel estimation example and gradually explore more complex examples as you become familiar with the framework patterns.