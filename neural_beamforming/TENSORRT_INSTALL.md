# TensorRT Installation Guide

## Overview

The neural beamforming example can work with or without TensorRT:
- **With TensorRT**: Full neural network-based beamforming capabilities
- **Without TensorRT**: Uses fallback implementation with stub interfaces

## Option 1: Build with Stub Implementation (Default)

The build system automatically creates a stub TensorRT implementation for development and testing:

```bash
cd neural_beamforming
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

This creates a minimal TensorRT interface that allows the code to compile and run, but neural network inference will use fallback algorithms.

## Option 2: Install Official TensorRT

For production use with full neural network capabilities, install official TensorRT from NVIDIA:

### Download TensorRT

1. **Register and Download**:
   - Visit [NVIDIA TensorRT Downloads](https://developer.nvidia.com/tensorrt)
   - Register for NVIDIA Developer account (free)
   - Download TensorRT 10.x for Linux x86_64

2. **Extract and Install**:
   ```bash
   # Example for TensorRT 10.0.1
   tar -xzf TensorRT-10.0.1.Ubuntu-22.04.x86_64-gnu.cuda-12.4.tar.gz
   sudo mv TensorRT-10.0.1 /opt/tensorrt
   
   # Set environment variables
   export TensorRT_ROOT=/opt/tensorrt
   export LD_LIBRARY_PATH=/opt/tensorrt/lib:$LD_LIBRARY_PATH
   ```

3. **Build with TensorRT**:
   ```bash
   cd neural_beamforming
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DTensorRT_ROOT=/opt/tensorrt ..
   make -j$(nproc)
   ```

### Using Package Manager (Ubuntu/Debian)

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install TensorRT
sudo apt-get install tensorrt tensorrt-dev python3-libnvinfer-dev

# Build
cd neural_beamforming
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Option 3: Disable TensorRT Completely

To build without any TensorRT interfaces:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DTENSORRT_FALLBACK_ONLY=ON ..
```

## Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `DOWNLOAD_TENSORRT` | Auto-create stub implementation | `ON` |
| `TENSORRT_FALLBACK_ONLY` | Skip TensorRT entirely | `OFF` |
| `TensorRT_ROOT` | Path to TensorRT installation | Auto-detect |

## Verification

Check TensorRT status in build output:

```
-- TensorRT: Found system installation
  Version: 10.0.1
  Include: /opt/tensorrt/include
  Libraries: /opt/tensorrt/lib/libnvinfer.so
  Type: Official NVIDIA implementation
```

Or for stub:

```
-- TensorRT: Created stub implementation  
  Version: 10.0.1-stub
  Include: .../build/external/tensorrt/include
  Type: Stub implementation (for development/testing)
```

## Performance Impact

| Implementation | Neural Beamforming | Performance | Use Case |
|----------------|-------------------|-------------|----------|
| Official TensorRT | Full GPU acceleration | Best | Production |
| Stub TensorRT | Fallback algorithms | Good | Development/Testing |
| No TensorRT | Classical algorithms only | Baseline | Minimal setup |