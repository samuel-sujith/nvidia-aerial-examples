
# AI Rx Example

This example demonstrates how to replace the receiver (Rx) side of a wireless pipeline with an AI model, using both Python and C++/CUDA (with TensorRT integration).

## Workflow
1. **Sionna Tx-Rx Pipeline (Python)**
	- Run `sionna_tx_rx_pipeline.py` to generate synthetic Tx/Rx data and save to `rx_data.npz`.
2. **AI Model Training (Python)**
	- Run `train_ai_rx_model.py` to train a neural network to mimic Rx and export to `rx_model.onnx`.
	- Convert the ONNX model to a TensorRT engine if desired (see TensorRT docs).
3. **Aerial Integration (C++/CUDA)**
	- Build and run the C++ example to load the TensorRT engine and perform Rx inference using the AI model, following the structure of `neural_beamforming`.
	- If no model is provided, the pipeline uses a default CUDA kernel for Rx inference.

## Files
- `sionna_tx_rx_pipeline.py`: Sionna-based wireless simulation
- `train_ai_rx_model.py`: AI Rx model training and export
- `ai_rx_example.cpp`, `ai_rx_pipeline.cpp/hpp`, `ai_rx_module.cu/hpp`: C++/CUDA integration with TensorRT support
- `CMakeLists.txt`: Build configuration (TensorRT integration)

## Usage
1. Install Sionna, PyTorch, and dependencies for Python scripts.
2. Run the Python scripts in order:
	```bash
	python3 sionna_tx_rx_pipeline.py
	python3 train_ai_rx_model.py
	```
3. (Optional) Convert the ONNX model to TensorRT engine:
	```bash
	# Example using trtexec
	trtexec --onnx=rx_model.onnx --saveEngine=rx_model.engine
	```
4. Build the C++ example using CMake:
	```bash
	mkdir build && cd build
	cmake ..
	make -j$(nproc)
	```
5. Run the executable:
	```bash
	# With TensorRT engine
	./ai_rx_example/ai_rx_example rx_model.engine
	# Or use default CUDA kernel
	./ai_rx_example/ai_rx_example
	```

## Example Output
```
AI Rx Example: Inference results
Symbol 0: 1
Symbol 1: 1
Symbol 2: 0
Symbol 3: 0
```

See source files for implementation details and further customization.
