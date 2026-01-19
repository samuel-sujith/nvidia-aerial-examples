# AI Rx Example

This example demonstrates replacing the receiver (Rx) side of a wireless pipeline with an AI model.

## Workflow
1. **Sionna Tx-Rx Pipeline (Python)**
	- Run `sionna_tx_rx_pipeline.py` to generate synthetic Tx/Rx data and save to `rx_data.npz`.
2. **AI Model Training (Python)**
	- Run `train_ai_rx_model.py` to train a neural network to mimic Rx and export to `rx_model.onnx`.
3. **Aerial Integration (C++/CUDA)**
	- Build and run the C++ example to load the ONNX model and perform Rx inference using the AI model, following the structure of `neural_beamforming`.

## Files
- `sionna_tx_rx_pipeline.py`: Sionna-based wireless simulation
- `train_ai_rx_model.py`: AI Rx model training and export
- `ai_rx_example.cpp`, `ai_rx_pipeline.cpp/hpp`, `ai_rx_module.cu/hpp`: C++/CUDA integration
- `CMakeLists.txt`: Build configuration

## Usage
1. Install Sionna, PyTorch, and dependencies for Python scripts.
2. Run the Python scripts in order:
	```bash
	python3 sionna_tx_rx_pipeline.py
	python3 train_ai_rx_model.py
	```
3. Build the C++ example using CMake, then run the executable.

See source files for implementation details.
