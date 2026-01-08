## Using Machine Learning (ML) Channel Estimation with TensorRT

This example supports ML-based channel estimation using a neural network model deployed with NVIDIA TensorRT. Below are the steps to train a model and use it in this example.

### 1. Training a Channel Estimation Model

You can use any deep learning framework (e.g., PyTorch, TensorFlow) to train a neural network for channel estimation. A typical workflow is:

1. **Prepare Training Data:**
   - Simulate or collect pairs of (received pilots, true channel response) for your target scenario.
   - Format: `X = received_pilots`, `Y = true_channel_response` (both as real/imag pairs or complex tensors).
2. **Design and Train Model:**
   - Use a suitable architecture (e.g., fully connected, CNN, transformer) to map pilots to channel estimates.
   - Train with MSE or similar loss.
3. **Export to ONNX:**
   - Export the trained model to ONNX format (e.g., `model.onnx`).
4. **Convert to TensorRT Engine:**
   - Use `trtexec` or TensorRT Python API to convert ONNX to TensorRT engine:
     ```bash
     trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
     ```
   - The resulting `model.engine` file is used by this example.

### 2. Using the ML Model in the Channel Estimation Example

1. **Build with TensorRT Support:**
   - Ensure TensorRT is installed and available on your system.
   - Build with `-DTENSORRT_AVAILABLE=ON` or ensure the macro is defined.
2. **Run the Example with ML Estimator:**
   - Pass the model path and select the ML algorithm:
     ```bash
     ./channel_estimation_example --algorithm ml_tensorrt --model_path /path/to/model.engine
     ```
   - Additional parameters (optional):
     - `--ml_input_size`, `--ml_output_size`, `--use_fp16`, `--max_batch_size`
   - The pipeline will use the ML estimator if TensorRT is available and a valid model is provided. Otherwise, it falls back to classic algorithms.

#### Example Command
```bash
./channel_estimation_example --algorithm ml_tensorrt --model_path ./my_model.engine --ml_input_size 128 --ml_output_size 256 --use_fp16 1 --max_batch_size 32
```

### 3. Notes
- The ML estimator expects the input/output tensor shapes to match those used during training and engine export.
- For best results, use the same pilot pattern and channel conditions in both training and deployment.
- See the main README sections above for parameter details and troubleshooting.

---
