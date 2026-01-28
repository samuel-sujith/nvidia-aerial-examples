# User Scheduling Example

This example demonstrates ML-assisted user scheduling in a simplified multi-UE scenario using NVIDIA Aerial Framework module/pipeline patterns. It includes scripts for synthetic data generation, model training, and a C++ example that loads the trained model to select UEs.

## Workflow
1. **Generate synthetic data**
   ```bash
   python3 generate_user_scheduling_data.py --num-ttis 500 --num-ues 32 --num-scheduled 8
   ```
2. **Train a scheduling model**
   ```bash
   python3 train_user_scheduling_model.py --data user_scheduling_data.npz --output user_scheduling_model.txt
   ```
3. **(Optional) Export ONNX and build TensorRT engine**
   ```bash
   python3 train_user_scheduling_model.py --data user_scheduling_data.npz \
       --output user_scheduling_model.txt --export-onnx user_scheduling_model.onnx

   trtexec --onnx=user_scheduling_model.onnx --saveEngine=user_scheduling_model.engine
   ```
4. **Run the C++ example**
   ```bash
   ./user_scheduling_example --model user_scheduling_model.txt --num-ues 16 --num-scheduled 4
   ./user_scheduling_example --model user_scheduling_model.engine --num-ues 16 --num-scheduled 4
   ```

## Files
- `generate_user_scheduling_data.py`: Synthetic data generator for UE scheduling features and labels.
- `train_user_scheduling_model.py`: Trains a lightweight logistic regression model and exports a text model.
- `user_scheduling_module.hpp/.cu`: Aerial framework module for scoring UEs on GPU (TensorRT optional).
- `user_scheduling_pipeline.hpp/.cpp`: Pipeline wrapper that feeds data to the module and selects UEs.
- `user_scheduling_example.cpp`: Uses the pipeline to perform scheduling decisions.
- `CMakeLists.txt`: Build configuration for the example.

## Notes
- The model file is a plain-text export designed for easy loading in C++.
- The TensorRT engine is built for the `num_ues` used during data generation.
- If no model is provided, the example falls back to a heuristic scheduling score.
