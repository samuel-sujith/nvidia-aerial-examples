"""
Step 2: Train AI model to mimic Rx side
Loads Sionna-generated data and trains a neural network.

Reference: Sionna - A Library for Link-Level Simulations in Wireless Communications
https://nvlabs.github.io/sionna/
"""
import numpy as np
import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim

class RxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train_model():
    # Load Sionna-generated data
    data = np.load('rx_data.npz')
    rx_symbols = data['rx_symbols']
    tx_bits = data['tx_bits']

    # Prepare data (real/imag as features)
    X = np.stack([rx_symbols.real, rx_symbols.imag], axis=1)
    y = tx_bits.reshape(-1, 1)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Model, loss, optimizer
    model = RxNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Export to ONNX
    torch.onnx.export(model, X_tensor[:10], "rx_model.onnx", input_names=['input'], output_names=['output'], opset_version=11)
    print("AI Rx model trained and exported to rx_model.onnx.")

def generate_training_data():
    print("Generating training data using Sionna Tx-Rx pipeline...")
    script_path = os.path.join(os.path.dirname(__file__), "sionna_tx_rx_pipeline.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Sionna Tx-Rx pipeline script not found at {script_path}")

    # Run the Sionna Tx-Rx pipeline script
    result = subprocess.run(["python3", script_path], capture_output=True, text=True)

    if result.returncode != 0:
        print("Error while generating training data:")
        print(result.stderr)
        raise RuntimeError("Failed to generate training data.")

    print("Training data generated successfully.")
    print(result.stdout)

if __name__ == "__main__":
    # Step 1: Generate training data
    generate_training_data()

    # Step 2: Train the model
    train_model()
