"""
Step 2: Train AI model to mimic Rx side
Loads Sionna-generated data and trains a neural network.
"""
import numpy as np
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

if __name__ == "__main__":
    train_model()
