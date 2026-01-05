#!/usr/bin/env python3
"""
ML Channel Estimation Model Generator
====================================

This script generates a neural network model for channel estimation.
The model learns to estimate channel coefficients from pilot signals and noisy observations.

Requirements:
    - torch >= 1.12.0
    - numpy >= 1.20.0
    - tensorrt >= 8.5.0 (optional, for deployment)
    - onnx >= 1.12.0

Usage:
    python generate_channel_estimation_model.py --antennas 64 --users 4 --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import json
import time
from typing import Tuple, Optional
from pathlib import Path

class ChannelEstimationNet(nn.Module):
    """
    Neural network for channel estimation.
    
    Architecture:
    - Input: Received pilot signals (real/imag parts) + noise level
    - Hidden: Multiple fully connected layers with residual connections
    - Output: Estimated channel coefficients (real/imag parts)
    """
    def __init__(self, num_antennas: int, num_users: int, hidden_dims: list = None):
        super(ChannelEstimationNet, self).__init__()
        self.num_antennas = num_antennas
        self.num_users = num_users
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 256, 512]
        # Input: received pilots (real + imag) + noise level
        input_dim = num_users * num_antennas * 2 + 1
        # Output: estimated channel (real + imag)
        output_dim = num_users * num_antennas * 2
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        self.residual1 = nn.Linear(input_dim, hidden_dims[2]) if len(hidden_dims) > 2 else None
        self.residual2 = nn.Linear(hidden_dims[2], output_dim) if len(hidden_dims) > 2 else None
    def forward(self, x):
        out = self.network(x)
        if self.residual1 is not None and self.residual2 is not None:
            residual = self.residual2(torch.relu(self.residual1(x)))
            out = out + 0.1 * residual
        return out

def generate_synthetic_data(num_antennas: int, num_users: int, num_samples: int, snr_range: Tuple[float, float] = (0, 30)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic pilot and channel data for training.
    Returns (inputs, targets):
      - inputs: [num_samples, num_users * num_antennas * 2 + 1] (real/imag pilots + noise)
      - targets: [num_samples, num_users * num_antennas * 2] (real/imag channel)
    """
    pilots = (torch.randn(num_samples, num_users, num_antennas, dtype=torch.complex64) +
              1j * torch.randn(num_samples, num_users, num_antennas, dtype=torch.complex64)) / np.sqrt(2)
    channels = (torch.randn(num_samples, num_users, num_antennas, dtype=torch.complex64) +
                1j * torch.randn(num_samples, num_users, num_antennas, dtype=torch.complex64)) / np.sqrt(2)
    snrs = np.random.uniform(snr_range[0], snr_range[1], size=(num_samples, 1)).astype(np.float32)
    noise_vars = 10 ** (-snrs / 10)
    noise = (torch.randn_like(pilots) + 1j * torch.randn_like(pilots)) * torch.tensor(noise_vars).reshape(-1, 1, 1) / np.sqrt(2)
    received = pilots * channels + noise
    inputs = torch.cat([received.real, received.imag], dim=-1).reshape(num_samples, -1)
    inputs = torch.cat([inputs, torch.tensor(noise_vars)], dim=1)
    targets = torch.cat([channels.real, channels.imag], dim=-1).reshape(num_samples, -1)
    return inputs, targets

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ChannelEstimationNet(args.antennas, args.users).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    inputs, targets = generate_synthetic_data(args.antennas, args.users, args.samples, (args.snr_min, args.snr_max))
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    net.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(dataset):.6f}")
    torch.save(net.state_dict(), args.out_prefix + '.pth')
    print(f"Saved PyTorch model to {args.out_prefix}.pth")
    # Export to ONNX
    dummy = torch.randn(1, args.users * args.antennas * 2 + 1, device=device)
    torch.onnx.export(net, dummy, args.out_prefix + '.onnx', input_names=['input'], output_names=['output'], opset_version=16)
    print(f"Exported ONNX model to {args.out_prefix}.onnx")
    # Optionally, export to TensorRT (if trtexec is available)
    trt_path = args.out_prefix + '.trt'
    os.system(f"trtexec --onnx={args.out_prefix}.onnx --saveEngine={trt_path} --fp16")
    if os.path.exists(trt_path):
        print(f"Exported TensorRT engine to {trt_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a neural network for channel estimation.")
    parser.add_argument('--antennas', type=int, default=64)
    parser.add_argument('--users', type=int, default=4)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--snr-min', type=float, default=0.0)
    parser.add_argument('--snr-max', type=float, default=30.0)
    parser.add_argument('--out-prefix', type=str, default='channel_estimation_model')
    args = parser.parse_args()
    train_model(args)

if __name__ == '__main__':
    main()
