#!/usr/bin/env python3
"""
Neural Beamforming Model Generator
==================================

This script generates a neural network model for beamforming weight optimization.
The model learns optimal beamforming weights from channel conditions and user requirements.

Requirements:
    - torch >= 1.12.0
    - numpy >= 1.20.0
    - tensorrt >= 8.5.0 (optional, for deployment)
    - onnx >= 1.12.0

Usage:
    python generate_beamforming_model.py --antennas 64 --users 4 --epochs 100
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

class BeamformingNet(nn.Module):
    """
    Neural network for beamforming weight optimization.
    
    Architecture:
    - Input: Channel matrix (real/imag parts) + user priorities + interference info
    - Hidden: Multiple fully connected layers with residual connections
    - Output: Beamforming weights (real/imag parts)
    """
    
    def __init__(self, num_antennas: int, num_users: int, hidden_dims: list = None):
        super(BeamformingNet, self).__init__()
        
        self.num_antennas = num_antennas
        self.num_users = num_users
        
        # Default architecture if not specified
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 256, 512]
        
        # Input: channel matrix (real + imag) + interference map + noise estimate
        # Channel: num_users * num_antennas * 2 (real/imag)
        # Interference: num_users
        # Noise: 1
        # User priorities: num_users
        input_dim = num_users * num_antennas * 2 + num_users + 1 + num_users
        
        # Output: beamforming weights (real + imag)
        # Weights: num_users * num_antennas * 2 (real/imag)
        output_dim = num_users * num_antennas * 2
        
        # Build network layers
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
        
        # Output layer with tanh activation for bounded weights
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        # Residual connections for better gradient flow
        self.residual1 = nn.Linear(input_dim, hidden_dims[2]) if len(hidden_dims) > 2 else None
        self.residual2 = nn.Linear(hidden_dims[2], output_dim) if len(hidden_dims) > 2 else None
        
    def forward(self, x):
        # Main path
        out = self.network(x)
        
        # Add residual connections if available
        if self.residual1 is not None and self.residual2 is not None:
            residual = self.residual2(torch.relu(self.residual1(x)))
            out = out + 0.1 * residual  # Scale residual contribution
        
        return out

def generate_synthetic_data(num_antennas: int, num_users: int, num_samples: int, 
                          snr_range: Tuple[float, float] = (0, 30)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic channel data and optimal beamforming weights.
    
    This creates realistic channel conditions and computes reference beamforming
    weights using classical algorithms (MVDR, ZF) as training targets.
    """
    
    # Generate complex channel matrix
    # Shape: [num_samples, num_users, num_antennas]
    channels_complex = (torch.randn(num_samples, num_users, num_antennas, dtype=torch.complex64) + 
                       1j * torch.randn(num_samples, num_users, num_antennas, dtype=torch.complex64)) / np.sqrt(2)
    
    # Add realistic channel effects
    for sample in range(num_samples):
        for user in range(num_users):
            # Add correlated fading (antenna correlation)
            correlation_matrix = torch.eye(num_antennas, dtype=torch.complex64)
            for i in range(num_antennas):
                for j in range(num_antennas):
                    if i != j:
                        correlation_matrix[i, j] = 0.3 * torch.exp(1j * torch.tensor(
                            2 * np.pi * abs(i - j) * 0.5 * np.sin(user * np.pi / 8)
                        ))
            
            # Apply correlation
            channels_complex[sample, user] = torch.matmul(
                torch.cholesky(correlation_matrix), 
                channels_complex[sample, user]
            )
    
    # Convert to real representation for network input
    channels_real = torch.stack([channels_complex.real, channels_complex.imag], dim=-1)
    channels_flat = channels_real.reshape(num_samples, -1)
    
    # Generate interference levels and user priorities
    interference = torch.rand(num_samples, num_users) * 0.5  # Random interference
    user_priorities = torch.ones(num_samples, num_users)  # Equal priority for simplicity
    
    # Generate noise levels based on SNR
    snr_db = torch.rand(num_samples) * (snr_range[1] - snr_range[0]) + snr_range[0]
    noise_power = 10 ** (-snr_db / 10)
    
    # Combine all inputs
    inputs = torch.cat([
        channels_flat,
        interference,
        noise_power.unsqueeze(1),
        user_priorities
    ], dim=1)
    
    # Generate optimal beamforming weights using MVDR algorithm
    optimal_weights = []
    
    for sample in range(num_samples):
        H = channels_complex[sample]  # [num_users, num_antennas]
        sigma2 = noise_power[sample].item()
        
        # Compute covariance matrix with regularization
        R = torch.eye(num_antennas, dtype=torch.complex64) * sigma2
        
        # Add interference from other users
        for user in range(num_users):
            for interferer in range(num_users):
                if user != interferer:
                    h_int = H[interferer].unsqueeze(0)  # [1, num_antennas]
                    R = R + interference[sample, interferer] * torch.matmul(h_int.conj().T, h_int)
        
        # Compute MVDR weights for each user
        weights_sample = torch.zeros(num_users, num_antennas, dtype=torch.complex64)
        
        for user in range(num_users):
            h = H[user].unsqueeze(1)  # [num_antennas, 1]
            
            try:
                # MVDR: w = (R^-1 * h) / (h^H * R^-1 * h)
                R_inv_h = torch.linalg.solve(R, h)
                denominator = torch.matmul(h.conj().T, R_inv_h)
                
                # Avoid division by zero
                if torch.abs(denominator) > 1e-8:
                    w_mvdr = R_inv_h.squeeze() / denominator.squeeze()
                else:
                    # Fallback to MRT
                    w_mvdr = h.squeeze() / torch.norm(h)
                
                weights_sample[user] = w_mvdr
                
            except RuntimeError:
                # Fallback to Maximum Ratio Transmission if MVDR fails
                weights_sample[user] = h.squeeze() / torch.norm(h)
        
        optimal_weights.append(weights_sample)
    
    # Convert weights to real representation
    optimal_weights = torch.stack(optimal_weights)  # [num_samples, num_users, num_antennas]
    weights_real = torch.stack([optimal_weights.real, optimal_weights.imag], dim=-1)
    weights_flat = weights_real.reshape(num_samples, -1)
    
    return inputs.float(), weights_flat.float()

def train_model(model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor], 
                val_data: Tuple[torch.Tensor, torch.Tensor], 
                epochs: int = 100, lr: float = 0.001, device: str = 'cuda') -> dict:
    """Train the beamforming neural network."""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data
    
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)
    
    batch_size = 64
    train_losses = []
    val_losses = []
    
    print(f"Training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training loop
        for i in range(0, len(train_inputs), batch_size):
            batch_inputs = train_inputs[i:i+batch_size]
            batch_targets = train_targets[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_inputs), batch_size):
                batch_inputs = val_inputs[i:i+batch_size]
                batch_targets = val_targets[i:i+batch_size]
                outputs = model(batch_inputs)
                val_loss += criterion(outputs, batch_targets).item()
        
        avg_train_loss = epoch_loss / (len(train_inputs) // batch_size)
        avg_val_loss = val_loss / (len(val_inputs) // batch_size)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }

def export_to_onnx(model: nn.Module, input_shape: Tuple[int, ...], output_path: str, device: str = 'cuda'):
    """Export trained model to ONNX format for TensorRT deployment."""
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape, device=device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['channel_input'],
        output_names=['beamforming_weights'],
        dynamic_axes={
            'channel_input': {0: 'batch_size'},
            'beamforming_weights': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to ONNX: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate Neural Beamforming Model')
    parser.add_argument('--antennas', type=int, default=64, help='Number of antennas')
    parser.add_argument('--users', type=int, default=4, help='Number of users')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./models', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[512, 256, 128, 256, 512],
                       help='Hidden layer dimensions')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training data
    print(f"Generating {args.samples} training samples...")
    print(f"Configuration: {args.antennas} antennas, {args.users} users")
    
    # Split data
    train_samples = int(args.samples * 0.8)
    val_samples = args.samples - train_samples
    
    train_inputs, train_targets = generate_synthetic_data(
        args.antennas, args.users, train_samples
    )
    val_inputs, val_targets = generate_synthetic_data(
        args.antennas, args.users, val_samples
    )
    
    print(f"Training data shape: {train_inputs.shape} -> {train_targets.shape}")
    print(f"Validation data shape: {val_inputs.shape} -> {val_targets.shape}")
    
    # Create and train model
    model = BeamformingNet(args.antennas, args.users, args.hidden_dims)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    start_time = time.time()
    training_stats = train_model(
        model, (train_inputs, train_targets), (val_inputs, val_targets),
        epochs=args.epochs, lr=args.lr, device=device
    )
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final validation loss: {training_stats['final_val_loss']:.6f}")
    
    # Save model
    model_path = output_dir / f"beamforming_model_{args.antennas}x{args.users}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_antennas': args.antennas,
            'num_users': args.users,
            'hidden_dims': args.hidden_dims,
        },
        'training_stats': training_stats,
        'input_shape': train_inputs.shape[1:],
        'output_shape': train_targets.shape[1:],
    }, model_path)
    
    print(f"PyTorch model saved: {model_path}")
    
    # Export to ONNX
    onnx_path = output_dir / f"beamforming_model_{args.antennas}x{args.users}.onnx"
    export_to_onnx(model, train_inputs.shape[1:], str(onnx_path), device)
    
    # Save configuration for C++ integration
    config = {
        'model_info': {
            'num_antennas': args.antennas,
            'num_users': args.users,
            'input_dim': train_inputs.shape[1],
            'output_dim': train_targets.shape[1],
            'hidden_dims': args.hidden_dims,
        },
        'performance': {
            'final_val_loss': training_stats['final_val_loss'],
            'training_time_seconds': training_time,
            'model_parameters': sum(p.numel() for p in model.parameters()),
        },
        'usage': {
            'input_format': 'channel_real, channel_imag, interference, noise_power, user_priorities',
            'output_format': 'weights_real, weights_imag',
            'normalization': 'tanh activation, weights in [-1, 1]',
        }
    }
    
    config_path = output_dir / f"beamforming_config_{args.antennas}x{args.users}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved: {config_path}")
    print("\nModel generation complete!")
    print(f"\nFiles generated:")
    print(f"  - PyTorch model: {model_path}")
    print(f"  - ONNX model: {onnx_path}")
    print(f"  - Configuration: {config_path}")
    print(f"\nTo use with TensorRT, convert ONNX to TensorRT engine:")
    print(f"  trtexec --onnx={onnx_path} --saveEngine=beamforming_model_{args.antennas}x{args.users}.trt")

if __name__ == "__main__":
    main()