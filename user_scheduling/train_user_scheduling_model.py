#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train_logistic_regression(X, y, epochs, lr):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float32)
    bias = 0.0

    for _ in range(epochs):
        logits = X @ weights + bias
        preds = sigmoid(logits)
        error = preds - y

        grad_w = (X.T @ error) / n_samples
        grad_b = float(np.mean(error))

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def save_model(path, feature_names, mean, std, weights, bias):
    lines = [
        "# user scheduling logistic regression model",
        f"num_features={len(feature_names)}",
        "feature_names=" + ",".join(feature_names),
        "mean=" + ",".join(f"{v:.6f}" for v in mean),
        "std=" + ",".join(f"{v:.6f}" for v in std),
        "weights=" + ",".join(f"{v:.6f}" for v in weights),
        f"bias={bias:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train a simple scheduling model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("user_scheduling_data.npz"),
        help="Input dataset path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("user_scheduling_model.txt"),
        help="Output model path",
    )
    parser.add_argument(
        "--export-onnx",
        type=Path,
        default=None,
        help="Optional ONNX export path (requires torch)",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Learning rate")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    feature_names = [str(name) for name in data["feature_names"]]
    num_ues = int(data["num_ues"]) if "num_ues" in data.files else 1

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X_norm = (X - mean) / std

    weights, bias = train_logistic_regression(X_norm, y, args.epochs, args.learning_rate)

    logits = X_norm @ weights + bias
    preds = sigmoid(logits)
    accuracy = float(np.mean((preds >= 0.5) == (y >= 0.5)))

    save_model(args.output, feature_names, mean, std, weights, bias)
    print(f"Saved model to {args.output}")
    print(f"Training accuracy: {accuracy:.4f}")

    if args.export_onnx:
        try:
            import torch
            import torch.nn as nn

            class SchedulingModel(nn.Module):
                def __init__(self, mean_tensor, std_tensor, weight_tensor, bias_tensor):
                    super().__init__()
                    self.register_buffer("mean", mean_tensor)
                    self.register_buffer("std", std_tensor)
                    self.linear = nn.Linear(weight_tensor.shape[0], 1, bias=True)
                    with torch.no_grad():
                        self.linear.weight.copy_(weight_tensor.unsqueeze(0))
                        self.linear.bias.copy_(bias_tensor)

                def forward(self, x):
                    x_norm = (x - self.mean) / self.std
                    logits = self.linear(x_norm)
                    return torch.sigmoid(logits)

            mean_tensor = torch.tensor(mean, dtype=torch.float32)
            std_tensor = torch.tensor(std, dtype=torch.float32)
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
            bias_tensor = torch.tensor([bias], dtype=torch.float32)
            model = SchedulingModel(mean_tensor, std_tensor, weight_tensor, bias_tensor)
            model.eval()

            dummy = torch.zeros((num_ues, X.shape[1]), dtype=torch.float32)
            torch.onnx.export(
                model,
                dummy,
                args.export_onnx,
                input_names=["ue_features"],
                output_names=["ue_scores"],
                opset_version=13,
            )
            print(f"Exported ONNX model to {args.export_onnx}")
        except Exception as exc:
            print(f"Failed to export ONNX model: {exc}")


if __name__ == "__main__":
    main()
