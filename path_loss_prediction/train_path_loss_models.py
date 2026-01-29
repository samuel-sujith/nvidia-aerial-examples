#!/usr/bin/env python3
"""
Train path loss models from synthetic CSV.

Exports models in a simple text format consumable by the C++ module:
  - type=mlp
  - type=xgboost
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import struct
import json

try:
    import numpy as np
except ImportError as exc:
    print("numpy is required for training. Please install numpy.", file=sys.stderr)
    raise


def load_csv(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data")
    feature_names = [
        "rsrp_dbm", "sinr_db", "distance_m", "carrier_freq_ghz",
        "shadow_fading_db", "ue_speed_mps", "cell_load", "buffer_norm"
    ]
    x = np.array([[float(r[n]) for n in feature_names] for r in rows], dtype=np.float32)
    y = np.array([float(r["path_loss_db"]) for r in rows], dtype=np.float32)
    return x, y


def write_binary_header(f, model_type, num_features, hidden_size, num_trees):
    # magic + version + type + dims
    f.write(b"PLM1")
    f.write(struct.pack("<I", 1))  # version
    f.write(struct.pack("<I", model_type))
    f.write(struct.pack("<I", num_features))
    f.write(struct.pack("<I", hidden_size))
    f.write(struct.pack("<I", num_trees))


def train_mlp(x, y, hidden_size=16, epochs=200, lr=1e-3, seed=42):
    rng = np.random.default_rng(seed)
    num_samples, num_features = x.shape

    w1 = rng.normal(scale=0.05, size=(hidden_size, num_features)).astype(np.float32)
    b1 = np.zeros((hidden_size,), dtype=np.float32)
    w2 = rng.normal(scale=0.05, size=(hidden_size,)).astype(np.float32)
    b2 = np.array([np.mean(y)], dtype=np.float32)

    for _ in range(epochs):
        # forward
        h = x @ w1.T + b1
        h_relu = np.maximum(h, 0)
        y_pred = h_relu @ w2 + b2[0]

        # loss and gradient
        err = (y_pred - y) / num_samples
        grad_b2 = np.sum(err)
        grad_w2 = h_relu.T @ err
        grad_h = err[:, None] * w2[None, :]
        grad_h[h <= 0] = 0
        grad_w1 = grad_h.T @ x
        grad_b1 = np.sum(grad_h, axis=0)

        # update
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2

    return w1, b1, w2, b2


def best_stump(x, residuals, thresholds_per_feature=5):
    num_samples, num_features = x.shape
    best = None
    best_loss = float("inf")

    for f in range(num_features):
        values = x[:, f]
        percentiles = np.linspace(10, 90, thresholds_per_feature)
        thresholds = np.percentile(values, percentiles)
        for thr in thresholds:
            left_mask = values < thr
            right_mask = ~left_mask
            if not np.any(left_mask) or not np.any(right_mask):
                continue
            left_val = residuals[left_mask].mean()
            right_val = residuals[right_mask].mean()
            pred = np.where(left_mask, left_val, right_val)
            mse = np.mean((residuals - pred) ** 2)
            if mse < best_loss:
                best_loss = mse
                best = (f, float(thr), float(left_val), float(right_val))

    if best is None:
        best = (0, float(x[:, 0].mean()), float(residuals.mean()), float(residuals.mean()))
    return best


def train_xgboost_stumps(x, y, num_trees=8, learning_rate=0.3):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is required for XGBoost training") from exc

    model = xgb.XGBRegressor(
        n_estimators=num_trees,
        max_depth=1,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(x, y)

    dump = model.get_booster().get_dump(dump_format="json")
    feature_idx = []
    thresholds = []
    left_values = []
    right_values = []

    for tree_json in dump:
        tree = json.loads(tree_json)
        split = tree.get("split", "f0")
        f = int(split[1:]) if split.startswith("f") else 0
        thr = float(tree.get("split_condition", 0.0))
        children = tree.get("children", [])
        left = right = 0.0
        if len(children) >= 2:
            left = float(children[0].get("leaf", 0.0))
            right = float(children[1].get("leaf", 0.0))
        feature_idx.append(f)
        thresholds.append(thr)
        left_values.append(left)
        right_values.append(right)

    return feature_idx, thresholds, left_values, right_values


def export_mlp(path, w1, b1, w2, b2):
    with open(path, "wb") as f:
        write_binary_header(f, model_type=0, num_features=w1.shape[1],
                            hidden_size=w1.shape[0], num_trees=0)
        f.write(w1.astype(np.float32).tobytes(order="C"))
        f.write(b1.astype(np.float32).tobytes(order="C"))
        f.write(w2.astype(np.float32).tobytes(order="C"))
        f.write(b2.astype(np.float32).tobytes(order="C"))


def export_xgboost(path, feature_idx, thresholds, left_values, right_values, num_features):
    with open(path, "wb") as f:
        write_binary_header(f, model_type=1, num_features=num_features,
                            hidden_size=0, num_trees=len(feature_idx))
        f.write(np.array(feature_idx, dtype=np.int32).tobytes(order="C"))
        f.write(np.array(thresholds, dtype=np.float32).tobytes(order="C"))
        f.write(np.array(left_values, dtype=np.float32).tobytes(order="C"))
        f.write(np.array(right_values, dtype=np.float32).tobytes(order="C"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="synthetic_path_loss.csv")
    parser.add_argument("--mlp-out", type=str, default="path_loss_mlp.bin")
    parser.add_argument("--xgb-out", type=str, default="path_loss_xgb.bin")
    parser.add_argument("--onnx-out", type=str, default="")
    parser.add_argument("--trt-engine-out", type=str, default="")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-trees", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    args = parser.parse_args()

    x, y = load_csv(args.input)

    w1, b1, w2, b2 = train_mlp(x, y, hidden_size=args.hidden_size,
                               epochs=args.epochs, lr=args.lr)
    export_mlp(args.mlp_out, w1, b1, w2, b2)

    feature_idx, thresholds, left_values, right_values = train_xgboost_stumps(
        x, y, num_trees=args.num_trees, learning_rate=args.learning_rate)
    export_xgboost(args.xgb_out, feature_idx, thresholds, left_values, right_values, x.shape[1])

    print(f"Wrote MLP model: {args.mlp_out}")
    print(f"Wrote XGBoost model: {args.xgb_out}")

    if args.onnx_out:
        try:
            import torch
            import torch.nn as nn

            class SimpleMLP(nn.Module):
                def __init__(self, in_features, hidden):
                    super().__init__()
                    self.fc1 = nn.Linear(in_features, hidden)
                    self.fc2 = nn.Linear(hidden, 1)

                def forward(self, x):
                    return self.fc2(torch.relu(self.fc1(x)))

            model = SimpleMLP(x.shape[1], args.hidden_size)
            with torch.no_grad():
                model.fc1.weight.copy_(torch.from_numpy(w1))
                model.fc1.bias.copy_(torch.from_numpy(b1))
                model.fc2.weight.copy_(torch.from_numpy(w2.reshape(1, -1)))
                model.fc2.bias.copy_(torch.from_numpy(b2))

            dummy = torch.zeros(1, x.shape[1], dtype=torch.float32)
            torch.onnx.export(model, dummy, args.onnx_out,
                              input_names=["features"], output_names=["path_loss_db"],
                              opset_version=13)
            print(f"Wrote ONNX model: {args.onnx_out}")
        except ImportError:
            print("torch is required for ONNX export. Skipping ONNX output.", file=sys.stderr)

    if args.trt_engine_out:
        if not args.onnx_out:
            print("ONNX output is required for TensorRT engine build.", file=sys.stderr)
            return
        if not os.path.exists(args.onnx_out):
            print(f"ONNX file not found: {args.onnx_out}", file=sys.stderr)
            return

        trtexec = os.environ.get("TRTEXEC", "trtexec")
        cmd = [
            trtexec,
            f"--onnx={args.onnx_out}",
            f"--saveEngine={args.trt_engine_out}",
            "--explicitBatch",
        ]
        print("Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            print(f"Wrote TensorRT engine: {args.trt_engine_out}")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"Failed to build TensorRT engine: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
