#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def generate_data(num_ttis, num_ues, num_scheduled, seed):
    rng = np.random.default_rng(seed)
    features = []
    labels = []

    feature_names = [
        "sinr_db",
        "buffer_bytes",
        "avg_rate_mbps",
        "qos_priority",
        "harq_pending",
    ]

    for _ in range(num_ttis):
        sinr_db = clamp(rng.normal(10.0, 6.0, size=num_ues), -5.0, 30.0)
        buffer_bytes = rng.integers(10_000, 5_000_000, size=num_ues)
        avg_rate_mbps = rng.uniform(1.0, 200.0, size=num_ues)
        qos_priority = rng.choice([0.5, 0.7, 0.9, 1.0], size=num_ues, p=[0.2, 0.3, 0.3, 0.2])
        harq_pending = rng.binomial(1, 0.2, size=num_ues)

        buffer_norm = buffer_bytes / 5_000_000.0
        rate_norm = avg_rate_mbps / 200.0
        utility = (
            0.50 * sigmoid(sinr_db / 5.0)
            + 0.30 * buffer_norm
            + 0.20 * qos_priority
            - 0.35 * rate_norm
            + 0.10 * harq_pending
        )

        scheduled_idx = np.argsort(utility)[-num_scheduled:]
        tti_labels = np.zeros(num_ues, dtype=np.float32)
        tti_labels[scheduled_idx] = 1.0

        tti_features = np.stack(
            [sinr_db, buffer_bytes, avg_rate_mbps, qos_priority, harq_pending], axis=1
        ).astype(np.float32)

        features.append(tti_features)
        labels.append(tti_labels)

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y, feature_names


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for user scheduling.")
    parser.add_argument("--num-ttis", type=int, default=500, help="Number of TTIs to simulate")
    parser.add_argument("--num-ues", type=int, default=32, help="Number of UEs per TTI")
    parser.add_argument("--num-scheduled", type=int, default=8, help="Number of scheduled UEs per TTI")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("user_scheduling_data.npz"),
        help="Output dataset path",
    )
    args = parser.parse_args()

    X, y, feature_names = generate_data(
        args.num_ttis, args.num_ues, args.num_scheduled, args.seed
    )

    np.savez(
        args.output,
        X=X,
        y=y,
        feature_names=np.array(feature_names),
        num_ttis=args.num_ttis,
        num_ues=args.num_ues,
        num_scheduled=args.num_scheduled,
    )

    summary = {
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "scheduled_ratio": float(y.mean()),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
