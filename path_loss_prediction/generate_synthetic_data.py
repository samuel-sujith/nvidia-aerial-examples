#!/usr/bin/env python3
"""
Generate synthetic path loss training data.

Outputs CSV with columns:
rsrp_dbm, sinr_db, distance_m, carrier_freq_ghz, shadow_fading_db,
ue_speed_mps, cell_load, buffer_norm, path_loss_db
"""

import argparse
import csv
import math
import random


def log_distance_path_loss(distance_m, freq_ghz, shadow_db):
    # Friis + log-distance model (simplified)
    d_km = max(distance_m / 1000.0, 0.01)
    pl = 32.4 + 20 * math.log10(freq_ghz * 1000) + 20 * math.log10(d_km)
    return pl + shadow_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--output", type=str, default="synthetic_path_loss.csv")
    parser.add_argument("--freq", type=float, default=3.5)
    args = parser.parse_args()

    rng = random.Random(42)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rsrp_dbm", "sinr_db", "distance_m", "carrier_freq_ghz",
            "shadow_fading_db", "ue_speed_mps", "cell_load", "buffer_norm",
            "path_loss_db"
        ])

        for _ in range(args.samples):
            distance_m = rng.uniform(50.0, 1500.0)
            shadow = rng.gauss(0.0, 4.0)
            speed = rng.uniform(0.0, 30.0)
            cell_load = rng.random()
            buffer_norm = rng.random()
            rsrp = -70.0 - 20.0 * math.log10(max(distance_m / 1000.0, 0.01)) + rng.uniform(-2.0, 2.0)
            sinr = 15.0 - 10.0 * (distance_m / 1000.0) + rng.uniform(-3.0, 3.0)
            path_loss = log_distance_path_loss(distance_m, args.freq, shadow)

            writer.writerow([
                rsrp, sinr, distance_m, args.freq,
                shadow, speed, cell_load, buffer_norm,
                path_loss
            ])


if __name__ == "__main__":
    main()
