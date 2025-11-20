"""
Generate toy data exhibiting hidden conditional dependence.
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np


def simulate_hidden_contamination(n_samples: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
    x = rng.normal(0.0, 1.0, size=(n_samples, 1))
    y_clean = rng.normal(0.0, 0.1, size=(n_samples, 1))
    contamination = 0.6 * x * (1 / (1 + np.exp(-5 * z)))
    y = y_clean + contamination
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy weak-lensing contamination dataset.")
    parser.add_argument("--n-samples", type=int, default=20000, help="Number of synthetic samples to draw")
    parser.add_argument(
        "--output", type=str, default=os.path.join("data", "toy_data.npz"), help="Output NPZ file for arrays X, Y, Z"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    x, y, z = simulate_hidden_contamination(args.n_samples, args.seed)
    np.savez(args.output, X=x, Y=y, Z=z)
    print(f"Saved synthetic dataset to {args.output} with shapes: X={x.shape}, Y={y.shape}, Z={z.shape}")


if __name__ == "__main__":
    main()
