"""
Generate toy data exhibiting conditional dependence via a collider variable.
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np


def simulate_hidden_contamination(n_samples: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_samples, 1))
    y = rng.normal(0.0, 1.0, size=(n_samples, 1))
    z = x + y + rng.normal(0.0, 0.2, size=(n_samples, 1))
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
