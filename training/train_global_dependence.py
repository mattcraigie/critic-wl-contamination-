"""Quick global dependence check between X and Y."""
from __future__ import annotations

import argparse
import numpy as np
from scipy.stats import pearsonr


def load_npz(path: str):
    data = np.load(path)
    return data["X"], data["Y"], data["Z"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a simple global Pearson correlation between X and Y.")
    parser.add_argument(
        "--data",
        type=str,
        default="../data/toy_data.npz",
        help="Path to NPZ produced by generate_toy_data",
    )
    args = parser.parse_args()

    x, y, _ = load_npz(args.data)
    corr, p = pearsonr(x.flatten(), y.flatten())
    print(f"Global Pearson correlation r={corr:.4f}, p-value={p:.2e}")


if __name__ == "__main__":
    main()
