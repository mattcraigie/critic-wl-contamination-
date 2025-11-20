"""Utility helpers for weak-lensing contamination experiments."""
from __future__ import annotations

import torch


def covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean(dim=0, keepdim=True)
    cov = x_centered.T @ x_centered / (x_centered.shape[0] - 1)
    return cov


def permutation_indices(n: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.randperm(n, device=device)
