"""Simple MLP mapping contextual variables Z to a low-dimensional T."""
from __future__ import annotations

import torch
from torch import nn


class TNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return the learned contextual representation T."""
        return self.model(z)


def build_t_network(z_dim: int, t_dim: int = 2, hidden_dim: int = 64) -> TNetwork:
    return TNetwork(input_dim=z_dim, hidden_dim=hidden_dim, output_dim=t_dim)
