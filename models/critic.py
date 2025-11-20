"""Contrastive critic scoring (X, Y, T) tuples."""
from __future__ import annotations

import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([x, y, t], dim=-1)
        return self.model(features).squeeze(-1)


def build_critic(x_dim: int, y_dim: int, t_dim: int, hidden_dim: int = 128) -> Critic:
    return Critic(input_dim=x_dim + y_dim + t_dim, hidden_dim=hidden_dim)
