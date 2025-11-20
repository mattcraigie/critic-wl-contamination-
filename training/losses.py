"""Contrastive divergence objectives for conditional dependence estimation."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dv_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -(pos_scores.mean() - torch.logsumexp(neg_scores, dim=0) + torch.log(torch.tensor(neg_scores.shape[0], device=neg_scores.device)))


def nwj_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -(pos_scores.mean() - (torch.exp(neg_scores - 1).mean()))


def info_nce_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    logits = torch.stack([pos_scores, neg_scores], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def select_loss_fn(name: str):
    name = name.lower()
    if name == "dv":
        return dv_loss
    if name == "info_nce":
        return info_nce_loss
    if name == "nwj":
        return nwj_loss
    raise ValueError(f"Unknown loss {name}")
