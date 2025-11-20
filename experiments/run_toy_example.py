"""End-to-end toy run for conditional contamination detection."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import pearsonr
from torch import optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data.generate_toy_data import simulate_hidden_contamination
from training.train_conditional import ToyDataset, estimate_conditional_mi, train_epoch
from models.critic import build_critic
from models.t_network import build_t_network
from training.losses import select_loss_fn


def ensure_data(path: str, n_samples: int = 20000) -> None:
    if os.path.exists(path):
        return
    x, y, z = simulate_hidden_contamination(n_samples)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X=x, Y=y, Z=z)
    print(f"Generated synthetic dataset at {path}")


def report_global_dependence(path: str) -> None:
    data = np.load(path)
    corr, p = pearsonr(data["X"].flatten(), data["Y"].flatten())
    print(f"Global X-Y Pearson correlation: r={corr:.4f}, p={p:.2e}")


def train_models(path: str, epochs: int = 5, batch_size: int = 256, loss_name: str = "info_nce", t_dim: int = 2) -> None:
    data = np.load(path)
    x, y, z = data["X"], data["Y"], data["Z"]
    dataset = ToyDataset(x, y, z)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = build_critic(x_dim=x.shape[1], y_dim=y.shape[1], t_dim=t_dim).to(device)
    t_net = build_t_network(z_dim=z.shape[1], t_dim=t_dim).to(device)
    optimizer = optim.Adam(list(critic.parameters()) + list(t_net.parameters()), lr=1e-3)
    loss_fn = select_loss_fn(loss_name)

    for epoch in range(epochs):
        loss = train_epoch(critic, t_net, loader, loss_fn, optimizer, device)
        mi_est = estimate_conditional_mi(critic, t_net, loader, device)
        print(f"Epoch {epoch+1}/{epochs}: loss={loss:.4f}, estimated I(X;Y|T)={mi_est:.4f}")

    output = Path("experiments/conditional_results.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"critic": critic.state_dict(), "t_net": t_net.state_dict()}, output)
    print(f"Saved trained weights to {output}")


def main() -> None:
    dataset_path = os.path.join("..", "data", "toy_data.npz")
    ensure_data(dataset_path)
    report_global_dependence(dataset_path)
    train_models(dataset_path)


if __name__ == "__main__":
    main()
