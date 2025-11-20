"""Joint training of T-network and critic to estimate conditional dependence."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from models.critic import build_critic
from models.t_network import build_t_network
from training.losses import select_loss_fn


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"], data["Y"], data["Z"]


class ToyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.z = torch.from_numpy(z)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.z[idx]


def train_epoch(critic: nn.Module, t_net: nn.Module, loader: DataLoader, loss_fn, optimizer: optim.Optimizer, device: torch.device):
    critic.train()
    t_net.train()
    total_loss = 0.0
    for x, y, z in loader:
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        optimizer.zero_grad()
        t = t_net(z)
        pos_scores = critic(x, y, t)
        perm = torch.randperm(y.shape[0], device=device)
        y_perm = y[perm]
        neg_scores = critic(x, y_perm, t)
        loss = loss_fn(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.shape[0]
    return total_loss / len(loader.dataset)


def estimate_conditional_mi(critic: nn.Module, t_net: nn.Module, loader: DataLoader, device: torch.device) -> float:
    critic.eval()
    t_net.eval()
    pos_scores = []
    neg_scores = []
    with torch.no_grad():
        for x, y, z in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            t = t_net(z)
            pos_scores.append(critic(x, y, t))
            perm = torch.randperm(y.shape[0], device=device)
            y_perm = y[perm]
            neg_scores.append(critic(x, y_perm, t))
    pos = torch.cat(pos_scores)
    neg = torch.cat(neg_scores)
    return (pos.mean() - torch.logsumexp(neg, dim=0) + torch.log(torch.tensor(neg.shape[0], device=device))).item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditional dependence critic on toy data.")
    parser.add_argument(
        "--data",
        type=str,
        default="../data/toy_data.npz",
        help="Path to NPZ produced by generate_toy_data",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--loss", type=str, default="info_nce", choices=["dv", "info_nce", "nwj"], help="Loss objective")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--t-dim", type=int, default=2, help="Dimensionality of T")
    parser.add_argument("--output", type=str, default="experiments/conditional_results.pt", help="Where to save trained weights")
    args = parser.parse_args()

    x, y, z = load_npz(args.data)
    dataset = ToyDataset(x, y, z)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = build_critic(x_dim=x.shape[1], y_dim=y.shape[1], t_dim=args.t_dim).to(device)
    t_net = build_t_network(z_dim=z.shape[1], t_dim=args.t_dim).to(device)

    parameters = list(critic.parameters()) + list(t_net.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)
    loss_fn = select_loss_fn(args.loss)

    for epoch in range(args.epochs):
        loss = train_epoch(critic, t_net, loader, loss_fn, optimizer, device)
        mi_est = estimate_conditional_mi(critic, t_net, loader, device)
        print(f"Epoch {epoch+1:02d} | loss={loss:.4f} | estimated I(X;Y|T)={mi_est:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"critic": critic.state_dict(), "t_net": t_net.state_dict()}, args.output)
    print(f"Saved trained models to {args.output}")


if __name__ == "__main__":
    main()
