"""Example: Analyze gradient norms and numerical health across deep models."""

from __future__ import annotations

import torch
import torch.nn as nn
import seedl as sd


class DeepMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 8):
        super().__init__()
        layers = []
        in_dim = 32
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def main():
    print("Initializing Deep MLP model...")
    model = DeepMLP(hidden_dim=64, num_layers=8)
    x = torch.randn(4, 32)

    print("Computing loss and backward pass...")
    out = model(x)
    loss = nn.functional.cross_entropy(out, torch.tensor([0, 1, 0, 1]))
    loss.backward()

    print("Auditing Gradient Health...")
    viz_health = sd.gradient_health(model, title="Deep MLP Gradient Health")
    path = viz_health.save("gradient_health.html")
    print(f"Gradient Health report saved to: {path}")


if __name__ == "__main__":
    main()
