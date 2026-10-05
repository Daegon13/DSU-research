"""Conventional dense MLP baseline; no experimental units."""

from torch import nn


class MLPReLU(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.layers(x)
