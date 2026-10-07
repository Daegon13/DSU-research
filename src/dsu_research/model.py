"""Sprint 0 baseline and the fixed Fashion-MNIST reproduction models."""

import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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


def _glorot_linear(layer: nn.Linear) -> None:
    """Match Keras Dense's kernel/bias defaults, not PyTorch Linear defaults."""
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


class VanillaANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dendrites = nn.Linear(784, 512)
        self.somata = nn.Linear(512, 128)
        self.output = nn.Linear(128, 10)
        for layer in (self.dendrites, self.somata, self.output):
            _glorot_linear(layer)

    def forward(self, x):
        x = F.leaky_relu(self.dendrites(x.flatten(1)), negative_slope=0.1)
        x = F.leaky_relu(self.somata(x), negative_slope=0.1)
        return self.output(x)


class ParameterMatchedMLP(nn.Module):
    """Frozen 784->13->17->10 dense baseline for S2-T05."""

    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 13)
        self.hidden2 = nn.Linear(13, 17)
        self.output = nn.Linear(17, 10)
        for layer in (self.hidden1, self.hidden2, self.output):
            _glorot_linear(layer)

    def forward(self, x):
        x = F.leaky_relu(self.hidden1(x.flatten(1)), negative_slope=0.1)
        x = F.leaky_relu(self.hidden2(x), negative_slope=0.1)
        return self.output(x)

    def structural_metrics(self) -> dict:
        return {
            "trainable_parameter_bytes": sum(p.numel() * p.element_size() for p in self.parameters() if p.requires_grad),
            "theoretical_main_macs_per_sample": sum(layer.weight.numel() for layer in
                                                     (self.hidden1, self.hidden2, self.output)),
        }


class CapacityMLP(nn.Module):
    """Two-hidden-layer dense family for the predefined S2-T06 search."""

    def __init__(self, hidden1: int):
        super().__init__()
        if type(hidden1) is not int or hidden1 < 1:
            raise ValueError("hidden1 must be a positive integer")
        hidden2 = round(17 / 13 * hidden1)
        self.hidden1 = nn.Linear(784, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2, 10)
        for layer in (self.hidden1, self.hidden2, self.output):
            _glorot_linear(layer)

    def forward(self, x):
        x = F.leaky_relu(self.hidden1(x.flatten(1)), negative_slope=0.1)
        x = F.leaky_relu(self.hidden2(x), negative_slope=0.1)
        return self.output(x)

    def structural_metrics(self) -> dict:
        return {
            "trainable_parameter_bytes": sum(p.numel() * p.element_size() for p in self.parameters()),
            "theoretical_main_macs_per_sample": sum(layer.weight.numel() for layer in
                                                     (self.hidden1, self.hidden2, self.output)),
        }


class DendriticANNRandom(VanillaANN):
    """Dense Keras-equivalent storage with fixed RANDOM and cable masks."""

    def __init__(self, seed: int):
        super().__init__()
        # Official random_connectivity: choose 8192 flattened positions from
        # an input-major (784, 512) matrix without replacement.
        rng = np.random.default_rng(seed)
        positions = rng.choice(784 * 512, 16 * 512, replace=False)
        input_mask = np.zeros((784, 512), dtype=np.uint8)
        input_mask.flat[positions] = 1
        input_mask = np.ascontiguousarray(input_mask.T)
        cable_mask = np.zeros((128, 512), dtype=np.uint8)
        for soma in range(128):
            cable_mask[soma, 4 * soma:4 * (soma + 1)] = 1
        self.register_buffer("input_mask", torch.from_numpy(input_mask).bool())
        self.register_buffer("cable_mask", torch.from_numpy(cable_mask).bool())
        with torch.no_grad():
            self.dendrites.weight.mul_(self.input_mask)
            self.somata.weight.mul_(self.cable_mask)

    def forward(self, x):
        x = F.linear(x.flatten(1), self.dendrites.weight * self.input_mask, self.dendrites.bias)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.linear(x, self.somata.weight * self.cable_mask, self.somata.bias)
        x = F.leaky_relu(x, negative_slope=0.1)
        return self.output(x)

    def connectivity_metrics(self) -> dict:
        counts = self.input_mask.sum(dim=1)
        input_active = int(counts.sum().item())
        cable_active = int(self.cable_mask.sum().item())
        return {
            "mask_active_connections": input_active + cable_active,
            "input_mask_active_connections": input_active,
            "cable_mask_active_connections": cable_active,
            "average_inputs_per_dendrite": float(counts.float().mean().item()),
            "min_inputs_per_dendrite": int(counts.min().item()),
            "max_inputs_per_dendrite": int(counts.max().item()),
            "input_mask_sha256": hashlib.sha256(self.input_mask.cpu().numpy().tobytes()).hexdigest(),
        }

    def effective_parameters(self) -> int:
        return (int(self.input_mask.sum()) + self.dendrites.bias.numel()
                + int(self.cable_mask.sum()) + self.somata.bias.numel()
                + self.output.weight.numel() + self.output.bias.numel())
