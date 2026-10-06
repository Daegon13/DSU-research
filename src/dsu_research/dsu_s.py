"""Physically compact, spatial DSU-S v0 with fixed per-dendrite topology."""

import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DSUSv0(nn.Module):
    input_dim = 784
    somas = 128
    branches = 4
    fan_in = 16
    outputs = 10

    def __init__(self, topology_seed: int):
        super().__init__()
        if type(topology_seed) is not int or not 0 <= topology_seed <= 2**32 - 1:
            raise ValueError("topology_seed must be a nonnegative 32-bit integer")
        self.topology_seed = topology_seed
        rng = np.random.default_rng(topology_seed)
        indices = np.stack([
            rng.choice(self.input_dim, self.fan_in, replace=False)
            for _ in range(self.somas * self.branches)
        ]).reshape(self.somas, self.branches, self.fan_in)
        self.register_buffer("G", torch.from_numpy(indices.astype(np.int64)))

        self.W_d = nn.Parameter(torch.empty(self.somas, self.branches, self.fan_in))
        self.bias_d = nn.Parameter(torch.zeros(self.somas, self.branches))
        self.W_s = nn.Parameter(torch.empty(self.somas, self.branches))
        self.bias_s = nn.Parameter(torch.zeros(self.somas))
        self.W_o = nn.Parameter(torch.empty(self.outputs, self.somas))
        self.bias_o = nn.Parameter(torch.zeros(self.outputs))
        # Explicit Glorot uniform for each compact weight bank; zero biases.
        # The flattened banks define (fan_out, fan_in) for PyTorch's initializer.
        nn.init.xavier_uniform_(self.W_d.view(-1, self.fan_in))
        nn.init.xavier_uniform_(self.W_s)
        nn.init.xavier_uniform_(self.W_o)

    def dendrite_pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("DSU-S expects input shape [batch, 784]")
        selected = x[:, self.G.reshape(-1)].reshape(-1, self.somas, self.branches, self.fan_in)
        return (selected * self.W_d).sum(dim=-1) + self.bias_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        dendrites = F.leaky_relu(self.dendrite_pre_activations(x), negative_slope=0.1)
        somata = F.leaky_relu((dendrites * self.W_s).sum(dim=-1) + self.bias_s,
                              negative_slope=0.1)
        return F.linear(somata, self.W_o, self.bias_o)

    def effective_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def structural_metrics(self) -> dict:
        trainable = [p for p in self.parameters() if p.requires_grad]
        return {
            "trainable_parameter_bytes": sum(p.numel() * p.element_size() for p in trainable),
            "topology_index_count": self.G.numel(),
            "topology_index_dtype": str(self.G.dtype),
            "topology_index_bytes": self.G.numel() * self.G.element_size(),
            "topology_sha256": hashlib.sha256(self.G.cpu().numpy().astype("<i8", copy=False).tobytes()).hexdigest(),
            "theoretical_main_macs_per_sample": (
                self.W_d.numel() + self.W_s.numel() + self.W_o.numel()),
        }
