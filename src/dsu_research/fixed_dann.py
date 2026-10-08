"""Dense-mask control paired exactly with a DSU-S v0 topology."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .dsu_s import DSUSv0


class FixedDANN(nn.Module):
    """Dense storage with exactly the effective graph of a supplied DSU-S ``G``."""

    input_dim = 784
    somas_count = 128
    branches = 4
    fan_in = 16
    dendrites_count = somas_count * branches
    outputs = 10

    def __init__(self, G: torch.Tensor):
        super().__init__()
        topology = self._validated_topology(G)
        self.register_buffer("G", topology)

        input_mask = torch.zeros(self.dendrites_count, self.input_dim, dtype=torch.bool)
        rows = torch.arange(self.dendrites_count).repeat_interleave(self.fan_in)
        input_mask[rows, topology.reshape(-1)] = True
        soma_mask = torch.zeros(self.somas_count, self.dendrites_count, dtype=torch.bool)
        soma_mask[torch.arange(self.somas_count).repeat_interleave(self.branches),
                  torch.arange(self.dendrites_count)] = True
        self.register_buffer("input_mask", input_mask)
        self.register_buffer("soma_mask", soma_mask)

        self.dendrites = nn.Linear(self.input_dim, self.dendrites_count)
        self.somata = nn.Linear(self.dendrites_count, self.somas_count)
        self.output = nn.Linear(self.somas_count, self.outputs)
        self._reset_parameters()

    @classmethod
    def _validated_topology(cls, G: torch.Tensor) -> torch.Tensor:
        if not isinstance(G, torch.Tensor):
            raise TypeError("G must be a torch.Tensor")
        if tuple(G.shape) != (cls.somas_count, cls.branches, cls.fan_in):
            raise ValueError("G must have shape [128, 4, 16]")
        if G.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("G must contain integer indices")
        topology = G.detach().to(device="cpu", dtype=torch.int64).clone()
        if int(topology.min()) < 0 or int(topology.max()) >= cls.input_dim:
            raise ValueError("G indices must be in [0, 783]")
        if not torch.all(torch.diff(topology.sort(dim=-1).values, dim=-1) > 0):
            raise ValueError("each dendrite must contain 16 distinct input indices")
        return topology

    def _reset_parameters(self) -> None:
        for layer in (self.dendrites, self.somata, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.dendrites.weight.mul_(self.input_mask)
            self.somata.weight.mul_(self.soma_mask)

    def dendrite_pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("Fixed-dANN expects input shape [batch, 784]")
        pre = F.linear(x, self.dendrites.weight * self.input_mask, self.dendrites.bias)
        return pre.reshape(-1, self.somas_count, self.branches)

    def dendrite_activations(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.dendrite_pre_activations(x), negative_slope=0.1)

    def soma_pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        dendrites = self.dendrite_activations(x).reshape(-1, self.dendrites_count)
        return F.linear(dendrites, self.somata.weight * self.soma_mask, self.somata.bias)

    def soma_activations(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.soma_pre_activations(x), negative_slope=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.soma_activations(x))

    def effective_parameters(self) -> int:
        return (int(self.input_mask.sum()) + self.dendrites.bias.numel()
                + int(self.soma_mask.sum()) + self.somata.bias.numel()
                + self.output.weight.numel() + self.output.bias.numel())

    def structural_metrics(self) -> dict[str, int]:
        return {
            "stored_parameters": sum(p.numel() for p in self.parameters()),
            "effective_parameters": self.effective_parameters(),
            "input_mask_connections": int(self.input_mask.sum()),
            "soma_mask_connections": int(self.soma_mask.sum()),
        }


def fixed_dann_from_dsu_s(dsu: DSUSv0) -> FixedDANN:
    """Create a dense-mask model with the exact graph and active values of ``dsu``."""
    fixed = FixedDANN(dsu.G).to(device=dsu.W_d.device, dtype=dsu.W_d.dtype)
    rows = torch.arange(fixed.dendrites_count, device=dsu.W_d.device).repeat_interleave(
        fixed.fan_in)
    columns = dsu.G.reshape(-1)
    soma_rows = torch.arange(fixed.somas_count, device=dsu.W_d.device).repeat_interleave(
        fixed.branches)
    dendrite_columns = torch.arange(fixed.dendrites_count, device=dsu.W_d.device)

    with torch.no_grad():
        fixed.dendrites.weight.zero_()
        fixed.somata.weight.zero_()
        fixed.dendrites.weight[rows, columns] = dsu.W_d.reshape(-1)
        fixed.dendrites.bias.copy_(dsu.bias_d.reshape(-1))
        fixed.somata.weight[soma_rows, dendrite_columns] = dsu.W_s.reshape(-1)
        fixed.somata.bias.copy_(dsu.bias_s)
        fixed.output.weight.copy_(dsu.W_o)
        fixed.output.bias.copy_(dsu.bias_o)
    return fixed
