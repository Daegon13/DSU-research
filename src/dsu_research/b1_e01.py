"""Frozen B1-E01 delayed-recall data and sparse temporal controls."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

D, H, K, C = 32, 64, 8, 2
DATASET_SEED = 41041
DELAYS = (4, 8, 16, 32)
SEEDS = (1, 2, 3)
LAMBDA = 2.0 ** (-1.0 / 16.0)
SLOPE = 0.1
SPLITS = {"train": (0, 50000), "validation": (1, 10000), "test": (2, 10000)}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return sha_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def topology(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    graph = np.stack([rng.choice(D, size=K, replace=False) for _ in range(H)])
    return torch.as_tensor(graph, dtype=torch.int64)


def initial_values(seed: int) -> dict[str, torch.Tensor]:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    w_in = torch.randn((H, K), generator=rng) * math.sqrt(2.0 / ((1 + SLOPE**2) * K))
    w_out = torch.empty((C, H), dtype=torch.float32)
    w_out.uniform_(-math.sqrt(6.0 / (H + C)), math.sqrt(6.0 / (H + C)), generator=rng)
    return {"W_in": w_in, "b_s": torch.zeros(H),
            "W_out": w_out, "b_out": torch.zeros(C)}


class PointSparseTemporal(nn.Module):
    """Identical sparse unit and readout, with an optional fixed EMA state."""

    def __init__(self, graph: torch.Tensor, values: dict[str, torch.Tensor], stateful: bool):
        super().__init__()
        if (graph.shape != (H, K) or graph.dtype != torch.int64 or
                torch.any(graph < 0) or torch.any(graph >= D) or
                any(torch.unique(row).numel() != K for row in graph)):
            raise ValueError("invalid IID sparse graph")
        self.register_buffer("G", graph.clone())
        self.W_in = nn.Parameter(values["W_in"].clone())
        self.b_s = nn.Parameter(values["b_s"].clone())
        self.W_out = nn.Parameter(values["W_out"].clone())
        self.b_out = nn.Parameter(values["b_out"].clone())
        self.stateful = stateful
        self.decay = float(np.float32(LAMBDA))

    def integrate(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != D:
            raise ValueError("expected 32 input channels")
        selected = x[..., self.G.reshape(-1)].reshape(*x.shape[:-1], H, K)
        z = (selected * self.W_in).sum(-1) + self.b_s
        return F.leaky_relu(z, negative_slope=SLOPE)

    def state_step(self, previous: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        return self.decay * previous + (1.0 - self.decay) * activation

    def forward(self, sequence: torch.Tensor, initial_state: torch.Tensor | None = None) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != D or sequence.shape[1] < 2:
            raise ValueError("expected [batch,time,32] sequence")
        if not self.stateful:
            if initial_state is not None:
                raise ValueError("static arm has no state")
            hidden = self.integrate(sequence[:, -1])
        else:
            activation = self.integrate(sequence)
            hidden = (torch.zeros((len(sequence), H), dtype=sequence.dtype, device=sequence.device)
                      if initial_state is None else initial_state)
            for t in range(sequence.shape[1]):
                hidden = self.state_step(hidden, activation[:, t])
        return F.linear(hidden, self.W_out, self.b_out)

    def cost(self, delay: int) -> dict:
        parameter_count = sum(p.numel() for p in self.parameters())
        return {"trainable_parameters": parameter_count,
                "weight_bytes_float32": parameter_count * 4,
                "topology_bytes_int64": self.G.numel() * self.G.element_size(),
                "persistent_tensor_payload_bytes": parameter_count * 4 + self.G.numel() * self.G.element_size(),
                "main_macs_per_output_timestep": H * K + H * C,
                "main_macs_per_sequence": H * K * (delay + 1) + H * C if self.stateful else H * K + H * C,
                "runtime_state_bytes_per_sample": H * 4 if self.stateful else 0,
                "state_update_elementwise_multiply_per_timestep": 2 * H if self.stateful else 0,
                "state_update_elementwise_add_per_timestep": H if self.stateful else 0}


def make_split(split: str, delay: int, dataset_seed: int = DATASET_SEED):
    """Return only x_1 and y; later steps are exactly zero except the final cue."""
    if split not in SPLITS or delay not in DELAYS:
        raise ValueError("invalid split or delay")
    split_id, size = SPLITS[split]
    first = np.zeros((size, D), dtype=np.float32)
    labels = np.arange(size, dtype=np.int64) % 2
    first[np.arange(size), labels] = 1.0
    for index in range(size):
        rng = np.random.default_rng(np.random.SeedSequence(
            [dataset_seed, split_id, delay, index]))
        first[index, 3:] = rng.uniform(-0.1, 0.1, size=29).astype(np.float32)
    return torch.from_numpy(first), torch.from_numpy(labels)


def sequence_batch(first: torch.Tensor, delay: int, oracle: bool = False) -> torch.Tensor:
    if first.ndim != 2 or first.shape[1] != D or delay not in DELAYS:
        raise ValueError("invalid initial input or delay")
    sequence = first.new_zeros((len(first), delay + 1, D))
    sequence[:, 0] = first
    sequence[:, -1, 2] = 1.0
    if oracle:
        sequence[:, -1, :2] = first[:, :2]
    return sequence


def split_fingerprint(first: torch.Tensor, labels: torch.Tensor, split: str, delay: int) -> str:
    payload = f"delayed_recall_easy_v1|41041|{split}|{delay}|".encode("ascii")
    return sha_bytes(payload + first.numpy().tobytes() + labels.numpy().tobytes())
