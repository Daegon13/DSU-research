"""S3-T06: compact sparse point and parameter-neutral compartment controls."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from .compartment_ablation import BRANCHES, SLOPE, base_topology


def _validate_g64(g64):
    if g64.shape != (128, 64) or g64.dtype != torch.int64:
        raise ValueError("G64 must be int64 [128,64]")
    if bool(torch.any((g64 < 0) | (g64 >= 784))):
        raise ValueError("G64 indices must be in [0,784)")
    if bool(torch.any(torch.sort(g64, dim=1).values[:, 1:] ==
                      torch.sort(g64, dim=1).values[:, :-1])):
        raise ValueError("G64 must contain 64 unique indices per soma")


def _sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def paired_values(seed):
    """Local generator; identical base input, soma bias and output for every arm."""
    generator = torch.Generator().manual_seed(seed)
    base_input = torch.randn((128, 64), generator=generator)
    soma_bias = torch.zeros(128)
    output = torch.empty((10, 128))
    output.uniform_(-math.sqrt(6 / 138), math.sqrt(6 / 138), generator=generator)
    output_bias = torch.zeros(10)
    return base_input, soma_bias, output, output_bias


class _CommonControl(nn.Module):
    def __init__(self, g64):
        super().__init__()
        _validate_g64(g64)
        self.register_buffer("G64", g64.clone())
        self.W_in = nn.Parameter(torch.empty(128, 64))
        self.bias_s = nn.Parameter(torch.zeros(128))
        self.W_o = nn.Parameter(torch.empty(10, 128))
        self.bias_o = nn.Parameter(torch.zeros(10))
        self.initialize(*paired_values(0))

    def initialize(self, base_input, soma_bias, output, output_bias):
        if (base_input.shape, soma_bias.shape, output.shape, output_bias.shape) != (
                (128, 64), (128,), (10, 128), (10,)):
            raise ValueError("paired initialization tensor shapes are invalid")
        # Kaiming variance per branch: Var(W)=2/((1+a^2)K).
        k = 64 // getattr(self, "branches", 1)
        scale = math.sqrt(2 / (1 + SLOPE**2) / k)
        with torch.no_grad():
            self.W_in.copy_(base_input * scale)
            self.bias_s.copy_(soma_bias)
            self.W_o.copy_(output)
            self.bias_o.copy_(output_bias)
            if hasattr(self, "aggregate_gain"):
                self.aggregate_gain.fill_(1)
                self.aggregate_offset.zero_()
        return {"base_input_sha256": _sha(base_input),
                "soma_bias_sha256": _sha(soma_bias),
                "output_sha256": _sha(output),
                "output_bias_sha256": _sha(output_bias),
                "input_scale": scale, "g64_sha256": _sha(self.G64)}

    def _selected(self, x):
        x = x.flatten(1)
        if x.shape[1] != 784:
            raise ValueError("expected 784 inputs")
        return x[:, self.G64.reshape(-1)].reshape(-1, 128, 64)

    def counts(self):
        return {"parameters": sum(p.numel() for p in self.parameters()),
                "input_weights": self.W_in.numel(),
                "main_macs_per_sample": self.W_in.numel() + self.W_o.numel(),
                "fixed_aggregate_additions_per_sample": 0 if not hasattr(self, "branches") else 128 * (self.branches - 1)}


class PointSparse(_CommonControl):
    """One sparse linear soma followed by a single LeakyReLU."""

    def intermediates(self, x):
        pre_s = (self._selected(x) * self.W_in).sum(-1) + self.bias_s
        act_s = F.leaky_relu(pre_s, negative_slope=SLOPE)
        return {"soma_preactivation": pre_s, "soma_activation": act_s,
                "logits": F.linear(act_s, self.W_o, self.bias_o)}

    def forward(self, x):
        return self.intermediates(x)["logits"]


class NeutralCompartments(_CommonControl):
    """Nested G64 partitions, fixed variance-aware branch aggregation."""

    def __init__(self, branches, g64):
        if branches not in BRANCHES:
            raise ValueError("branches must be one of 1,2,4,8")
        self.branches = branches
        super().__init__(g64)
        self.register_buffer("aggregate_gain", torch.ones(()))
        self.register_buffer("aggregate_offset", torch.zeros(()))

    def _raw(self, x):
        k = 64 // self.branches
        selected = self._selected(x).reshape(-1, 128, self.branches, k)
        pre_d = (selected * self.W_in.reshape(128, self.branches, k)).sum(-1)
        act_d = F.leaky_relu(pre_d, negative_slope=SLOPE)
        raw_u = act_d.sum(-1) / math.sqrt(self.branches)
        return pre_d, act_d, raw_u

    @torch.no_grad()
    def calibrate(self, inputs, reference):
        """Label-free train batch; freeze a single global affine correction."""
        if not isinstance(reference, PointSparse) or not torch.equal(self.G64, reference.G64):
            raise ValueError("calibration reference must be PointSparse with the same G64")
        raw_u = self._raw(inputs)[-1]
        target = reference.intermediates(inputs)["soma_preactivation"]
        raw_std = raw_u.std(unbiased=False)
        target_std = target.std(unbiased=False)
        if not bool(torch.isfinite(raw_std) & torch.isfinite(target_std)) or raw_std <= 1e-8:
            raise ValueError("invalid calibration scale")
        gain = target_std / raw_std
        offset = target.mean() - gain * raw_u.mean()
        self.aggregate_gain.copy_(gain)
        self.aggregate_offset.copy_(offset)
        return {"gain": gain.item(), "offset": offset.item(),
                "raw_mean": raw_u.mean().item(), "raw_std": raw_std.item(),
                "target_mean": target.mean().item(), "target_std": target_std.item()}

    def intermediates(self, x):
        pre_d, act_d, raw_u = self._raw(x)
        aggregate = self.aggregate_gain * raw_u + self.aggregate_offset
        act_s = F.leaky_relu(aggregate + self.bias_s, negative_slope=SLOPE)
        return {"dendrite_preactivation": pre_d, "dendrite_activation": act_d,
                "aggregate": aggregate, "soma_activation": act_s,
                "logits": F.linear(act_s, self.W_o, self.bias_o)}

    def forward(self, x):
        return self.intermediates(x)["logits"]


def diagnostics(model, inputs):
    with torch.no_grad():
        values = model.intermediates(inputs)
        logits = values["logits"]
        probabilities = logits.softmax(-1)
        result = {name: {"mean": value.mean().item(), "std": value.std(unbiased=False).item()}
                  for name, value in values.items()}
        result["mean_logit_l2"] = logits.norm(dim=-1).mean().item()
        result["predictive_entropy"] = (-(probabilities * probabilities.log()).sum(-1)).mean().item()
        return result


__all__ = ["PointSparse", "NeutralCompartments", "base_topology", "paired_values", "diagnostics"]
