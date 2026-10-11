"""Forward-only activation adapters and label-free proxy calculations."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from dsu_research.compartment_ablation import CompartmentalDSU
from dsu_research.dsu_s import DSUSv0
from dsu_research.fixed_dann import FixedDANN
from dsu_research.model import (CapacityMLP, DendriticANNRandom,
                                ParameterMatchedMLP, VanillaANN)
from dsu_research.sparse_compartment_controls import NeutralCompartments, PointSparse

NEAR_ZERO_EPS = 1e-6
RANK_EPS = 1e-12


@dataclass(frozen=True)
class Activations:
    preactivation: torch.Tensor
    activation: torch.Tensor
    logits: torch.Tensor


def collect_activations(model: torch.nn.Module, inputs: torch.Tensor) -> Activations:
    """Select the last hidden representation without changing parameters or grads."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            if isinstance(model, PointSparse):
                values = model.intermediates(inputs)
                result = Activations(values["soma_preactivation"], values["soma_activation"],
                                     values["logits"])
            elif isinstance(model, NeutralCompartments):
                values = model.intermediates(inputs)
                pre = values["aggregate"] + model.bias_s
                result = Activations(pre, values["soma_activation"], values["logits"])
            elif isinstance(model, CompartmentalDSU):
                _, _, pre, act, logits = model.intermediates(inputs)
                result = Activations(pre, act, logits)
            elif isinstance(model, DSUSv0):
                branches = F.leaky_relu(model.dendrite_pre_activations(inputs.flatten(1)), 0.1)
                pre = (branches * model.W_s).sum(-1) + model.bias_s
                act = F.leaky_relu(pre, 0.1)
                result = Activations(pre, act, F.linear(act, model.W_o, model.bias_o))
            elif isinstance(model, FixedDANN):
                pre = model.soma_pre_activations(inputs)
                act = F.leaky_relu(pre, 0.1)
                result = Activations(pre, act, model.output(act))
            elif isinstance(model, DendriticANNRandom):
                x = inputs.flatten(1)
                dendrites = F.leaky_relu(F.linear(x, model.dendrites.weight * model.input_mask,
                                                    model.dendrites.bias), 0.1)
                pre = F.linear(dendrites, model.somata.weight * model.cable_mask,
                               model.somata.bias)
                act = F.leaky_relu(pre, 0.1)
                result = Activations(pre, act, model.output(act))
            elif isinstance(model, VanillaANN):
                dendrites = F.leaky_relu(model.dendrites(inputs.flatten(1)), 0.1)
                pre = model.somata(dendrites)
                act = F.leaky_relu(pre, 0.1)
                result = Activations(pre, act, model.output(act))
            elif isinstance(model, (ParameterMatchedMLP, CapacityMLP)):
                first = F.leaky_relu(model.hidden1(inputs.flatten(1)), 0.1)
                pre = model.hidden2(first)
                act = F.leaky_relu(pre, 0.1)
                result = Activations(pre, act, model.output(act))
            else:
                raise TypeError(f"no activation adapter for {type(model).__name__}")
    finally:
        model.train(was_training)
    if (result.preactivation.ndim != 2 or result.activation.shape != result.preactivation.shape
            or result.logits.shape[0] != result.activation.shape[0]):
        raise ValueError("adapter returned invalid activation shapes")
    return result


def state_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def input_presence(model: torch.nn.Module, input_dim: int = 784) -> torch.Tensor | None:
    """Boolean [soma, input] map; dense models return None."""
    if isinstance(model, (PointSparse, NeutralCompartments)):
        indices = model.G64.reshape(model.G64.shape[0], -1)
    elif isinstance(model, (CompartmentalDSU, DSUSv0)):
        indices = model.G.reshape(128, -1)
    elif isinstance(model, (FixedDANN, DendriticANNRandom)):
        return model.input_mask.reshape(128, 4, input_dim).any(dim=1).detach().cpu()
    else:
        return None
    presence = torch.zeros((indices.shape[0], input_dim), dtype=torch.bool)
    presence.scatter_(1, indices.detach().cpu().to(torch.int64), True)
    return presence


def feature_coverage(presence: torch.Tensor | None, input_dim: int = 784) -> dict:
    if presence is None:
        return {"unique_input_features": input_dim, "coverage_ratio": 1.0,
                "mean_references_per_feature": None, "reference_count_sd": None,
                "max_reference_count": None, "mean_jaccard_between_units": None}
    if presence.ndim != 2 or presence.shape[1] != input_dim or presence.dtype != torch.bool:
        raise ValueError("presence must be bool [units,input_dim]")
    counts = presence.to(torch.float64).sum(0)
    active = int((counts > 0).sum().item())
    common = presence.to(torch.float64) @ presence.to(torch.float64).T
    sizes = presence.sum(1).to(torch.float64)
    union = sizes[:, None] + sizes[None, :] - common
    pairs = torch.triu_indices(len(sizes), len(sizes), offset=1)
    jaccard = torch.where(union[pairs[0], pairs[1]] > 0,
                          common[pairs[0], pairs[1]] / union[pairs[0], pairs[1]].clamp_min(1),
                          torch.zeros_like(common[pairs[0], pairs[1]]))
    return {"unique_input_features": active, "coverage_ratio": active / input_dim,
            "mean_references_per_feature": counts.mean().item(),
            "reference_count_sd": counts.std(unbiased=False).item(),
            "max_reference_count": int(counts.max().item()),
            "mean_jaccard_between_units": jaccard.mean().item() if len(jaccard) else None}


def _scale(values: torch.Tensor) -> dict[str, float]:
    x = values.to(torch.float64)
    return {"mean": x.mean().item(), "std": x.std(unbiased=False).item(),
            "rms": x.square().mean().sqrt().item(), "max_abs": x.abs().max().item()}


def compute_proxies(values: Activations, presence: torch.Tensor | None = None,
                    input_dim: int = 784) -> dict:
    """Pure statistics on detached forward outputs; no labels or model state."""
    if any(t.requires_grad for t in (values.preactivation, values.activation, values.logits)):
        raise ValueError("forward-only proxies require detached activations")
    pre, act, logits = (t.detach().to(torch.float64) for t in
                        (values.preactivation, values.activation, values.logits))
    if act.ndim != 2 or pre.shape != act.shape or logits.ndim != 2 or logits.shape[0] != act.shape[0]:
        raise ValueError("expected pre/activation [N,U] and logits [N,O]")
    n, u = act.shape
    if n < 2 or u < 1:
        raise ValueError("need at least two samples and one unit")
    centered = act - act.mean(0)
    covariance = centered.T @ centered / (n - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    if total <= RANK_EPS:
        effective_rank = participation = 0.0
    else:
        p = eigenvalues / total
        p = p[p > 0]
        effective_rank = torch.exp(-(p * p.log()).sum()).item()
        participation = (total.square() / eigenvalues.square().sum()).item()
    std = covariance.diag().clamp_min(0).sqrt()
    valid = std > RANK_EPS
    valid_count = int(valid.sum().item())
    if valid_count >= 2:
        sub = covariance[valid][:, valid] / (std[valid, None] * std[None, valid])
        offdiag = sub.abs().clamp_max(1)[torch.triu_indices(valid_count, valid_count, 1).unbind()]
        corr_mean = offdiag.mean().item()
        corr_median = torch.quantile(offdiag, 0.5).item()
    else:
        corr_mean = corr_median = None
    signs = (act > 0).to(torch.float64)
    pattern_kernel = (signs @ signs.T + (1 - signs) @ (1 - signs).T) / u
    sign, logabsdet = torch.linalg.slogdet(torch.eye(n, dtype=torch.float64) + pattern_kernel)
    if sign <= 0:
        raise ArithmeticError("activation pattern kernel has nonpositive determinant")
    log_probs = logits.log_softmax(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1).mean().item()
    coverage = feature_coverage(presence, input_dim)
    result = {
        "representation_units": u, "samples": n,
        "effective_rank_raw": effective_rank,
        "effective_rank_norm": effective_rank / min(n, u),
        "participation_ratio_raw": participation,
        "participation_ratio_norm": participation / min(n, u),
        "mean_abs_corr": corr_mean, "median_abs_corr": corr_median,
        "constant_units_excluded": u - valid_count,
        "near_zero_fraction": (act.abs() < NEAR_ZERO_EPS).to(torch.float64).mean().item(),
        "exact_zero_fraction": (act == 0).to(torch.float64).mean().item(),
        "preactivation_scale": _scale(pre), "activation_scale": _scale(act),
        "activation_std": act.std(unbiased=False).item(),
        "logit_mean": logits.mean().item(),
        "logit_std": logits.std(unbiased=False).item(),
        "logit_norm": torch.linalg.vector_norm(logits, dim=-1).mean().item(),
        "entropy": entropy,
        "activation_pattern_logdet": (logabsdet / n).item(),
        **coverage,
        "peak_tensor_size_bytes_estimate": max(n * 784 * 4, pre.numel() * 4,
                                              act.numel() * 4, logits.numel() * 4,
                                              u * u * 8, n * n * 8),
    }
    def check_finite(item):
        if isinstance(item, dict):
            for value in item.values():
                check_finite(value)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ArithmeticError("nonfinite proxy value")
    check_finite(result)
    return result
