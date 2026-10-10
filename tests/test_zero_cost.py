"""Numerical and state invariants for forward-only screening."""

import math

import torch

from dsu_research.analysis.zero_cost import (Activations, collect_activations,
                                             compute_proxies, feature_coverage,
                                             input_presence, state_fingerprint)
from dsu_research.analysis.zero_cost_collection import build_untrained_model
from dsu_research.analysis.zero_cost_validation import kendall_tau_b, pairwise, spearman
from dsu_research.model import ParameterMatchedMLP
from dsu_research.sparse_compartment_controls import PointSparse, base_topology


def _proxies(activation):
    n = activation.shape[0]
    return compute_proxies(Activations(activation, activation,
                                       torch.zeros((n, 2))))


def test_rank_correlation_and_bounds():
    common = torch.tensor([-1., 0., 1., 2.])
    rank_one = _proxies(torch.stack((common, 2 * common), dim=1))
    assert abs(rank_one["effective_rank_raw"] - 1) < 1e-10
    assert abs(rank_one["participation_ratio_raw"] - 1) < 1e-10
    assert abs(rank_one["mean_abs_corr"] - 1) < 1e-10
    orthogonal = _proxies(torch.tensor([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]]))
    assert abs(orthogonal["effective_rank_raw"] - 2) < 1e-10
    assert abs(orthogonal["participation_ratio_raw"] - 2) < 1e-10
    assert abs(orthogonal["mean_abs_corr"]) < 1e-10
    for result in (rank_one, orthogonal):
        assert 0 <= result["effective_rank_norm"] <= 1
        assert 0 <= result["participation_ratio_norm"] <= 1
        assert math.isfinite(result["activation_pattern_logdet"])


def test_coverage_and_constant_units():
    presence = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=torch.bool)
    coverage = feature_coverage(presence, 4)
    assert coverage["unique_input_features"] == 3
    assert coverage["coverage_ratio"] == 0.75
    assert abs(coverage["mean_jaccard_between_units"] - 1 / 3) < 1e-12
    result = _proxies(torch.ones((4, 2)))
    assert result["constant_units_excluded"] == 2
    assert result["effective_rank_raw"] == 0
    assert result["mean_abs_corr"] is None


def test_adapter_is_deterministic_and_does_not_mutate_or_create_gradients():
    x = torch.linspace(0, 1, 4 * 784).reshape(4, 784)
    for model in (ParameterMatchedMLP(), PointSparse(base_topology(1))):
        before = state_fingerprint(model)
        first = collect_activations(model, x)
        second = collect_activations(model, x)
        assert torch.equal(first.logits, second.logits)
        assert torch.allclose(first.logits, model(x))
        assert state_fingerprint(model) == before
        assert all(p.grad is None for p in model.parameters())
        assert not first.activation.requires_grad
        assert compute_proxies(first, input_presence(model)) == compute_proxies(second, input_presence(model))


def test_frozen_initialization_repeats():
    inputs = torch.linspace(0, 1, 8 * 784).reshape(8, 784)
    model_a, passes_a = build_untrained_model("s3_t07", "2x32", 2, inputs)
    model_b, passes_b = build_untrained_model("s3_t07", "2x32", 2, inputs)
    assert passes_a == passes_b == 3
    assert state_fingerprint(model_a) == state_fingerprint(model_b)
    assert compute_proxies(collect_activations(model_a, inputs), input_presence(model_a)) == \
           compute_proxies(collect_activations(model_b, inputs), input_presence(model_b))


def test_rank_correlations_and_near_ties():
    assert spearman([1, 2, 3], [3, 2, 1]) == -1
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1
    assert kendall_tau_b([1, 2], [1, 2]) is None
    result = pairwise([1, 2, 3], [1, 1.0005, 3], 0.001)
    assert result["near_tie_pairs"] == 1
    assert result["scored_pairs"] == 2
    assert result["accuracy_excluding_near_ties"] == 1
    assert result["accuracy_including_near_ties"] == 2.5 / 3
