"""Structural and calibration contracts for S3-T06."""

import pytest
import torch

from dsu_research.sparse_compartment_controls import (
    NeutralCompartments, PointSparse, base_topology, diagnostics, paired_values,
)


def test_all_arms_share_topology_banks_and_counts():
    g64 = base_topology(3)
    values = paired_values(7)
    models = [PointSparse(g64)] + [NeutralCompartments(b, g64) for b in (1, 2, 4, 8)]
    metadata = [model.initialize(*values) for model in models]
    assert len({m["base_input_sha256"] for m in metadata}) == 1
    assert len({m["output_sha256"] for m in metadata}) == 1
    assert all(torch.equal(model.G64, g64) for model in models)
    assert all(torch.unique(row).numel() == 64 for row in g64)
    assert all(model.counts()["parameters"] == 9610 for model in models)
    assert all(model.counts()["input_weights"] == 8192 for model in models)
    assert all(model.counts()["main_macs_per_sample"] == 9472 for model in models)
    assert all(torch.equal(model.W_o, models[0].W_o) for model in models)
    assert all(set(dict(model.named_parameters())) == {"W_in", "bias_s", "W_o", "bias_o"} for model in models)
    for model in models:
        x = torch.randn(3, 784, requires_grad=True)
        assert model(x).shape == (3, 10)
        model(x).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert torch.isfinite(x.grad).all()
    for coarse, fine in zip(models[1:], models[2:]):
        factor = fine.branches // coarse.branches
        assert torch.equal(coarse.G64.reshape(128, coarse.branches, 64 // coarse.branches),
                           fine.G64.reshape(128, coarse.branches, factor, 64 // fine.branches).flatten(-2))


def test_calibration_is_deterministic_and_label_free():
    g64 = base_topology(1)
    point = PointSparse(g64)
    family = [NeutralCompartments(b, g64) for b in (1, 2, 4, 8)]
    values = paired_values(1)
    point.initialize(*values)
    x = torch.rand((64, 784), generator=torch.Generator().manual_seed(13))
    for model in family:
        model.initialize(*values)
        first = model.calibrate(x, point)
        second = model.calibrate(x, point)
        assert first == second
        assert model.aggregate_gain.requires_grad is False
        assert model.aggregate_offset.requires_grad is False
        assert diagnostics(model, x)["logits"]["std"] > 0
        assert torch.allclose(model.intermediates(x)["aggregate"].std(unbiased=False),
                              point.intermediates(x)["soma_preactivation"].std(unbiased=False))


def test_invalid_topology_rejected():
    bad = base_topology(1)
    bad[0, 1] = bad[0, 0]
    with pytest.raises(ValueError):
        PointSparse(bad)
    with pytest.raises(ValueError, match="same G64"):
        NeutralCompartments(2, base_topology(2)).calibrate(
            torch.rand(4, 784), PointSparse(base_topology(3)))
