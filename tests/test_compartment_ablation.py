"""Structural and methodological checks for S3-T05."""

import pytest
import torch

from dsu_research.compartment_ablation import (
    BRANCHES, CompartmentalDSU, base_topology, initialize_paired,
    shared_initialization, summarize,
)


def test_nested_topology_counts_forward_and_backward():
    g64 = base_topology(7)
    assert torch.equal(g64, base_topology(7))
    assert not torch.equal(g64, base_topology(8))
    assert g64.shape == (128, 64)
    assert torch.all((g64 >= 0) & (g64 < 784))
    assert all(torch.unique(row).numel() == 64 for row in g64)
    models = [CompartmentalDSU(b, g64) for b in BRANCHES]
    for b, model in zip(BRANCHES, models):
        assert b * model.fan_in == 64
        assert torch.equal(model.G.reshape(128, 64), g64)
        assert model.counts() == {"parameters": 9610 + 256 * b,
                                  "main_macs_per_sample": 9472 + 128 * b,
                                  "input_connections": 8192}
        assert list(model.buffers()) == [model.G]
        assert [tuple(p.shape) for p in model.parameters()] == [
            (128, b, 64 // b), (128, b), (128, b), (128,), (10, 128), (10,)]
        x = torch.randn(3, 784, requires_grad=True)
        logits = model(x)
        assert logits.shape == (3, 10)
        logits.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert x.grad is not None and torch.isfinite(x.grad).all()
    for coarse, fine in zip(models, models[1:]):
        factor = fine.branches // coarse.branches
        assert torch.equal(coarse.G, fine.G.reshape(128, coarse.branches, factor, fine.fan_in).flatten(-2))


def test_paired_initialization_and_zero_biases():
    g64 = base_topology(2)
    base = shared_initialization(3)
    models = [CompartmentalDSU(b, g64) for b in BRANCHES]
    meta = [initialize_paired(model, *base) for model in models]
    assert len({item["base_input_sha256"] for item in meta}) == 1
    assert len({item["base_soma8_sha256"] for item in meta}) == 1
    assert len({item["output_sha256"] for item in meta}) == 1
    for model in models:
        assert torch.equal(model.W_o, models[0].W_o)
        assert torch.equal(model.bias_d, torch.zeros_like(model.bias_d))
        assert torch.equal(model.bias_s, torch.zeros_like(model.bias_s))
        assert torch.equal(model.bias_o, torch.zeros_like(model.bias_o))
        assert torch.allclose(model.W_d.flatten() / (2 / 1.01 / model.fan_in) ** 0.5,
                              base[0].flatten())


def test_reject_duplicate_inputs_within_soma():
    g64 = base_topology(1)
    g64[0, 1] = g64[0, 0]
    with pytest.raises(ValueError):
        CompartmentalDSU(4, g64)


def test_summary_rejects_unpaired_batch_order():
    records = []
    for seed in (1, 2, 3):
        for b in BRANCHES:
            records.append({
                "branches": b, "config": {"seed": seed},
                "topology": {"g64_sha256": str(seed)},
                "initialization": {name: str(seed) for name in
                                   ("base_input_sha256", "base_soma8_sha256", "output_sha256")},
                "batch_order": {"combined_sha256": str(seed)},
                "dataset": {"train_indices_sha256": str(seed),
                            "validation_indices_sha256": str(seed)},
                "counts": {"parameters": 9610 + 256 * b,
                           "main_macs_per_sample": 9472 + 128 * b},
                "test": {"loss": 0.4, "accuracy": 0.85},
                "training": {"history": [{"validation": {"loss": 0.35, "accuracy": 0.87}}]},
            })
    records[1]["batch_order"]["combined_sha256"] = "different"
    with pytest.raises(ValueError, match="unpaired minibatches"):
        summarize(records)
