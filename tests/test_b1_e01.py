"""Pretraining gates for B1-E01's frozen temporal contrast."""

import math

import pytest
import torch

from dsu_research.b1_e01 import (
    D, H, K, LAMBDA, PointSparseTemporal, initial_values, make_split,
    sequence_batch, split_fingerprint, topology,
)
from dsu_research.b1_e01_run import summarize


def test_dataset_balance_determinism_and_no_final_step_leakage():
    first, labels = make_split("validation", 16)
    again, labels_again = make_split("validation", 16)
    different, _ = make_split("validation", 16, dataset_seed=41042)
    train, train_labels = make_split("train", 16)
    assert first.shape == (10000, D)
    assert labels.bincount().tolist() == [5000, 5000]
    assert train_labels.bincount().tolist() == [25000, 25000]
    assert torch.equal(first, again) and torch.equal(labels, labels_again)
    assert not torch.equal(first, different)
    assert split_fingerprint(first, labels, "validation", 16) != split_fingerprint(train, train_labels, "train", 16)
    assert torch.equal(first[:, :2], torch.nn.functional.one_hot(labels, num_classes=2).float())
    nuisance = first[:, 3:]
    assert nuisance.min() >= -0.1 and nuisance.max() <= 0.1
    assert (nuisance[labels == 0].mean(0) - nuisance[labels == 1].mean(0)).abs().max() < .005
    seq = sequence_batch(first[:128], 16)
    assert seq.shape == (128, 17, D)
    assert seq[:, 1:-1].count_nonzero() == 0
    assert seq[:, -1, :2].count_nonzero() == 0
    assert seq[:, -1, 2].eq(1).all()
    assert seq[:, -1, 3:].count_nonzero() == 0
    assert torch.equal(seq[:, -1], seq[0, -1].expand_as(seq[:, -1]))


def test_pair_counts_initialization_and_temporal_paths():
    graph, values = topology(1), initial_values(1)
    static = PointSparseTemporal(graph, values, False)
    stateful = PointSparseTemporal(graph, values, True)
    assert graph.shape == (H, K)
    assert set(graph.flatten().tolist()) >= {0, 1}
    assert all(torch.unique(row).numel() == K for row in graph)
    assert static.cost(16)["trainable_parameters"] == stateful.cost(16)["trainable_parameters"] == 706
    assert stateful.cost(16)["runtime_state_bytes_per_sample"] == 256
    for name in ("W_in", "b_s", "W_out", "b_out"):
        assert torch.equal(getattr(static, name), getattr(stateful, name))
    first = torch.zeros((2, D))
    first[0, 0], first[1, 1] = 1, 1
    seq = sequence_batch(first, 4)
    assert torch.equal(static.integrate(seq[:, 0]), stateful.integrate(seq[:, 0]))
    assert torch.equal(static(seq), static(seq.flip(0)))  # x_T identical
    assert (stateful(seq)[0] - stateful(seq)[1]).abs().max() > 0
    seq.requires_grad_()
    stateful(seq)[0, 0].backward()
    assert seq.grad[0, 0].abs().sum() > 0
    assert seq.grad[:, 1:-1].abs().sum() >= 0
    assert static(seq).grad_fn is not None
    static_gradient = torch.autograd.grad(static(seq)[0, 0], seq)[0]
    assert static_gradient[:, 0].count_nonzero() == 0


def test_reset_and_analytic_memory_curve():
    model = PointSparseTemporal(topology(2), initial_values(2), True)
    a = sequence_batch(torch.nn.functional.one_hot(torch.tensor([0, 1]), num_classes=D).float(), 8)
    b = sequence_batch(torch.nn.functional.one_hot(torch.tensor([1, 0]), num_classes=D).float(), 8)
    reference = model(b)
    _ = model(a)
    assert torch.equal(model(b), reference)
    previous = torch.zeros((1, H), dtype=torch.float64)
    impulse = torch.ones_like(previous)
    for delay in (4, 8, 16, 32):
        state = model.state_step(previous, impulse)
        written = state.clone()
        for _ in range(delay):
            state = model.state_step(state, previous)
        measured = (state / written).mean().item()
        assert measured == pytest.approx(model.decay ** delay, abs=1e-12)
        assert measured == pytest.approx(2 ** (-delay / 16), abs=1e-6)
        assert math.isfinite(measured)


def test_frozen_decision_requires_paired_endpoint_and_oracle_gate():
    records = []
    for delay in (4, 8, 16, 32):
        for seed in (1, 2, 3):
            for arm in ("S0", "S1"):
                records.append({"arm": arm, "delay": delay, "seed": seed,
                    "topology": {"sha256": str(seed)},
                    "initialization_sha256": {"all": str(seed)},
                    "training": {"batch_order_sha256": [str(seed)], "duration_seconds": 1},
                    "dataset": {"fingerprint": str(delay)},
                    "test": {"accuracy": .5 if arm == "S0" else .8,
                             "loss": .693 if arm == "S0" else .3}})
    assert summarize(records)["status"] == "SUPPORTED"
    for row in records:
        if row["arm"] == "S1" and row["delay"] == 16:
            row["test"] = {"accuracy": .5, "loss": .693}
    assert summarize(records)["status"] == "INCONCLUSIVE"
    oracle = [{"test": {"accuracy": .95}} for _ in range(3)]
    assert summarize(records, oracle)["status"] == "NOT SUPPORTED"
    records[-1]["dataset"] = {"fingerprint": "broken-pair"}
    with pytest.raises(ValueError, match="pairing failed"):
        summarize(records, oracle)
