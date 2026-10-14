"""Pretraining checks for the frozen B1-E02 NOISE intervention."""

import numpy as np
import json
import pytest
import torch

from dsu_research.b1_e01 import (
    D, LAMBDA, PointSparseTemporal, initial_values, make_split,
    sequence_batch, topology,
)
from dsu_research.b1_e02 import (
    make_noise_split, noise_sequence_batch, noise_split_fingerprint,
)
from dsu_research.b1_e02_run import (
    build, frozen_protocol, order_hashes, summarize, training_order,
)
from dsu_research import b1_e02_run


def test_noise_is_preregistered_deterministic_and_class_independent():
    first, labels, noise = make_noise_split("validation", 16)
    again = make_noise_split("validation", 16)
    different = make_noise_split("validation", 16, dataset_seed=41042)
    easy_first, easy_labels = make_split("validation", 16)
    assert labels.bincount().tolist() == [5000, 5000]
    assert torch.equal(first, easy_first) and torch.equal(labels, easy_labels)
    assert all(torch.equal(a, b) for a, b in zip((first, labels, noise), again))
    assert not torch.equal(noise, different[2])
    assert noise.shape == (10000, 15, 31)
    assert noise.min() >= -.1 and noise.max() <= .1
    for index in (0, 1, 137):
        expected = np.random.default_rng(np.random.SeedSequence(
            [41041, 1, 16, index, 99])).uniform(-.1, .1, size=(15, 31)).astype(np.float32)
        assert np.array_equal(noise[index].numpy(), expected)
    assert (noise[labels == 0].mean((0, 1)) - noise[labels == 1].mean((0, 1))).abs().max() < .005
    assert noise[:, :, :2].count_nonzero() > 0
    seq = noise_sequence_batch(first[:128], noise[:128], 16)
    easy = sequence_batch(first[:128], 16)
    assert seq.shape == (128, 17, D)
    assert torch.equal(seq[:, 0], easy[:, 0])
    assert torch.equal(seq[:, -1], easy[:, -1])
    assert seq[:, 1:-1, 2].count_nonzero() == 0
    assert torch.equal(seq[:, -1], seq[0, -1].expand_as(seq[:, -1]))
    assert noise_split_fingerprint(first, labels, noise, "validation", 16) != noise_split_fingerprint(*different, "validation", 16, 41042)


def test_state_reset_gradient_and_frozen_pair():
    protocol = frozen_protocol()
    assert protocol["lambda_double"] == pytest.approx(LAMBDA, abs=1e-15)
    static, stateful = build("S0", 1), build("S1", 1)
    assert static.decay == stateful.decay == float(np.float32(LAMBDA))
    assert torch.equal(static.G, stateful.G)
    assert static.cost(16)["trainable_parameters"] == stateful.cost(16)["trainable_parameters"] == 706
    for name in ("W_in", "b_s", "W_out", "b_out"):
        assert torch.equal(getattr(static, name), getattr(stateful, name))
    first = torch.zeros((2, D))
    first[0, 0], first[1, 1] = 1, 1
    noise = torch.ones((2, 15, 31)) * .04
    seq = noise_sequence_batch(first, noise, 16).requires_grad_()
    assert torch.equal(static(seq)[0], static(seq)[1])
    assert not torch.equal(stateful(seq)[0], stateful(seq)[1])
    assert torch.autograd.grad(stateful(seq)[0, 0], seq)[0][0, 0].abs().sum() > 0
    assert torch.autograd.grad(static(seq)[0, 0], seq)[0][:, 0].count_nonzero() == 0
    reference = stateful(seq.flip(0))
    _ = stateful(seq)
    assert torch.equal(stateful(seq.flip(0)), reference)
    assert order_hashes(1, 50000) == order_hashes(1, 50000)
    assert not torch.equal(training_order(1, 1, 128), training_order(2, 1, 128))


def test_preregistered_decision_and_pairing():
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


def test_resume_uses_existing_valid_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(b1_e02_run, "RUNS", tmp_path)
    record = {"schema": "saved"}
    b1_e02_run.result_path("S0", 4, 1).write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(b1_e02_run, "validate_record", lambda found, *_: found == record or 1 / 0)
    monkeypatch.setattr(b1_e02_run, "run_one", lambda *_: 1 / 0)
    assert b1_e02_run.load_or_run("S0", 4, 1, {}, {}) == record
