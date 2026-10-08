"""Protocol checks for the S3-E01 fixed fan-in ablation."""

import torch
from torch.utils.data import RandomSampler

from dsu_research.fixed_fanin_ablation import (
    RecordingSampler,
    build_paired_models,
    classify_h009,
    historical_trial_is_anomalous,
)


def test_paired_models_share_underlying_dense_initialization():
    dann, fixed, digest = build_paired_models(7, 11)
    assert len(digest) == 64
    assert dann.input_mask.sum() == fixed.input_mask.sum() == 8192
    assert dann.cable_mask.sum() == fixed.soma_mask.sum() == 512
    assert sum(parameter.numel() for parameter in dann.parameters()) == 468874
    assert sum(parameter.numel() for parameter in fixed.parameters()) == 468874
    assert dann.effective_parameters() == fixed.effective_parameters() == 10634
    overlap = dann.input_mask & fixed.input_mask
    assert torch.equal(dann.dendrites.weight[overlap], fixed.dendrites.weight[overlap])
    assert torch.equal(dann.dendrites.bias, fixed.dendrites.bias)
    assert torch.equal(dann.somata.bias, fixed.somata.bias)
    assert torch.equal(dann.output.weight, fixed.output.weight)
    assert torch.equal(dann.output.bias, fixed.output.bias)


def test_recording_sampler_reproduces_exact_epoch_orders():
    source_indices = torch.arange(20) * 3
    first_generator = torch.Generator().manual_seed(13)
    second_generator = torch.Generator().manual_seed(13)
    first = RecordingSampler(RandomSampler(range(20), generator=first_generator), source_indices)
    second = RecordingSampler(RandomSampler(range(20), generator=second_generator), source_indices)
    for _ in range(3):
        assert list(first) == list(second)
    assert first.epoch_sha256 == second.epoch_sha256
    assert len(set(first.epoch_sha256)) == 3
    assert first.combined_sha256() == second.combined_sha256()


def test_h009_predefined_classification_boundaries():
    assert classify_h009(0.005, 0.0, 4, 2) == "SUPPORTED"
    assert classify_h009(0.0001, -0.015, 3, 4) == "SUPPORTED"
    assert classify_h009(0.002, 0.0, 3, 2) == "WEAK SUPPORT"
    assert classify_h009(0.0, -0.006, 2, 3) == "WEAK SUPPORT"
    assert classify_h009(0.00149, 0.0049, 2, 2) == "NOT SUPPORTED / NEGLIGIBLE"
    assert classify_h009(-0.005, 0.0, 1, 2) == "CONTRARY"
    assert classify_h009(0.0, 0.015, 2, 1) == "CONTRARY"
    assert classify_h009(0.0016, 0.006, 2, 2) == "INCONCLUSIVE"


def test_historical_sanity_thresholds():
    baseline = {"test": {"accuracy": 0.87, "loss": 0.36}}
    assert not historical_trial_is_anomalous(
        {"test": {"accuracy": 0.851, "loss": 0.311}}, baseline)
    assert historical_trial_is_anomalous(
        {"test": {"accuracy": 0.8499, "loss": 0.36}}, baseline)
    assert historical_trial_is_anomalous(
        {"test": {"accuracy": 0.87, "loss": 0.4101}}, baseline)
