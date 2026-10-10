"""Metric and frozen-calibration checks for S3-T07."""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dsu_research.s3_t06_diagnostics import train_calibration_batch
from dsu_research.sparse_compartment_experiment import build_arm, diagnostic_evaluate


class FixedLogits(torch.nn.Module):
    def forward(self, x):
        return x


def test_test_diagnostics_perfect_predictions_and_ece():
    logits = torch.tensor([[2.0] + [0.0] * 9]).repeat(10000, 1)
    targets = torch.zeros(10000, dtype=torch.long)
    metrics = diagnostic_evaluate(FixedLogits(), DataLoader(TensorDataset(logits, targets), batch_size=128))
    confidence = logits[0].softmax(0)[0].item()
    assert metrics["accuracy"] == 1.0
    assert metrics["fraction_positive_margins"] == 1.0
    assert metrics["correct_class_margin"] == pytest.approx(2.0)
    assert metrics["ece_15"] == pytest.approx(1 - confidence, abs=1e-7)


@pytest.mark.skipif(not Path("runs/s3_t06_initial_diagnostics.json").exists(), reason="local dataset diagnostic absent")
def test_frozen_calibration_reproduces_s3_t06():
    frozen = json.loads(Path("runs/s3_t06_initial_diagnostics.json").read_text(encoding="utf-8"))
    for record in frozen["records"]:
        inputs, _ = train_calibration_batch(record["seed"])
        model, initialization, calibration = build_arm(record["seed"], record["arm"], inputs)
        assert initialization == record["initialization"]
        assert calibration == record.get("calibration")
        assert model.counts()["parameters"] == 9610
        assert model.counts()["main_macs_per_sample"] == 9472
