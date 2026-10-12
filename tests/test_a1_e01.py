"""A1-E01 gates: frozen rules and test access after the complete protocol."""

import json

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from dsu_research import a1_e01
from dsu_research import a1_e01_analysis
from dsu_research.a1_e01_analysis import classify


def test_preregistered_thresholds_and_nll_compatibility():
    assert classify([.8, .7, .2], [-.01, -.01, .002])["status"] == "MATERIAL"
    assert classify([.8, .7, .2], [.01, .01, -.002])["status"] == "INCONCLUSIVE"
    assert classify([.4, .35, -.02], [.02, .01, -.001])["status"] == "SMALL_BUT_REAL_SIGNAL"
    assert not classify([.4, .35, -.02], [.02, .01, -.001])["nll_compatible"]
    assert classify([.1, -.1, .1], [.001, -.002, .001])["status"] == "NEGLIGIBLE"


def test_preflight_stops_on_frozen_artifact_drift(monkeypatch):
    monkeypatch.setattr(a1_e01, "FROZEN_SHA256", {
        "data/a1_topology_structural.json": "0" * 64})
    with pytest.raises(ValueError, match="frozen A1 source drift"):
        a1_e01.preflight()


def test_analysis_waits_for_all_nine_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(a1_e01_analysis, "RUNS", tmp_path)
    monkeypatch.setattr(a1_e01_analysis, "preflight", lambda: pytest.fail("opened results before completion"))
    with pytest.raises(FileNotFoundError, match="A1-E01 incomplete"):
        a1_e01_analysis.summarize()


class _TinyTrain(Dataset):
    indices = list(range(4))

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.zeros(784), torch.tensor(index % 10)


def test_test_split_remains_closed_if_pairing_fails(monkeypatch):
    old = json.loads((a1_e01.T07 / "PointSparse_seed1.json").read_text(encoding="utf-8"))
    opened = []

    def fake_load(_config, *, include_test):
        assert include_test is False
        train = DataLoader(_TinyTrain(), batch_size=128, shuffle=True,
                           generator=torch.Generator().manual_seed(1))
        val = DataLoader(_TinyTrain(), batch_size=128, shuffle=False)
        return train, val, None, old["dataset"]

    monkeypatch.setattr(a1_e01, "load_fashion_mnist", fake_load)
    monkeypatch.setattr(a1_e01, "FashionMNIST", lambda **kwargs: opened.append(kwargs))
    with pytest.raises(ValueError, match="minibatch order mismatch"):
        a1_e01.run_one("A1", 1, {"frozen_sha256": a1_e01.FROZEN_SHA256})
    assert opened == []
