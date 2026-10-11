"""Frozen cohort, test firewall, and resume integrity for S3-T10."""

import json

import pytest
import torch

from dsu_research.analysis.structural_cost import ArchitectureSpec, calculate_cost
from dsu_research.s3_t10 import (COHORT, _validate_pilot, architecture_id, build_model,
                                pilot, structural, topology)


@pytest.mark.parametrize("h,k", COHORT)
def test_iso_budget_and_macs(h, k):
    assert h * (k + 11) + 10 == 9610
    cost = structural(h, k)
    assert cost["trainable_parameters"] == 9610
    assert cost["effective_connections"] == h * (k + 10)
    assert cost["main_macs_per_sample"] == h * (k + 10)
    assert cost["topology_bytes"] == 8 * h * k
    assert sum(p.numel() for p in build_model(h, k, 1, 1).parameters()) == 9610


def test_topology_deterministic_unique_and_seed_separation():
    for h, k in COHORT:
        a = topology(h, k, 1)
        assert torch.equal(a, topology(h, k, 1))
        assert not torch.equal(a, topology(h, k, 2))
        assert all(len(torch.unique(row)) == k for row in a)
        x = build_model(h, k, 1, 1)
        y = build_model(h, k, 2, 1)
        assert torch.equal(x.G64, y.G64)
        assert not torch.equal(x.W_in, y.W_in)


def test_pilot_uses_only_train_and_validation(monkeypatch, tmp_path):
    from dsu_research import s3_t10
    from torch.utils.data import DataLoader, TensorDataset

    calls = []
    dataset = TensorDataset(torch.rand(16, 784), torch.zeros(16, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(1))
    loader.dataset.indices = list(range(16))  # source order for the recording sampler
    validation = DataLoader(dataset, batch_size=8)

    def fake_loader(config, *, include_test=True):
        calls.append(include_test)
        assert include_test is False
        return loader, validation, None, {"splits": {"train": 16, "validation": 16, "test": 10000}}

    monkeypatch.setattr(s3_t10, "load_fashion_mnist", fake_loader)
    result = s3_t10.run_pilot(64, 139, "sha", "commit")
    assert calls == [False]
    assert result["test_access"] is False
    assert "test" not in result
    assert len(result["history"]) == 3


def test_resume_skips_valid_and_rejects_tamper(monkeypatch, tmp_path):
    from dsu_research import s3_t10
    import hashlib

    raw = b'{"schema":"s3_t10_prereg_v1","primary_proxy":"effective_rank_raw"}'
    prereg_sha = hashlib.sha256(raw).hexdigest()
    rows = [{"architecture_id": architecture_id(h, k), "H": h, "K": k,
             "structural_cost": structural(h, k),
             "proxy_seeds": [{"topology_sha256": "top", "initialization_sha256": "init"}]}
            for h, k in COHORT]
    prereg = {"architectures": rows}
    monkeypatch.setattr(s3_t10, "sealed_prereg", lambda: (prereg, prereg_sha, "commit"))
    monkeypatch.setattr(s3_t10, "RUNS", tmp_path)
    calls = []

    def fake_run(h, k, sha, commit):
        calls.append((h, k))
        return {"schema": "s3_t10_pilot_v1", "architecture_id": architecture_id(h, k),
                "H": h, "K": k, "prereg_sha256": sha, "prereg_commit": commit,
                "test_access": False, "structural_cost": structural(h, k),
                "topology_sha256": "top", "initialization_sha256": "init",
                "batch_order": ["a", "b", "c"], "history": [
                    {"epoch": e, "train_loss": 1.0,
                     "validation": {"loss": 1.0, "accuracy": .5}} for e in (1, 2, 3)]}

    monkeypatch.setattr(s3_t10, "run_pilot", fake_run)
    pilot()
    assert len(calls) == 8
    pilot()
    assert len(calls) == 8
    path = tmp_path / f"{architecture_id(*COHORT[0])}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["test"] = {"accuracy": 1}
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid resume"):
        pilot()


def test_prereg_must_be_committed_before_pilot(monkeypatch, tmp_path):
    from dsu_research import s3_t10
    path = tmp_path / "prereg.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(s3_t10, "PREREG", path)
    monkeypatch.setattr(s3_t10, "_git", lambda *args: "prereg_commit")
    monkeypatch.setattr(s3_t10.subprocess, "check_output", lambda *a, **k: b"different")
    with pytest.raises(RuntimeError, match="committed unchanged"):
        s3_t10.sealed_prereg()
