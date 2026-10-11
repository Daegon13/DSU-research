"""Reuse, commit gate, and test-access sequencing for the frozen shortlist."""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dsu_research import s3_t11


@pytest.mark.skipif(not Path("runs/sparse_compartment_s3_t07/PointSparse_seed1.json").exists(),
                    reason="local historical S3-T07 artifacts absent")
def test_anchor_reuse_audit_is_objective():
    result = s3_t11.audit_anchor()
    assert result["decision"] == "REUSE_S3_T07_N3"
    assert [row["seed"] for row in result["seeds"]] == [1, 2, 3]
    assert len({row["state_initial_sha256"] for row in result["seeds"]}) == 3


def test_training_requires_committed_prereg(monkeypatch, tmp_path):
    path = tmp_path / "prereg.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(s3_t11, "PREREG", path)
    monkeypatch.setattr(s3_t11, "_git", lambda *args: "")
    with pytest.raises(RuntimeError, match="committed"):
        s3_t11.train()


def test_test_dataset_opens_only_after_epoch_25(monkeypatch):
    data = TensorDataset(torch.rand(8, 784), torch.zeros(8, dtype=torch.long))
    data.indices = list(range(8))
    loader = DataLoader(data, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(1))
    val = DataLoader(data, batch_size=4)
    epoch_evaluations = []
    test_opened = []

    def fake_loader(config, *, include_test=True):
        assert include_test is False
        return loader, val, None, {"splits": {"train": 8, "validation": 8, "test": 10000}}

    def fake_eval(model, validation, device):
        epoch_evaluations.append(1)
        return {"loss": 1., "accuracy": .5}

    def fake_test(*args, **kwargs):
        test_opened.append(len(epoch_evaluations))
        return data

    monkeypatch.setattr(s3_t11, "load_fashion_mnist", fake_loader)
    monkeypatch.setattr(s3_t11, "evaluate", fake_eval)
    monkeypatch.setattr(s3_t11, "FashionMNIST", fake_test)
    monkeypatch.setattr(s3_t11, "diagnostic_evaluate", lambda model, test: {"accuracy": .5})
    result = s3_t11.run_one(80, 109, 1, "sha", "commit")
    assert len(result["history"]) == 25
    assert test_opened == [25]
    assert result["checkpoint_selected"] == "final_epoch_25"


def test_resume_does_not_repeat_valid_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(s3_t11, "RUNS", tmp_path)
    monkeypatch.setattr(s3_t11, "sealed_prereg", lambda: ({"anchor_audit": {"decision": "REUSE_S3_T07_N3"}}, "sha", "commit"))
    monkeypatch.setattr(s3_t11, "audit_anchor", lambda: {"decision": "REUSE_S3_T07_N3"})
    validated = []
    monkeypatch.setattr(s3_t11, "_validate_new", lambda record, h, k, seed, *args: validated.append((h, k, seed)))
    monkeypatch.setattr(s3_t11, "run_one", lambda *args: pytest.fail("resumed run was repeated"))
    for h, k in s3_t11.SHORTLIST[:2]:
        for seed in s3_t11.SEEDS:
            (tmp_path / f"PointSparse_{h}x{k}_seed{seed}.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert len(s3_t11.train()) == 6
    assert len(validated) == 6
