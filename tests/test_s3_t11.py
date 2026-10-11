"""Reuse, commit gate, and test-access sequencing for the frozen shortlist."""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dsu_research import s3_t11
from dsu_research import s3_t11_analysis


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


@pytest.mark.skipif(not Path("runs/sparse_compartment_s3_t07/PointSparse_seed1.json").exists(),
                    reason="local historical curves absent")
def test_historical_analysis_keeps_families_separate():
    prereg, _, _ = s3_t11.sealed_prereg()
    groups = s3_t11_analysis.historical_analysis(prereg)
    assert set(groups) == set(s3_t11.HISTORICAL_GROUPS)
    assert len(groups["s3_t07_parameter_neutral"]["architecture_ids"]) == 5
    assert len(groups["s3_t05_original_compartments"]["architecture_ids"]) == 4
    assert len(groups["s2_t06_mlp_capacity"]["architecture_ids"]) == 6
    assert groups["s2_t06_mlp_capacity"]["test_comparable"] is False
    assert all(set(group["epochs"]) == {"1", "2", "3", "5"} for group in groups.values())


def test_early_classifier_requires_multiple_groups():
    def group(rho, pair):
        return {"epochs": {"3": {"vs_final": {"final_validation_accuracy":
            {"spearman": rho, "pairwise": {"accuracy_excluding_near_ties": pair}}}}}}
    assert s3_t11_analysis.classify_early({"a": group(.8, .8), "b": group(.6, .75),
                                           "c": group(.1, .5)}) == "PROMISING"
    assert s3_t11_analysis.classify_early({"a": group(.8, .8), "b": group(.4, .6),
                                           "c": group(-.2, .4)}) == "WEAK"
    assert s3_t11_analysis.classify_early({"a": group(.8, .8), "b": group(.7, .8),
                                           "c": group(-.7, .2)}) == "FAILED"


def test_width_classifier_uses_validation_test_and_paired_seeds():
    def arm(val, test, loss):
        return {"validation_accuracy": {"mean": sum(val) / 3, "by_seed": val},
                "test_accuracy": {"mean": sum(test) / 3, "by_seed": test},
                "test_loss": {"mean": sum(loss) / 3, "by_seed": loss}}
    baseline = arm([.86] * 3, [.85] * 3, [.4] * 3)
    strong = arm([.866] * 3, [.857] * 3, [.39] * 3)
    flat = arm([.861] * 3, [.851] * 3, [.401] * 3)
    status, contrasts = s3_t11_analysis.classify_width({
        "PointSparse_128x64": baseline, "PointSparse_80x109": strong,
        "PointSparse_160x49": flat})
    assert status == "SUPPORTED_WITHIN_TESTED_RANGE"
    assert contrasts["PointSparse_80x109"]["strong_rule"] is True
