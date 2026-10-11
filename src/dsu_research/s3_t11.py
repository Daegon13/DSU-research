"""S3-T11 frozen PointSparse shortlist confirmation and early-pilot audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from .analysis.zero_cost import state_fingerprint
from .analysis.zero_cost_validation import kendall_tau_b, pairwise, spearman
from .compartment_ablation import sha
from .config import ExperimentConfig
from .data import load_fashion_mnist
from .fixed_fanin_ablation import RecordingSampler
from .harness import evaluate, seed_everything
from .s3_t06_diagnostics import train_calibration_batch
from .s3_t10 import build_model, structural
from .sparse_compartment_experiment import build_arm, diagnostic_evaluate, validate_record

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "data" / "s3_t11_preregistered_confirmation.json"
RESULTS = ROOT / "data" / "s3_t11_confirmation_results.json"
RUNS = ROOT / "runs" / "s3_t11_point_sparse_shortlist"
T07 = ROOT / "runs" / "sparse_compartment_s3_t07"
T10 = ROOT / "data" / "s3_t10_pilot_results.json"
SHORTLIST = ((80, 109), (160, 49), (128, 64))
SEEDS = (1, 2, 3)
EPOCHS_EARLY = (1, 2, 3, 5)
NEAR_TIE_ACC = .002
HISTORICAL_GROUPS = {
    "s3_t07_parameter_neutral": ("runs/sparse_compartment_s3_t07", ("PointSparse", "1x64", "2x32", "4x16", "8x8")),
    "s3_t05_original_compartments": ("runs/compartment_s3_t05", ("1x64", "2x32", "4x16", "8x8")),
    "s2_t06_mlp_capacity": ("runs/mlp_capacity_s2_t06", ("h16", "h19", "h20", "h21", "h22", "h25")),
}


def _bytes(path):
    return Path(path).read_bytes()


def _sha(path):
    return hashlib.sha256(_bytes(path)).hexdigest()


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def config(seed):
    return ExperimentConfig(model="s3_t11_point_sparse", dataset="fashion_mnist", seed=seed,
                            topology_seed=seed, epochs=25, batch_size=128,
                            learning_rate=.001, device="cpu")


def audit_anchor():
    """Objective equivalence gate; validates the old records before reuse."""
    frozen = json.loads((ROOT / "runs" / "s3_t06_initial_diagnostics.json").read_text(encoding="utf-8"))
    t10 = json.loads(T10.read_text(encoding="utf-8"))
    proxy_rows = json.loads((ROOT / "data" / "s3_t10_preregistered_screen.json").read_text(encoding="utf-8"))["architectures"]
    anchor_proxy = next(row for row in proxy_rows if (row["H"], row["K"]) == (128, 64))
    pilot = next(row for row in t10["pilot_records"] if row["H"] == 128)
    audit = []
    for seed in SEEDS:
        path = T07 / f"PointSparse_seed{seed}.json"
        old = json.loads(path.read_text(encoding="utf-8"))
        validate_record(old, seed, "PointSparse", frozen)
        old_model, initialization, calibration = build_arm(seed, "PointSparse", train_calibration_batch(seed)[0])
        new_model = build_model(128, 64, seed, seed)
        state_sha = state_fingerprint(new_model)
        old_config = old["config"]
        training = old["training"]
        proxy = anchor_proxy["proxy_seeds"][seed - 1]
        if (type(old_model) is not type(new_model) or
            state_fingerprint(old_model) != state_sha or
            proxy["initialization_sha256"] != state_sha or
            proxy["topology_sha256"] != sha(new_model.G64) or
            old["topology"]["g64_sha256"] != sha(new_model.G64) or
            initialization != old["initialization"] or calibration is not None or
            old_config["seed"] != seed or old_config["topology_seed"] != seed or
            old_config["dataset"] != "fashion_mnist" or old_config["epochs"] != 25 or
            old_config["batch_size"] != 128 or old_config["learning_rate"] != .001 or
            old_config["device"] != "cpu" or old_config["train_samples"] is not None or
            old_config["test_samples"] is not None or
            old["counts"]["parameters"] != 9610 or old["counts"]["main_macs_per_sample"] != 9472 or
            old["dataset"]["splits"] != {"train": 54000, "validation": 6000, "test": 10000} or
            old["dataset"]["transform"] != "ToTensor() scales uint8 pixels to float32 by 255; no other normalization" or
            training["optimizer"] != "Adam" or training["betas"] != [.9, .999] or
            training["epsilon"] != 1e-7 or training["weight_decay"] != 0 or
            training["checkpoint_selected"] != "final_epoch_25" or
            len(training["history"]) != 25 or len(old["batch_order"]["epoch_sha256"]) != 25):
            raise ValueError(f"anchor protocol differs in seed {seed}")
        if seed == 1:
            if (pilot["initialization_sha256"] != state_sha or
                pilot["topology_sha256"] != sha(new_model.G64) or
                pilot["batch_order"] != old["batch_order"]["epoch_sha256"][:3] or
                any(pilot["dataset"][key] != old["dataset"][key] for key in
                    ("train_indices_sha256", "validation_indices_sha256")) or
                any(pilot["history"][i]["validation"]["accuracy"] !=
                    training["history"][i]["validation"]["accuracy"] for i in range(3))):
                raise ValueError("anchor pilot and historical first three epochs differ")
        audit.append({"seed": seed, "historical_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                      "historical_file_sha256": _sha(path), "state_initial_sha256": state_sha,
                      "topology_sha256": sha(new_model.G64),
                      "train_indices_sha256": old["dataset"]["train_indices_sha256"],
                      "validation_indices_sha256": old["dataset"]["validation_indices_sha256"],
                      "minibatch_order_combined_sha256": old["batch_order"]["combined_sha256"],
                      "epoch_3_validation_accuracy": training["history"][2]["validation"]["accuracy"]})
    return {"decision": "REUSE_S3_T07_N3", "reason": "same PointSparse class, function, initialized state and topology per seed; frozen 54k/6k split, optimizer and 25-epoch protocol; T10 seed-1 first-three batch hashes and validation accuracy match",
            "seeds": audit, "cpu_thread_note": "T10 used one CPU thread; S3-T07 did not freeze thread count. Epoch-3 losses differ at small floating-point scale while accuracy matches."}


def preregister():
    if PREREG.exists():
        raise FileExistsError("T11 preregistration exists; never overwrite")
    audit = audit_anchor()
    for h, k in SHORTLIST:
        if structural(h, k)["trainable_parameters"] != 9610:
            raise ValueError("budget mismatch")
    sources = {}
    for group, (directory, arms) in HISTORICAL_GROUPS.items():
        paths = [ROOT / directory / f"{arm}_seed{seed}.json" for arm in arms for seed in SEEDS]
        if any(not path.exists() for path in paths):
            raise FileNotFoundError(f"historical cohort incomplete: {group}")
        sources[group] = [{"file": str(p.relative_to(ROOT)).replace("\\", "/"),
                           "sha256": _sha(p)} for p in paths]
    t10_pilot = json.loads(T10.read_text(encoding="utf-8"))
    predicted = ["PointSparse_80x109", "PointSparse_128x64", "PointSparse_160x49"]
    actual = [r["architecture_id"] for r in sorted(
        (r for r in t10_pilot["pilot_records"] if (r["H"], r["K"]) in SHORTLIST),
        key=lambda r: r["history"][2]["validation"]["accuracy"], reverse=True)]
    if actual != predicted:
        raise ValueError("T10 shortlist pilot ranking changed")
    data = {"schema": "s3_t11_prereg_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit_before_prereg": _git("rev-parse", "HEAD"),
            "shortlist": [{"H": h, "K": k, "structural_cost": structural(h, k)} for h, k in SHORTLIST],
            "anchor_audit": audit, "new_runs": ["80x109", "160x49"],
            "seeds": list(SEEDS), "training_seed_equals_topology_seed": True,
            "protocol": {"dataset": "Fashion-MNIST", "split": "NumPy seed shuffle 60000; final 6000 validation",
                         "train": 54000, "validation": 6000, "test": 10000,
                         "transform": "ToTensor /255", "optimizer": "Adam", "lr": .001,
                         "betas": [.9, .999], "epsilon": 1e-7, "weight_decay": 0,
                         "batch_size": 128, "epochs": 25, "checkpoint": "final_epoch_25",
                         "activation": "LeakyReLU 0.1", "loss": "CrossEntropyLoss",
                         "test_rule": "selected before T11; evaluate once per fixed seed/architecture after epoch 25; no tuning or repeats from test"},
            "pilot_predictiveness": {"primary": "epoch_3_validation_accuracy",
                                      "secondary_epochs": list(EPOCHS_EARLY),
                                      "targets": ["final_epoch_25_validation_accuracy", "final_epoch_25_test_accuracy_when_all_arms_have_it"],
                                      "aggregation": "within each protocol family, average each architecture over seeds 1,2,3 at each epoch; compute rank correlations over architectures; report per-seed diagnostics separately",
                                      "statistics": ["Spearman", "Kendall_tau_b", "pairwise_accuracy"],
                                      "accuracy_near_tie_strict_less_than": NEAR_TIE_ACC,
                                      "pairwise_near_tie_credit": .5,
                                      "cohorts": list(HISTORICAL_GROUPS),
                                      "no_cross_family_global_correlation": True},
            "historical_source_files": sources,
            "t10_results_sha256": _sha(T10), "t10_shortlist_epoch3_ranking": predicted,
            "zero_cost_comparison": "ER raw mean over proxy seeds vs epoch-3 validation accuracy, same cohort and target; S3-T07/S3-T05 use T09 proxy records, T11 shortlist uses T10 proxy records; S2-T06 has no complete ER group",
            "interpretation": {"H_WIDTH_FANIN": "SUPPORTED_WITHIN_TESTED_RANGE if either candidate-anchor pair has abs mean validation and test accuracy delta >=0.50 pp in same direction, >=2/3 paired seed directions, and NLL difference not opposed by >=0.005; WEAK_SUPPORT with >=0.20 pp same direction and >=2/3 seeds; NOT_SUPPORTED if both candidate-anchor abs deltas <0.20 pp for validation/test and abs NLL <0.005; otherwise INCONCLUSIVE",
                               "EARLY_PILOT_3E": "PROMISING if >=2 historical groups have rho>=0.50 and pairwise>0.70 for final validation and no group has rho<0 or pairwise<0.50; WEAK if >=2 groups are positive but any material inconsistency; FAILED otherwise or if any group is systematically inverse (rho<=-0.50 and pairwise<=0.30). T11 shortlist is prospective stress test, not an extra historical group"},
            "compute": {"naive_epoch_equivalents": 600, "t10": 24, "new_t11_planned": 150,
                        "total_planned": 174},
            "no_new_proxy_fitting": True, "no_new_candidates": True}
    _write(PREREG, data)
    return data


def sealed_prereg():
    raw = PREREG.read_bytes()
    commit = _git("log", "-1", "--format=%H", "--", "data/s3_t11_preregistered_confirmation.json")
    if not commit:
        raise RuntimeError("T11 preregistration must be committed before training")
    blob = subprocess.check_output(["git", "show", f"{commit}:data/s3_t11_preregistered_confirmation.json"], cwd=ROOT)
    if raw.replace(b"\r\n", b"\n") != blob:
        raise RuntimeError("T11 preregistration changed after commit")
    data = json.loads(blob)
    if data["schema"] != "s3_t11_prereg_v1" or data["anchor_audit"]["decision"] != "REUSE_S3_T07_N3":
        raise ValueError("invalid T11 preregistration")
    return data, hashlib.sha256(blob).hexdigest(), commit


def _validate_new(record, h, k, seed, prereg_sha, prereg_commit):
    anchor = json.loads((T07 / f"PointSparse_seed{seed}.json").read_text(encoding="utf-8"))
    if (record.get("schema") != "s3_t11_run_v1" or
        (record.get("H"), record.get("K"), record.get("seed")) != (h, k, seed) or
        record.get("prereg_sha256") != prereg_sha or record.get("prereg_commit") != prereg_commit or
        record.get("seeds", {}).get("training_seed") != seed or
        record.get("seeds", {}).get("topology_seed") != seed or
        record.get("structural_cost") != structural(h, k) or
        record.get("dataset", {}).get("splits") != {"train": 54000, "validation": 6000, "test": 10000} or
        any(record.get("dataset", {}).get(field) != anchor["dataset"][field] for field in
            ("train_indices_sha256", "validation_indices_sha256")) or
        record.get("batch_order") != anchor["batch_order"]["epoch_sha256"] or
        record.get("checkpoint_selected") != "final_epoch_25" or
        [row.get("epoch") for row in record.get("history", [])] != list(range(1, 26)) or
        len(record.get("batch_order", [])) != 25 or
        set(record.get("test", {})) != {"loss", "accuracy", "mean_logit_l2", "predictive_entropy",
                                           "correct_class_margin", "fraction_positive_margins", "ece_15"}):
        raise ValueError(f"invalid or incomplete run {h}x{k} seed {seed}")
    expected = build_model(h, k, seed, seed)
    if record["initialization_sha256"] != state_fingerprint(expected) or record["topology_sha256"] != sha(expected.G64):
        raise ValueError("initialization/topology drift")
    values = [record["duration_seconds"], *record["test"].values()]
    for row in record["history"]:
        values.extend((row["train_loss"], row["validation"]["loss"], row["validation"]["accuracy"]))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite run metric")


def run_one(h, k, seed, prereg_sha, prereg_commit):
    seed_everything(seed)
    torch.set_num_threads(1)
    model = build_model(h, k, seed, seed)
    initial_sha = state_fingerprint(model)
    cfg = config(seed)
    train, validation, test, dataset = load_fashion_mnist(cfg, include_test=False)
    if test is not None:
        raise RuntimeError("test opened before training")
    recorder = RecordingSampler(train.sampler, train.dataset.indices)
    train = DataLoader(train.dataset, batch_size=128, sampler=recorder, generator=train.generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-7, weight_decay=0)
    history = []
    start = time.perf_counter()
    for epoch in range(1, 26):
        model.train()
        total = count = 0
        for inputs, targets in train:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
        history.append({"epoch": epoch, "train_loss": total / count,
                        "validation": evaluate(model, validation, torch.device("cpu"))})
    duration = time.perf_counter() - start
    # The shortlist, seeds and final checkpoint were sealed before this access.
    official_test = FashionMNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
    test_metrics = diagnostic_evaluate(model, DataLoader(official_test, batch_size=128, shuffle=False))
    return {"schema": "s3_t11_run_v1", "architecture_id": f"PointSparse_{h}x{k}",
            "H": h, "K": k, "seed": seed, "prereg_sha256": prereg_sha,
            "prereg_commit": prereg_commit, "seeds": {"training_seed": seed,
            "topology_seed": seed, "python": seed, "numpy": seed, "torch": seed},
            "protocol": config(seed).to_dict(), "structural_cost": structural(h, k),
            "initialization_sha256": initial_sha, "topology_sha256": sha(model.G64),
            "dataset": dataset, "batch_order": recorder.epoch_sha256,
            "checkpoint_selected": "final_epoch_25", "history": history,
            "test": test_metrics, "duration_seconds": duration,
            "environment": {"device": "cpu", "precision": "float32", "torch": torch.__version__,
                            "cpu": platform.processor() or platform.uname().processor,
                            "os": platform.platform(), "torch_num_threads": torch.get_num_threads()}}


def train():
    prereg, prereg_sha, commit = sealed_prereg()
    if audit_anchor() != prereg["anchor_audit"]:
        raise ValueError("anchor audit drifted")
    RUNS.mkdir(parents=True, exist_ok=True)
    records = []
    for h, k in SHORTLIST[:2]:
        for seed in SEEDS:
            path = RUNS / f"PointSparse_{h}x{k}_seed{seed}.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                _validate_new(record, h, k, seed, prereg_sha, commit)
                print(f"resumed {h}x{k} seed {seed}", flush=True)
            else:
                record = run_one(h, k, seed, prereg_sha, commit)
                _validate_new(record, h, k, seed, prereg_sha, commit)
                _write(path, record)
                print(f"completed {h}x{k} seed {seed}: val {record['history'][-1]['validation']['accuracy']:.4f}, test {record['test']['accuracy']:.4f}", flush=True)
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preregister", "train"))
    args = parser.parse_args()
    if args.phase == "preregister":
        print(json.dumps(preregister()["anchor_audit"], indent=2))
    else:
        train()


if __name__ == "__main__":
    main()
