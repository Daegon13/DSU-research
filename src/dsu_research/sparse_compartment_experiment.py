"""S3-T07: paired, resumable PointSparse / neutral compartment experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .compartment_ablation import _load_data, sha, write_json
from .config import load_config
from .harness import evaluate, seed_everything
from .s3_t06_diagnostics import train_calibration_batch
from .sparse_compartment_controls import NeutralCompartments, PointSparse, base_topology, paired_values

SEEDS = (1, 2, 3)
ARMS = ("PointSparse", "1x64", "2x32", "4x16", "8x8")


def diagnostic_evaluate(model, loader):
    model.eval()
    sums = dict(loss=0.0, correct=0, logit_l2=0.0, entropy=0.0,
                correct_margin=0.0, positive_margin=0, ece=0.0)
    bins_count = torch.zeros(15, dtype=torch.float64)
    bins_conf = torch.zeros(15, dtype=torch.float64)
    bins_correct = torch.zeros(15, dtype=torch.float64)
    n = 0
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x)
            probs = logits.softmax(-1)
            conf, pred = probs.max(-1)
            correct = pred.eq(y)
            target_logit = logits.gather(1, y[:, None]).squeeze(1)
            other = logits.scatter(1, y[:, None], -torch.inf).max(1).values
            margin = target_logit - other
            sums["loss"] += F.cross_entropy(logits, y, reduction="sum").item()
            sums["correct"] += correct.sum().item()
            sums["logit_l2"] += logits.norm(dim=-1).sum().item()
            sums["entropy"] += (-(probs * probs.clamp_min(1e-30).log()).sum(-1)).sum().item()
            sums["correct_margin"] += margin.sum().item()
            sums["positive_margin"] += (margin > 0).sum().item()
            which = torch.clamp(torch.ceil(conf * 15).long() - 1, 0, 14)
            bins_count += torch.bincount(which, minlength=15).to(torch.float64)
            bins_conf += torch.bincount(which, weights=conf.to(torch.float64), minlength=15)
            bins_correct += torch.bincount(which, weights=correct.to(torch.float64), minlength=15)
            n += len(y)
    if n != 10000:
        raise ValueError("test set is not the full official 10000 examples")
    active = bins_count > 0
    ece = ((bins_correct[active] / bins_count[active] -
            bins_conf[active] / bins_count[active]).abs() * bins_count[active]).sum().item() / n
    return {"loss": sums["loss"] / n, "accuracy": sums["correct"] / n,
            "mean_logit_l2": sums["logit_l2"] / n,
            "predictive_entropy": sums["entropy"] / n,
            "correct_class_margin": sums["correct_margin"] / n,
            "fraction_positive_margins": sums["positive_margin"] / n,
            "ece_15": ece}


def build_arm(seed, arm, calibration_inputs):
    g64 = base_topology(seed)
    values = paired_values(seed)
    point = PointSparse(g64)
    point_init = point.initialize(*values)
    if arm == "PointSparse":
        return point, point_init, None
    branches = int(arm.split("x")[0])
    model = NeutralCompartments(branches, g64)
    init = model.initialize(*values)
    calibration = model.calibrate(calibration_inputs, point)
    return model, init, calibration


def run_arm(config, arm):
    seeds = seed_everything(config.seed)
    calibration_inputs, calibration_hash = train_calibration_batch(config.seed)
    model, initialization, calibration = build_arm(config.seed, arm, calibration_inputs)
    train, validation, test, dataset, recorder = _load_data(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                 eps=1e-7, weight_decay=0)
    history = []
    started = time.perf_counter()
    for epoch in range(25):
        model.train()
        loss_sum = count = 0
        for x, y in train:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            count += len(y)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / count,
                        "validation": evaluate(model, validation, torch.device("cpu"))})
    duration = time.perf_counter() - started
    result = {"schema": "s3_t07_v1", "run_utc": datetime.now(timezone.utc).isoformat(),
              "arm": arm, "config": config.to_dict(), "seeds": {**seeds,
              "training_seed": config.seed, "topology_seed": config.topology_seed},
              "dataset": dataset, "topology": {"g64_sha256": sha(model.G64),
              "unique_inputs_per_soma": 64}, "initialization": initialization,
              "calibration_indices_sha256": calibration_hash, "calibration": calibration,
              "counts": model.counts(), "batch_order": {"epoch_sha256": recorder.epoch_sha256,
              "combined_sha256": recorder.combined_sha256()},
              "training": {"optimizer": "Adam", "betas": [0.9, 0.999], "epsilon": 1e-7,
              "weight_decay": 0, "scheduler": None, "gradient_clipping": None,
              "mixed_precision": False, "early_stopping": False,
              "checkpoint_selected": "final_epoch_25", "history": history,
              "duration_seconds": duration}, "test": diagnostic_evaluate(model, test),
              "environment": {"device": "cpu", "precision": "float32",
              "cpu": platform.processor() or platform.uname().processor,
              "os": platform.platform(), "python": platform.python_version(),
              "torch": torch.__version__, "torchvision": version("torchvision"),
              "numpy": np.__version__}}
    return result


def validate_record(record, seed, arm, frozen_calibration):
    config = record.get("config", {})
    history = record.get("training", {}).get("history", [])
    order = record.get("batch_order", {}).get("epoch_sha256", [])
    if (record.get("schema") != "s3_t07_v1" or record.get("arm") != arm or
        config.get("seed") != seed or config.get("topology_seed") != seed or
        config.get("model") != "sparse_compartment" or
        config.get("dataset") != "fashion_mnist" or config.get("epochs") != 25 or
        config.get("batch_size") != 128 or config.get("learning_rate") != .001 or
        config.get("device") != "cpu" or config.get("train_samples") is not None or
        config.get("test_samples") is not None or
        record.get("dataset", {}).get("splits") != {"train": 54000, "validation": 6000, "test": 10000} or
        record.get("topology", {}).get("g64_sha256") != sha(base_topology(seed)) or
        record.get("counts", {}).get("parameters") != 9610 or
        record.get("counts", {}).get("input_weights") != 8192 or
        record.get("counts", {}).get("main_macs_per_sample") != 9472 or
        record.get("counts", {}).get("fixed_aggregate_additions_per_sample") !=
            (0 if arm == "PointSparse" else 128 * (int(arm.split("x")[0]) - 1)) or
        record.get("seeds", {}).get("training_seed") != seed or
        record.get("seeds", {}).get("topology_seed") != seed or
        record.get("training", {}).get("checkpoint_selected") != "final_epoch_25" or
        not math.isfinite(record.get("training", {}).get("duration_seconds", math.nan)) or
        record.get("environment", {}).get("precision") != "float32" or
        len(history) != 25 or [h.get("epoch") for h in history] != list(range(1, 26)) or
        len(order) != 25 or any(len(d) != 64 for d in order)):
        raise ValueError(f"invalid record {arm} seed {seed}")
    _, expected_init, expected_cal = build_arm(
        seed, arm, train_calibration_batch(seed)[0])
    if record["initialization"] != expected_init or record["calibration"] != expected_cal:
        raise ValueError("initialization or calibration changed")
    if record["calibration_indices_sha256"] != train_calibration_batch(seed)[1]:
        raise ValueError("calibration indices changed")
    if record["batch_order"]["combined_sha256"] != hashlib.sha256(
            b"".join(bytes.fromhex(d) for d in order)).hexdigest():
        raise ValueError("batch order digest changed")
    if arm != "PointSparse":
        reference = next(r for r in frozen_calibration["records"] if r["seed"] == seed and r["arm"] == arm)
        if record["calibration"] != reference["calibration"]:
            raise ValueError("S3-T06 calibration coefficients changed")
    for h in history:
        if not all(math.isfinite(v) for v in (h["train_loss"], h["validation"]["loss"], h["validation"]["accuracy"])):
            raise ValueError("invalid history")
    if set(record.get("test", {})) != {"loss", "accuracy", "mean_logit_l2", "predictive_entropy",
                                       "correct_class_margin", "fraction_positive_margins", "ece_15"}:
        raise ValueError("incomplete test metrics")
    if not all(math.isfinite(v) for v in record["test"].values()):
        raise ValueError("non-finite test metrics")


def stats(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def summarize(records):
    grouped = {a: sorted((r for r in records if r["arm"] == a), key=lambda r: r["config"]["seed"]) for a in ARMS}
    if any([r["config"]["seed"] for r in group] != list(SEEDS) for group in grouped.values()):
        raise ValueError("summary requires 15 runs")
    fields = ("g64_sha256",)
    for seed in SEEDS:
        paired = [grouped[a][seed - 1] for a in ARMS]
        for key in fields:
            if len({r["topology"][key] for r in paired}) != 1:
                raise ValueError("topology pairing failed")
        for key in ("base_input_sha256", "output_sha256", "soma_bias_sha256", "output_bias_sha256"):
            if len({r["initialization"][key] for r in paired}) != 1:
                raise ValueError("initialization pairing failed")
        for path in (("batch_order", "combined_sha256"), ("dataset", "train_indices_sha256"),
                     ("dataset", "validation_indices_sha256")):
            if len({r[path[0]][path[1]] for r in paired}) != 1:
                raise ValueError("data pairing failed")
        if len({r["calibration_indices_sha256"] for r in paired}) != 1:
            raise ValueError("calibration pairing failed")
    arms = {}
    for arm, group in grouped.items():
        arms[arm] = {key: stats([r["test"][key] for r in group]) for key in group[0]["test"]}
        arms[arm]["validation_loss"] = stats([r["training"]["history"][-1]["validation"]["loss"] for r in group])
        arms[arm]["validation_accuracy"] = stats([r["training"]["history"][-1]["validation"]["accuracy"] for r in group])
    comparisons = {}
    for low, high in (("PointSparse", "1x64"), ("1x64", "2x32"), ("2x32", "4x16"),
                      ("4x16", "8x8"), ("1x64", "4x16")):
        name = f"{high}_minus_{low}"
        comparisons[name] = {key: stats([grouped[high][i]["test"][key] - grouped[low][i]["test"][key]
                                          for i in range(3)]) for key in grouped[low][0]["test"]}
    def classify(key):
        d = comparisons[key]
        acc = d["accuracy"]["mean"] * 100
        loss = d["loss"]["mean"]
        wins = sum(v > 0 for v in d["accuracy"]["values"])
        if acc >= .50 and wins >= 2 and loss <= .005:
            return "PRELIMINARY_SUPPORT"
        if .15 <= acc < .50 and wins >= 2 and loss <= .005:
            return "WEAK_SUPPORT"
        if abs(acc) < .15 and abs(loss) < .005:
            return "NEGLIGIBLE"
        if acc <= -.50 or (loss >= .01 and all(v > 0 for v in d["loss"]["values"])):
            return "CONTRARY"
        return "INCONCLUSIVE"
    point_acc = arms["PointSparse"]["accuracy"]["mean"] * 100
    mlp_gain = point_acc - 85.883
    dsu_gap = 87.253 - point_acc
    if mlp_gain < .30:
        h014 = "NOT_SUPPORTED"
    elif mlp_gain >= .75 and dsu_gap <= .50:
        h014 = "SUPPORTED"
    elif .30 <= mlp_gain < .75 or .50 < dsu_gap <= 1.0:
        h014 = "WEAK_SUPPORT"
    else:
        h014 = "INCONCLUSIVE"
    return {"schema": "s3_t07_summary_v1", "arms": arms, "comparisons": comparisons,
            "historical": {"mlp_accuracy_percent": 85.883, "dsu_s_accuracy_percent": 87.253,
            "point_minus_mlp_pp": mlp_gain, "point_minus_dsu_s_pp": -dsu_gap,
            "fraction_historical_gap_retained": mlp_gain / 1.370},
            "classification": {"H-014": h014,
            "H-015": classify("4x16_minus_1x64"),
            "H-016": classify("1x64_minus_PointSparse")},
            "seeds": list(SEEDS), "completed_utc": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/sparse_compartment_s3_t07"))
    args = parser.parse_args()
    base = load_config(args.config)
    if (base.model != "sparse_compartment" or base.dataset != "fashion_mnist" or
        base.epochs != 25 or base.batch_size != 128 or base.learning_rate != .001 or
        base.device != "cpu" or base.train_samples is not None or base.test_samples is not None):
        parser.error("S3-T07 requires the frozen Fashion-MNIST CPU protocol")
    frozen = json.loads(Path("runs/s3_t06_initial_diagnostics.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in SEEDS:
        for arm in ARMS:
            path = args.output / f"{arm}_seed{seed}.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                validate_record(record, seed, arm, frozen)
                print(f"resumed {path}", flush=True)
            else:
                record = run_arm(replace(base, seed=seed, topology_seed=seed), arm)
                validate_record(record, seed, arm, frozen)
                write_json(path, record)
                print(f"completed {path}", flush=True)
            records.append(record)
    summary = summarize(records)
    write_json(args.output / "summary.json", summary)
    print(summary["classification"], flush=True)


if __name__ == "__main__":
    main()
