"""B1-E02 runner; the held-out split is read after the fixed final epoch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .b1_e01 import (
    C, D, DELAYS, H, K, LAMBDA, SEEDS, DATASET_SEED, PointSparseTemporal,
    initial_values, sequence_batch, tensor_hash, topology,
)
from .b1_e02 import make_noise_split, noise_sequence_batch, noise_split_fingerprint, NOISE_CHANNELS

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "experiments" / "configs" / "b1_e01_frozen.json"
RUNS = ROOT / "runs" / "b1_e02"
DESIGN = ROOT / "docs" / "B1_NOISE_DESIGN.md"


def file_hash(path: Path) -> str:
    # Git's Windows checkout uses CRLF; the frozen preregistration hashed LF bytes.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def frozen_protocol() -> dict:
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))
    if file_hash(ROOT / protocol["preregistered_design"]) != protocol["preregistered_design_sha256"]:
        raise ValueError("preregistered B1-E01 design changed")
    if (protocol["dimensions"] != {"D": D, "H": H, "K": K, "C": C} or
            protocol["delay_transitions"] != list(DELAYS) or
            protocol["training"]["training_seeds"] != list(SEEDS) or
            protocol["training"]["topology_seeds"] != list(SEEDS) or
            protocol["dataset"]["dataset_seed"] != DATASET_SEED or
            abs(protocol["lambda_double"] - LAMBDA) > 1e-15 or
            protocol["training"]["epochs"] != 20 or
            protocol["training"]["batch_size"] != 128):
        raise ValueError("frozen B1 config and implementation differ")
    if not DESIGN.exists():
        raise ValueError("missing B1-NOISE preregistration")
    historical = ROOT / "runs" / "b1_e01" / "S1_d16_seed1.json"
    if historical.exists():
        previous = json.loads(historical.read_text(encoding="utf-8"))
        if previous["lambda"] != float(np.float32(protocol["lambda_double"])):
            raise ValueError("lambda differs from historical B1-E01 artifact")
    return protocol


@lru_cache(maxsize=12)
def split_data(split: str, delay: int):
    first, labels, noise = make_noise_split(split, delay)
    return first, labels, noise, noise_split_fingerprint(first, labels, noise, split, delay)


def build(arm: str, seed: int) -> PointSparseTemporal:
    return PointSparseTemporal(topology(seed), initial_values(seed), arm == "S1")


def preflight(protocol: dict) -> dict:
    """Functional, leakage, cost, and analytical checks before any optimization."""
    torch.set_num_threads(1)
    costs = {}
    graphs = {}
    initializations = {}
    for seed in SEEDS:
        graph = topology(seed)
        if not (graph == 0).any() or not (graph == 1).any():
            raise ValueError(f"signal channels uncovered: topology_seed={seed}")
        s0, s1 = build("S0", seed), build("S1", seed)
        if tensor_hash(s0.G) != tensor_hash(s1.G):
            raise ValueError("unpaired topology")
        inits = {name: tensor_hash(getattr(s0, name)) for name in ("W_in", "b_s", "W_out", "b_out")}
        if inits != {name: tensor_hash(getattr(s1, name)) for name in inits}:
            raise ValueError("unpaired initialization")
        graphs[str(seed)] = tensor_hash(graph)
        initializations[str(seed)] = inits
        costs[str(seed)] = {arm: build(arm, seed).cost(16) for arm in ("S0", "S1")}
        if any(row["trainable_parameters"] != 706 or row["persistent_tensor_payload_bytes"] != 6920
               for row in costs[str(seed)].values()):
            raise ValueError("parameter/storage budget changed")
        sample_first = torch.zeros((2, D), dtype=torch.float32)
        sample_first[0, 0] = sample_first[1, 1] = 1
        sample_noise = torch.zeros((2, 15, 31), dtype=torch.float32)
        seq = noise_sequence_batch(sample_first, sample_noise, 16)
        if not torch.equal(s0(seq)[0], s0(seq)[1]):
            raise ValueError("static temporal leakage")
        if torch.equal(s1(seq)[0], s1(seq)[1]):
            raise ValueError("stateful has no temporal path")
        seq.requires_grad_()
        grad_s1 = torch.autograd.grad(s1(seq)[0, 0], seq, retain_graph=False)[0]
        grad_s0 = torch.autograd.grad(s0(seq)[0, 0], seq, retain_graph=False)[0]
        if grad_s1[:, 0].abs().sum() <= 0 or grad_s0[:, 0].count_nonzero() != 0:
            raise ValueError("temporal gradient gate failed")
        alternative = seq.detach().flip(0)
        reference_b = s1(alternative)
        _ = s1(seq.detach())
        if not torch.equal(s1(alternative), reference_b):
            raise ValueError("sequence reset failed")
    memory = []
    s1 = build("S1", 1)
    for delay in DELAYS:
        written = s1.state_step(torch.zeros((1, H), dtype=torch.float64), torch.ones((1, H), dtype=torch.float64))
        state = written.clone()
        for _ in range(delay):
            state = s1.state_step(state, torch.zeros_like(state))
        measured = (state / written).mean().item()
        predicted = LAMBDA ** delay
        if abs(measured - predicted) > 1e-6:
            raise ValueError("analytical memory curve gate failed")
        memory.append({"delay": delay, "predicted_retention": predicted,
                       "measured_retention": measured, "absolute_error": abs(measured - predicted)})
    dataset_audit = {}
    for delay in DELAYS:
        dataset_audit[str(delay)] = {}
        for split in ("train", "validation"):
            first, labels, noise, fingerprint = split_data(split, delay)
            seq = noise_sequence_batch(first[:128], noise[:128], delay)
            if (int(labels.sum()) != len(labels) // 2 or seq.shape[1] != delay + 1 or
                    not torch.equal(seq[:, -1], seq[0, -1].expand_as(seq[:, -1]))):
                raise ValueError(f"dataset leakage/structure gate failed: {split} delay={delay}")
            if (not torch.equal(seq[:, 0], sequence_batch(first[:128], delay)[:, 0]) or
                    not torch.equal(seq[:, -1], sequence_batch(first[:128], delay)[:, -1]) or
                    seq[:, 1:-1, 2].count_nonzero() != 0 or
                    noise.min() < -.1 or noise.max() > .1 or
                    noise[:, :, :2].count_nonzero() == 0):
                raise ValueError(f"NOISE construction failed: {split} delay={delay}")
            nuisance_delta = (first[labels == 0, 3:].mean(0) - first[labels == 1, 3:].mean(0)).abs().max().item()
            noise_delta = (noise[labels == 0].mean((0, 1)) - noise[labels == 1].mean((0, 1))).abs().max().item()
            if noise_delta >= .005:
                raise ValueError(f"NOISE class mean gap: {split} delay={delay}")
            dataset_audit[str(delay)][split] = {"size": len(labels), "labels_0": len(labels) // 2,
                "labels_1": len(labels) // 2, "fingerprint": fingerprint,
                "final_only_majority_accuracy": .5, "final_step_unique_vectors": 1,
                "max_abs_nuisance_mean_difference_by_label": nuisance_delta,
                "max_abs_noise_mean_difference_by_label": noise_delta,
                "signal_channel_noise_nonzero": int(noise[:, :, :2].count_nonzero()),
                "noise_min": noise.min().item(), "noise_max": noise.max().item()}
    record = {"schema": "b1_e02_preflight_v1", "run_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_protocol": protocol, "frozen_config_sha256": file_hash(FROZEN),
        "design_sha256": file_hash(DESIGN),
        "b1_e01_model_source_sha256": file_hash(ROOT / "src" / "dsu_research" / "b1_e01.py"),
        "topology_sha256": graphs,
        "initialization_sha256": initializations, "cost_audit": costs,
        "dataset_audit": dataset_audit, "analytical_memory": memory,
        "gate": "PASS", "test_split_accessed": False}
    write_json(RUNS / "preflight.json", record)
    return record


def training_order(seed: int, epoch: int, size: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + (epoch - 1) * 1000003)
    return torch.randperm(size, generator=generator)


def order_hashes(seed: int, size: int):
    return [tensor_hash(training_order(seed, epoch, size)) for epoch in range(1, 21)]


@torch.inference_mode()
def evaluate(model, data, delay: int, oracle: bool = False):
    model.eval()
    first, labels, noise, _ = data
    total_loss = total_correct = 0
    for start in range(0, len(labels), 128):
        seq = noise_sequence_batch(first[start:start + 128], noise[start:start + 128], delay, oracle=oracle)
        target = labels[start:start + 128]
        logits = model(seq)
        total_loss += F.cross_entropy(logits, target, reduction="sum").item()
        total_correct += (logits.argmax(-1) == target).sum().item()
    return {"loss": total_loss / len(labels), "accuracy": total_correct / len(labels)}


def run_one(arm: str, delay: int, seed: int, protocol: dict, pre: dict):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build("S0" if arm == "O" else arm, seed)
    train = split_data("train", delay)
    valid = split_data("validation", delay)
    oracle = arm == "O"
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
    history = []
    hashes = []
    start_time = time.perf_counter()
    for epoch in range(1, 21):
        model.train()
        order = training_order(seed, epoch, len(train[1]))
        hashes.append(tensor_hash(order))
        loss_sum = 0.0
        for batch_indices in order.split(128):
            inputs = noise_sequence_batch(train[0][batch_indices], train[2][batch_indices], delay, oracle=oracle)
            target = train[1][batch_indices]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(target)
        metrics = evaluate(model, valid, delay, oracle=oracle)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train[1]),
            "validation_loss": metrics["loss"], "validation_accuracy": metrics["accuracy"]})
        print(f"B1-E02 {arm} delay={delay} seed={seed} epoch={epoch}/20 "
              f"train={history[-1]['train_loss']:.5f} val={metrics['accuracy']:.4f}", flush=True)
    duration = time.perf_counter() - start_time
    if hashes != order_hashes(seed, len(train[1])):
        raise ValueError("training order changed; test remains sealed")
    # The held-out split is first instantiated only after the fixed final epoch.
    test = split_data("test", delay)
    result = evaluate(model, test, delay, oracle=oracle)
    record = {"schema": "b1_e02_run_v1", "experiment_id": "B1-E02", "arm": arm,
        "delay": delay, "sequence_length": delay + 1, "seed": seed,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_protocol_sha256": pre["frozen_config_sha256"],
        "preregistered_design_sha256": pre["design_sha256"],
        "architecture": "PointSparse-Static-Temporal" if arm == "S0" else
            "PointSparse-T-v0" if arm == "S1" else "Oracle-current-static",
        "state_equation": protocol["state_equation"] if arm == "S1" else "none",
        "lambda": model.decay if arm == "S1" else None,
        "noise_preregister_sha256": pre["design_sha256"],
        "seeds": {"dataset_seed": DATASET_SEED, "topology_seed": seed,
                  "training_seed": seed, "python": seed, "numpy": seed, "torch": seed, "cuda": None},
        "topology": {"G": model.G.tolist(), "sha256": tensor_hash(model.G)},
        "initialization_sha256": {name: tensor_hash(value) for name, value in initial_values(seed).items()},
        "dataset": {split: {"size": len(split_data(split, delay)[1]),
                            "fingerprint": split_data(split, delay)[3]}
                    for split in ("train", "validation", "test")},
        "training": {"optimizer": "Adam", "learning_rate": .001, "betas": [.9, .999],
            "epsilon": 1e-8, "weight_decay": 0, "batch_size": 128, "epochs": 20,
            "checkpoint": "final_epoch_20", "loss": "CrossEntropyLoss",
            "early_stopping": False, "scheduler": None, "gradient_clipping": None,
            "mixed_precision": False, "history": history, "duration_seconds": duration,
            "batch_order_sha256": hashes},
        "cost": model.cost(delay), "test": result,
        "environment": {"cpu": platform.processor() or platform.uname().processor,
            "os": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "device": "cpu", "precision": "float32", "torch_num_threads": torch.get_num_threads()}}
    return record


def validate_record(record: dict, arm: str, delay: int, seed: int, pre: dict):
    expected = build("S0" if arm == "O" else arm, seed)
    if (record.get("schema") != "b1_e02_run_v1" or record.get("arm") != arm or
            record.get("delay") != delay or record.get("seed") != seed or
            record.get("sequence_length") != delay + 1 or
            record.get("frozen_protocol_sha256") != pre["frozen_config_sha256"] or
            record.get("preregistered_design_sha256") != pre["design_sha256"] or
            record.get("noise_preregister_sha256") != pre["design_sha256"] or
            record.get("topology", {}).get("sha256") != tensor_hash(expected.G) or
            record.get("topology", {}).get("G") != expected.G.tolist() or
            record.get("cost") != expected.cost(delay) or
            record.get("training", {}).get("batch_order_sha256") != order_hashes(seed, 50000) or
            record.get("seeds") != {"dataset_seed": DATASET_SEED, "topology_seed": seed,
                "training_seed": seed, "python": seed, "numpy": seed, "torch": seed, "cuda": None} or
            record.get("initialization_sha256") !=
                {name: tensor_hash(value) for name, value in initial_values(seed).items()} or
            any(record.get("dataset", {}).get(split, {}).get("fingerprint") != split_data(split, delay)[3]
                for split in ("train", "validation", "test")) or
            record.get("training", {}).get("checkpoint") != "final_epoch_20" or
            [row.get("epoch") for row in record.get("training", {}).get("history", [])] != list(range(1, 21))):
        raise ValueError(f"invalid B1-E02 artifact {arm} delay={delay} seed={seed}")
    metrics = [record["training"]["duration_seconds"],
               record["test"]["accuracy"], record["test"]["loss"]]
    for row in record["training"]["history"]:
        metrics += [row["train_loss"], row["validation_loss"], row["validation_accuracy"]]
    if not all(math.isfinite(value) for value in metrics):
        raise ValueError("nonfinite B1 metrics")
    if not (0 <= record["test"]["accuracy"] <= 1):
        raise ValueError("invalid B1 accuracy")


def result_path(arm: str, delay: int, seed: int) -> Path:
    return RUNS / f"{arm}_d{delay}_seed{seed}.json"


def load_or_run(arm: str, delay: int, seed: int, protocol: dict, pre: dict):
    path = result_path(arm, delay, seed)
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_record(record, arm, delay, seed, pre)
        print(f"REUSE {path}", flush=True)
        return record
    record = run_one(arm, delay, seed, protocol, pre)
    validate_record(record, arm, delay, seed, pre)
    write_json(path, record)
    return record


def stats(values):
    return {"values": values, "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values)}


def summarize(records, oracle_records=None):
    grouped = {(r["arm"], r["delay"], r["seed"]): r for r in records}
    if len(grouped) != 24 or any((arm, delay, seed) not in grouped
                               for arm in ("S0", "S1") for delay in DELAYS for seed in SEEDS):
        raise ValueError("B1-E02 summary requires all 24 primary runs")
    curves = []
    for delay in DELAYS:
        pair = []
        for seed in SEEDS:
            s0, s1 = grouped["S0", delay, seed], grouped["S1", delay, seed]
            if (s0["topology"]["sha256"] != s1["topology"]["sha256"] or
                    s0["initialization_sha256"] != s1["initialization_sha256"] or
                    s0["training"]["batch_order_sha256"] != s1["training"]["batch_order_sha256"] or
                    s0["dataset"] != s1["dataset"]):
                raise ValueError("pairing failed")
            pair.append({"seed": seed, "static_accuracy": s0["test"]["accuracy"],
                "stateful_accuracy": s1["test"]["accuracy"],
                "memory_advantage_pp": 100 * (s1["test"]["accuracy"] - s0["test"]["accuracy"]),
                "static_nll": s0["test"]["loss"], "stateful_nll": s1["test"]["loss"],
                "delta_nll": s1["test"]["loss"] - s0["test"]["loss"]})
        curves.append({"delay": delay, "theoretical_retention": LAMBDA ** delay,
            "seeds": pair,
            "static_accuracy_percent": stats([100 * p["static_accuracy"] for p in pair]),
            "stateful_accuracy_percent": stats([100 * p["stateful_accuracy"] for p in pair]),
            "memory_advantage_pp": stats([p["memory_advantage_pp"] for p in pair]),
            "static_nll": stats([p["static_nll"] for p in pair]),
            "stateful_nll": stats([p["stateful_nll"] for p in pair]),
            "delta_nll": stats([p["delta_nll"] for p in pair])})
    primary = next(row for row in curves if row["delay"] == 16)
    s0 = primary["static_accuracy_percent"]["mean"]
    s1 = primary["stateful_accuracy_percent"]["mean"]
    advantage = primary["memory_advantage_pp"]["mean"]
    nll = primary["delta_nll"]["mean"]
    wins = sum(row["memory_advantage_pp"] > 0 for row in primary["seeds"])
    oracle_accuracy = None if oracle_records is None else statistics.mean(
        100 * row["test"]["accuracy"] for row in oracle_records)
    if advantage >= 10 and wins == 3 and s1 >= 70 and nll < 0:
        verdict = "SUPPORTED"
    elif 45 <= s0 <= 55 and advantage < 5 and (not 45 <= s1 <= 55 or
             (oracle_accuracy is not None and oracle_accuracy >= 90)):
        verdict = "NOT SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"
    return {"schema": "b1_e02_summary_v1", "experiment_id": "B1-E02",
        "hypothesis": "H-B1-INTERFERENCE", "status": verdict,
        "primary_delay": 16, "memory_curve": curves,
        "oracle_triggered": 45 <= s0 <= 55 and 45 <= s1 <= 55,
        "oracle_accuracy_percent_mean": oracle_accuracy,
        "total_primary_training_seconds": sum(r["training"]["duration_seconds"] for r in records),
        "interpretation_scope": "delayed_recall_noise_v1 only; no LSTM/GRU/energy/runtime comparison"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    protocol = frozen_protocol()
    pre = preflight(protocol)
    print(f"PREFLIGHT {pre['gate']} {pre['frozen_config_sha256']}", flush=True)
    if args.preflight_only:
        return
    records = []
    for delay in DELAYS:
        for seed in SEEDS:
            for arm in ("S0", "S1"):
                records.append(load_or_run(arm, delay, seed, protocol, pre))
    first_summary = summarize(records)
    oracle_records = None
    if first_summary["oracle_triggered"]:
        oracle_records = [load_or_run("O", 16, seed, protocol, pre) for seed in SEEDS]
    summary = summarize(records, oracle_records)
    summary["frozen_protocol_sha256"] = pre["frozen_config_sha256"]
    summary["preregistered_design_sha256"] = pre["design_sha256"]
    write_json(RUNS / "summary.json", summary)
    print(f"B1-E02 {summary['status']} primary delay=16; summary={RUNS / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
