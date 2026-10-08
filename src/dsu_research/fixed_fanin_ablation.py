"""S3-E01 matched dense-mask ablation for variable versus fixed fan-in."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from .config import ExperimentConfig, load_config
from .data import load_fashion_mnist
from .dsu_s import DSUSv0
from .fixed_dann import FixedDANN
from .harness import evaluate, seed_everything
from .model import DendriticANNRandom, VanillaANN


SEEDS = (1, 2, 3, 4, 5)
ARMS = ("dann_r", "fixed_dann")
HISTORICAL_ACCURACY_TOLERANCE = 0.02
HISTORICAL_LOSS_TOLERANCE = 0.05


class RecordingSampler(Sampler[int]):
    """Record the exact source-example order emitted by another sampler."""

    def __init__(self, sampler, source_indices):
        self.sampler = sampler
        self.source_indices = np.asarray(source_indices, dtype=np.int64)
        self.epoch_sha256: list[str] = []

    def __iter__(self):
        order = list(iter(self.sampler))
        source_order = self.source_indices[np.asarray(order, dtype=np.int64)]
        self.epoch_sha256.append(hashlib.sha256(
            source_order.astype("<i8", copy=False).tobytes()).hexdigest())
        return iter(order)

    def __len__(self):
        return len(self.sampler)

    def combined_sha256(self) -> str:
        digest = hashlib.sha256()
        for epoch_hash in self.epoch_sha256:
            digest.update(bytes.fromhex(epoch_hash))
        return digest.hexdigest()


def load_recorded_fashion_mnist(config: ExperimentConfig):
    """Preserve the existing loader semantics while fingerprinting actual train order."""
    train_loader, validation_loader, test_loader, metadata = load_fashion_mnist(config)
    recorder = RecordingSampler(train_loader.sampler, train_loader.dataset.indices)
    recorded_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=config.batch_size,
        sampler=recorder,
        generator=train_loader.generator,
    )
    return recorded_train_loader, validation_loader, test_loader, metadata, recorder


def _tensor_hash(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().numpy().astype("<f4", copy=False)
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_paired_models(training_seed: int, topology_seed: int):
    """Generate dense tensors once, copy them to A/B, then apply each mask."""
    seed_everything(training_seed)
    template = VanillaANN()
    shared = tuple(parameter.detach().clone() for parameter in template.parameters())
    initialization_sha256 = _tensor_hash(shared)

    dann = DendriticANNRandom(topology_seed)
    fixed_topology = DSUSv0(topology_seed).G
    fixed = FixedDANN(fixed_topology)
    for model in (dann, fixed):
        with torch.no_grad():
            for target, source in zip(model.parameters(), shared):
                target.copy_(source)
            model.dendrites.weight.mul_(model.input_mask)
            soma_mask = model.cable_mask if isinstance(model, DendriticANNRandom) else model.soma_mask
            model.somata.weight.mul_(soma_mask)

    if not torch.equal(dann.dendrites.bias, fixed.dendrites.bias):
        raise RuntimeError("paired dendritic biases differ")
    if not torch.equal(dann.somata.bias, fixed.somata.bias):
        raise RuntimeError("paired soma biases differ")
    if not torch.equal(dann.output.weight, fixed.output.weight):
        raise RuntimeError("paired output weights differ")
    if not torch.equal(dann.output.bias, fixed.output.bias):
        raise RuntimeError("paired output biases differ")
    overlap = dann.input_mask & fixed.input_mask
    if not torch.equal(dann.dendrites.weight[overlap], fixed.dendrites.weight[overlap]):
        raise RuntimeError("shared active dendritic weights differ")
    return dann, fixed, initialization_sha256


def _mask_metrics(model) -> dict:
    mask = model.input_mask
    counts = mask.sum(dim=1).to(torch.float64)
    soma_mask = model.cable_mask if isinstance(model, DendriticANNRandom) else model.soma_mask
    return {
        "input_connections": int(mask.sum()),
        "soma_connections": int(soma_mask.sum()),
        "fan_in_min": int(counts.min()),
        "fan_in_max": int(counts.max()),
        "fan_in_mean": float(counts.mean()),
        "fan_in_sample_std": float(counts.std(unbiased=True)),
        "input_mask_sha256": hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest(),
    }


def train_arm(model, config: ExperimentConfig, arm: str, initialization_sha256: str) -> dict:
    seed_everything(config.seed)
    train_loader, validation_loader, test_loader, dataset, recorder = load_recorded_fashion_mnist(config)
    device = torch.device(config.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-7, weight_decay=0)
    history = []
    start = time.perf_counter()
    samples_seen = 0
    for epoch in range(config.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += len(targets)
        validation = evaluate(model, validation_loader, device)
        samples_seen += count
        history.append({
            "epoch": epoch + 1,
            "loss": loss_sum / count,
            "accuracy": correct / count,
            "validation": validation,
        })
    duration = time.perf_counter() - start
    test = evaluate(model, test_loader, device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "config": config.to_dict(),
        "seeds": {"training_seed": config.seed, "topology_seed": config.topology_seed},
        "initialization": {
            "policy": "one VanillaANN dense Glorot/zero-bias template copied to both arms before masks",
            "dense_tensors_sha256": initialization_sha256,
        },
        "dataset": dataset,
        "batch_order": {
            "policy": "independent loaders with identical persistent seeded RandomSampler state",
            "epoch_sha256": recorder.epoch_sha256,
            "combined_sha256": recorder.combined_sha256(),
        },
        "parameters": {
            "stored_parameters": parameters,
            "effective_parameters": model.effective_parameters(),
        },
        "topology": _mask_metrics(model),
        "training": {
            "optimizer": "Adam", "learning_rate": config.learning_rate,
            "adam_betas": [0.9, 0.999], "adam_epsilon": 1e-7,
            "weight_decay": 0.0, "scheduler": None, "epochs": config.epochs,
            "batch_size": config.batch_size, "checkpoint_selected": f"final_epoch_{config.epochs}",
            "samples_seen": samples_seen, "history": history,
            "duration_seconds": duration,
        },
        "test": test,
        "environment": {
            "device": str(device), "cpu": platform.processor() or platform.uname().processor,
            "os": platform.platform(), "precision": "float32",
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": version("torchvision"), "numpy": np.__version__,
        },
    }


def _stats(values) -> dict:
    return {"values": values, "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values)}


def historical_trial_is_anomalous(current: dict, historical: dict) -> bool:
    return (abs(current["test"]["accuracy"] - historical["test"]["accuracy"])
            > HISTORICAL_ACCURACY_TOLERANCE
            or abs(current["test"]["loss"] - historical["test"]["loss"])
            > HISTORICAL_LOSS_TOLERANCE)


def classify_h009(delta_accuracy: float, delta_loss: float,
                  accuracy_wins: int, loss_wins: int) -> str:
    if ((delta_accuracy >= 0.005 and accuracy_wins >= 4)
            or (delta_loss <= -0.015 and delta_accuracy > 0)):
        return "SUPPORTED"
    if ((delta_accuracy <= -0.005)
            or (delta_loss >= 0.015 and loss_wins <= 1)):
        return "CONTRARY"
    if abs(delta_accuracy) < 0.0015 and abs(delta_loss) < 0.005:
        return "NOT SUPPORTED / NEGLIGIBLE"
    if ((0.0015 <= delta_accuracy < 0.005 and accuracy_wins >= 3 and delta_loss < 0.005)
            or (-0.015 < delta_loss <= -0.005 and delta_accuracy > -0.0015)):
        return "WEAK SUPPORT"
    return "INCONCLUSIVE"


def summarize(records: list[dict], historical: dict) -> dict:
    by_arm = {arm: sorted((record for record in records if record["arm"] == arm),
                          key=lambda record: record["config"]["seed"])
              for arm in ARMS}
    if any(len(group) != len(SEEDS) for group in by_arm.values()):
        raise ValueError("summary requires five complete trials per arm")
    for left, right in zip(by_arm["dann_r"], by_arm["fixed_dann"]):
        if left["config"]["seed"] != right["config"]["seed"]:
            raise ValueError("paired seeds differ")
        if left["initialization"]["dense_tensors_sha256"] != right["initialization"]["dense_tensors_sha256"]:
            raise ValueError("paired initialization hashes differ")
        if left["batch_order"] != right["batch_order"]:
            raise ValueError("paired minibatch order differs")
        for key in ("train_indices_sha256", "validation_indices_sha256"):
            if left["dataset"][key] != right["dataset"][key]:
                raise ValueError(f"paired {key} differs")

    aggregate = {"arms": {}, "paired_deltas_fixed_minus_dann": {}}
    for arm, group in by_arm.items():
        aggregate["arms"][arm] = {}
        for split, metric in (("test", "loss"), ("test", "accuracy"),
                              ("validation", "loss"), ("validation", "accuracy")):
            values = [(record["test"] if split == "test" else
                       record["training"]["history"][-1]["validation"])[metric]
                      for record in group]
            aggregate["arms"][arm][f"{split}_{metric}"] = _stats(values)

    for split, metric in (("test", "loss"), ("test", "accuracy"),
                          ("validation", "loss"), ("validation", "accuracy")):
        deltas = []
        for left, right in zip(by_arm["dann_r"], by_arm["fixed_dann"]):
            left_metrics = left["test"] if split == "test" else left["training"]["history"][-1]["validation"]
            right_metrics = right["test"] if split == "test" else right["training"]["history"][-1]["validation"]
            deltas.append(right_metrics[metric] - left_metrics[metric])
        aggregate["paired_deltas_fixed_minus_dann"][f"{split}_{metric}"] = _stats(deltas)

    test_accuracy_deltas = aggregate["paired_deltas_fixed_minus_dann"]["test_accuracy"]["values"]
    test_loss_deltas = aggregate["paired_deltas_fixed_minus_dann"]["test_loss"]["values"]
    accuracy_wins = sum(delta > 0 for delta in test_accuracy_deltas)
    loss_wins = sum(delta < 0 for delta in test_loss_deltas)
    aggregate["direction_counts"] = {
        "fixed_accuracy_wins": accuracy_wins,
        "dann_accuracy_wins": sum(delta < 0 for delta in test_accuracy_deltas),
        "accuracy_ties": sum(delta == 0 for delta in test_accuracy_deltas),
        "fixed_loss_wins": loss_wins,
        "dann_loss_wins": sum(delta > 0 for delta in test_loss_deltas),
        "loss_ties": sum(delta == 0 for delta in test_loss_deltas),
    }
    aggregate["h009_classification"] = classify_h009(
        statistics.mean(test_accuracy_deltas), statistics.mean(test_loss_deltas),
        accuracy_wins, loss_wins)

    historical_trials = historical["models"]["dann_r"]
    new_accuracy = aggregate["arms"]["dann_r"]["test_accuracy"]["mean"]
    new_loss = aggregate["arms"]["dann_r"]["test_loss"]["mean"]
    historical_accuracy = historical_trials["accuracy"]["mean"]
    historical_loss = historical_trials["loss"]["mean"]
    aggregate["historical_sanity"] = {
        "historical_test_accuracy": historical_accuracy,
        "new_test_accuracy": new_accuracy,
        "accuracy_delta": new_accuracy - historical_accuracy,
        "historical_test_loss": historical_loss,
        "new_test_loss": new_loss,
        "loss_delta": new_loss - historical_loss,
        "material_anomaly": (abs(new_accuracy - historical_accuracy) > HISTORICAL_ACCURACY_TOLERANCE
                             or abs(new_loss - historical_loss) > HISTORICAL_LOSS_TOLERANCE),
        "diagnostic_thresholds": {
            "absolute_accuracy": HISTORICAL_ACCURACY_TOLERANCE,
            "absolute_loss": HISTORICAL_LOSS_TOLERANCE,
        },
    }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S3-E01 fixed fan-in ablation")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/fixed_fanin_s3_t03"))
    parser.add_argument("--historical", type=Path, default=Path("runs/reproduction_001/summary.json"))
    args = parser.parse_args()
    base = load_config(args.config)
    if (base.model != "fixed_fanin_pair" or base.dataset != "fashion_mnist"
            or base.epochs != 25 or base.batch_size != 128 or base.learning_rate != 0.001
            or base.device != "cpu"):
        parser.error("S3-E01 requires the frozen fixed_fanin_pair CPU protocol")
    args.output.mkdir(parents=True, exist_ok=True)
    historical = json.loads(args.historical.read_text(encoding="utf-8"))
    records = []
    for seed in SEEDS:
        config = replace(base, seed=seed, topology_seed=seed)
        dann, fixed, initialization_sha256 = build_paired_models(seed, seed)
        for arm, model in (("dann_r", dann), ("fixed_dann", fixed)):
            result = train_arm(model, config, arm, initialization_sha256)
            path = args.output / f"{arm}_seed{seed}.json"
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            records.append(result)
            print(f"{arm} seed={seed} test={result['test']} duration={result['training']['duration_seconds']:.1f}s",
                  flush=True)
            if arm == "dann_r":
                historical_trial_path = args.historical.parent / f"dann_r_seed{seed}.json"
                historical_trial = json.loads(historical_trial_path.read_text(encoding="utf-8"))
                if historical_trial_is_anomalous(result, historical_trial):
                    raise RuntimeError(f"new dANN-R seed {seed} materially diverges from historical baseline")
    summary = summarize(records, historical)
    if summary["historical_sanity"]["material_anomaly"]:
        raise RuntimeError("new dANN-R materially diverges from historical N=5 baseline")
    summary["seeds"] = list(SEEDS)
    summary["protocol"] = base.to_dict()
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    path = args.output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {path}; H-009={summary['h009_classification']}", flush=True)


if __name__ == "__main__":
    main()
