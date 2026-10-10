"""S3-T05: nested, paired 64-input dendritic compartment ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
from torch.nn import functional as F

from .config import load_config
from .data import load_fashion_mnist
from .fixed_fanin_ablation import RecordingSampler
from .harness import evaluate, seed_everything
from torch.utils.data import DataLoader


BRANCHES = (1, 2, 4, 8)
SEEDS = (1, 2, 3)
SLOPE = 0.1


def sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def base_topology(seed: int) -> torch.Tensor:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("topology seed must be a nonnegative 32-bit integer")
    rng = np.random.default_rng(seed)
    values = np.stack([rng.choice(784, 64, replace=False) for _ in range(128)])
    return torch.from_numpy(values.astype(np.int64))


class CompartmentalDSU(nn.Module):
    """Compact model; G64 is partitioned by contiguous nested blocks."""

    def __init__(self, branches: int, g64: torch.Tensor):
        super().__init__()
        if branches not in BRANCHES or g64.shape != (128, 64):
            raise ValueError("expected branches in {1,2,4,8} and G64 [128,64]")
        if torch.any((g64 < 0) | (g64 >= 784)) or torch.any(torch.sort(g64, dim=1).values[:, 1:] == torch.sort(g64, dim=1).values[:, :-1]):
            raise ValueError("each soma needs 64 distinct input indices")
        self.branches = branches
        self.fan_in = 64 // branches
        self.register_buffer("G", g64.reshape(128, branches, self.fan_in).clone())
        self.W_d = nn.Parameter(torch.empty(128, branches, self.fan_in))
        self.bias_d = nn.Parameter(torch.zeros(128, branches))
        self.W_s = nn.Parameter(torch.empty(128, branches))
        self.bias_s = nn.Parameter(torch.zeros(128))
        self.W_o = nn.Parameter(torch.empty(10, 128))
        self.bias_o = nn.Parameter(torch.zeros(10))
        nn.init.xavier_uniform_(self.W_d.reshape(-1, self.fan_in))
        nn.init.xavier_uniform_(self.W_s)
        nn.init.xavier_uniform_(self.W_o)

    def intermediates(self, x: torch.Tensor):
        x = x.flatten(1)
        if x.shape[1] != 784:
            raise ValueError("expected 784 inputs")
        selected = x[:, self.G.reshape(-1)].reshape(-1, 128, self.branches, self.fan_in)
        pre_d = (selected * self.W_d).sum(-1) + self.bias_d
        act_d = F.leaky_relu(pre_d, negative_slope=SLOPE)
        pre_s = (act_d * self.W_s).sum(-1) + self.bias_s
        act_s = F.leaky_relu(pre_s, negative_slope=SLOPE)
        logits = F.linear(act_s, self.W_o, self.bias_o)
        return pre_d, act_d, pre_s, act_s, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.intermediates(x)[-1]

    def counts(self) -> dict:
        return {"parameters": sum(p.numel() for p in self.parameters()),
                "main_macs_per_sample": self.W_d.numel() + self.W_s.numel() + self.W_o.numel(),
                "input_connections": self.W_d.numel()}


def initialize_paired(model: CompartmentalDSU, base_input: torch.Tensor,
                      base_soma8: torch.Tensor, output: torch.Tensor) -> dict:
    """Shared N(0,1) draws, Kaiming variance for LeakyReLU, nested soma draws."""
    b, k = model.branches, model.fan_in
    gain = math.sqrt(2 / (1 + SLOPE**2))
    soma_group = 8 // b
    soma_base = base_soma8.reshape(128, b, soma_group).sum(-1) / math.sqrt(soma_group)
    with torch.no_grad():
        model.W_d.copy_(base_input.reshape(128, b, k) * (gain / math.sqrt(k)))
        model.W_s.copy_(soma_base * (gain / math.sqrt(b)))
        model.W_o.copy_(output)
    return {"policy": "shared standard-normal input draws scaled sqrt(2/(1+0.1^2))/sqrt(K); shared eight soma draws aggregated by nested normalized sum and scaled sqrt(2/(1+0.1^2))/sqrt(B); shared Xavier-uniform output; zero biases",
            "base_input_sha256": sha(base_input), "base_soma8_sha256": sha(base_soma8),
            "output_sha256": sha(output), "input_scale": gain / math.sqrt(k),
            "soma_scale": gain / math.sqrt(b)}


def shared_initialization(seed: int):
    generator = torch.Generator().manual_seed(seed)
    base_input = torch.randn((128, 64), generator=generator)
    base_soma8 = torch.randn((128, 8), generator=generator)
    output = torch.empty(10, 128)
    bound = math.sqrt(6 / (128 + 10))
    output.uniform_(-bound, bound, generator=generator)
    return base_input, base_soma8, output


def initial_diagnostics(model: CompartmentalDSU, inputs: torch.Tensor) -> dict:
    names = ("dendrite_preactivation", "dendrite_activation", "soma_preactivation",
             "soma_activation", "logits")
    with torch.no_grad():
        values = model.intermediates(inputs)
    return {name: {"mean": value.mean().item(), "std": value.std().item()}
            for name, value in zip(names, values)}


def _load_data(config):
    train, validation, test, metadata = load_fashion_mnist(config)
    recorder = RecordingSampler(train.sampler, train.dataset.indices)
    train = DataLoader(train.dataset, batch_size=config.batch_size, sampler=recorder,
                       generator=train.generator)
    return train, validation, test, metadata, recorder


def run_arm(config, branches: int) -> dict:
    seed_everything(config.seed)
    g64 = base_topology(config.topology_seed)
    model = CompartmentalDSU(branches, g64)
    initialization = initialize_paired(model, *shared_initialization(config.seed))
    train, validation, test, dataset, recorder = _load_data(config)
    fixed_inputs = next(iter(validation))[0]
    diagnostics = initial_diagnostics(model, fixed_inputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-7, weight_decay=0)
    history = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        loss_sum = correct = count = 0
        for inputs, targets in train:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            count += len(targets)
        history.append({"epoch": epoch + 1, "loss": loss_sum / count,
                        "accuracy": correct / count,
                        "validation": evaluate(model, validation, torch.device("cpu"))})
    duration = time.perf_counter() - started
    result = {"schema": "s3_t05_v1", "run_utc": datetime.now(timezone.utc).isoformat(),
              "architecture": f"{branches}x{64 // branches}", "branches": branches,
              "fan_in": 64 // branches, "config": config.to_dict(),
              "seeds": {"python": config.seed, "numpy": config.seed, "torch": config.seed,
                        "cuda": None, "training_seed": config.seed,
                        "topology_seed": config.topology_seed},
              "topology": {"g64_sha256": sha(g64), "G_sha256": sha(model.G),
                           "unique_inputs_per_soma": 64, "partition": "contiguous nested blocks of G64"},
              "initialization": initialization, "initial_diagnostics": diagnostics,
              "dataset": dataset,
              "batch_order": {"epoch_sha256": recorder.epoch_sha256,
                              "combined_sha256": recorder.combined_sha256()},
              "counts": model.counts(),
              "training": {"optimizer": "Adam", "adam_betas": [0.9, 0.999],
                           "adam_epsilon": 1e-7, "weight_decay": 0, "scheduler": None,
                           "gradient_clipping": None, "mixed_precision": False,
                           "early_stopping": False, "checkpoint_selected": "final_epoch_25",
                           "history": history, "duration_seconds": duration},
              "test": evaluate(model, test, torch.device("cpu")),
              "environment": {"device": "cpu", "cpu": platform.processor() or platform.uname().processor,
                              "os": platform.platform(), "precision": "float32",
                              "python": platform.python_version(), "torch": torch.__version__,
                              "torchvision": version("torchvision"), "numpy": np.__version__}}
    return result


def validate_record(record: dict, seed: int, branches: int) -> None:
    config = record.get("config", {})
    history = record.get("training", {}).get("history", [])
    order = record.get("batch_order", {}).get("epoch_sha256", [])
    if (record.get("schema") != "s3_t05_v1" or record.get("branches") != branches
            or record.get("architecture") != f"{branches}x{64//branches}"
            or record.get("fan_in") != 64 // branches
            or config.get("seed") != seed or config.get("topology_seed") != seed
            or config.get("model") != "compartment_ablation"
            or config.get("dataset") != "fashion_mnist"
            or config.get("epochs") != 25 or config.get("batch_size") != 128
            or config.get("learning_rate") != 0.001 or config.get("device") != "cpu"
            or record.get("counts") != {"parameters": 9610 + 256 * branches,
                                         "main_macs_per_sample": 9472 + 128 * branches,
                                         "input_connections": 8192}
            or [epoch.get("epoch") for epoch in history] != list(range(1, 26))
            or len(order) != 25 or any(len(digest) != 64 for digest in order)
            or not all(math.isfinite(record["test"][key]) for key in ("loss", "accuracy"))):
        raise ValueError(f"incomplete or invalid record for {branches}x{64//branches} seed {seed}")
    if record["topology"]["g64_sha256"] != sha(base_topology(seed)):
        raise ValueError("topology hash mismatch")
    if record["topology"]["G_sha256"] != record["topology"]["g64_sha256"]:
        raise ValueError("partitioned topology hash mismatch")
    if record["dataset"]["splits"] != {"train": 54000, "validation": 6000, "test": 10000}:
        raise ValueError("dataset split sizes mismatch")
    if record["batch_order"]["combined_sha256"] != hashlib.sha256(
            b"".join(bytes.fromhex(digest) for digest in order)).hexdigest():
        raise ValueError("batch order digest mismatch")
    for epoch in history:
        if not all(math.isfinite(value) for value in (epoch["loss"], epoch["accuracy"],
                                                 epoch["validation"]["loss"],
                                                 epoch["validation"]["accuracy"])):
            raise ValueError("non-finite training history")
    expected_initialization = initialize_paired(
        CompartmentalDSU(branches, base_topology(seed)), *shared_initialization(seed))
    for digest in ("base_input_sha256", "base_soma8_sha256", "output_sha256"):
        if record["initialization"][digest] != expected_initialization[digest]:
            raise ValueError("initialization hash mismatch")


def _stats(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values),
            "values": values}


def summarize(records: list[dict]) -> dict:
    by_arm = {b: sorted((r for r in records if r["branches"] == b),
                        key=lambda r: r["config"]["seed"]) for b in BRANCHES}
    if any([r["config"]["seed"] for r in group] != list(SEEDS) for group in by_arm.values()):
        raise ValueError("summary requires all 12 trials")
    for seed in SEEDS:
        pair = [by_arm[b][seed - 1] for b in BRANCHES]
        for field in ("g64_sha256",):
            if len({r["topology"][field] for r in pair}) != 1:
                raise ValueError(f"unpaired topology seed {seed}")
        for field in ("base_input_sha256", "base_soma8_sha256", "output_sha256"):
            if len({r["initialization"][field] for r in pair}) != 1:
                raise ValueError(f"unpaired initialization seed {seed}")
        if len({r["batch_order"]["combined_sha256"] for r in pair}) != 1:
            raise ValueError(f"unpaired minibatches seed {seed}")
        for field in ("train_indices_sha256", "validation_indices_sha256"):
            if len({r["dataset"][field] for r in pair}) != 1:
                raise ValueError(f"unpaired split seed {seed}")
    arms = {}
    for b in BRANCHES:
        group = by_arm[b]
        arms[f"{b}x{64//b}"] = {
            "parameters": group[0]["counts"]["parameters"],
            "main_macs_per_sample": group[0]["counts"]["main_macs_per_sample"],
            "main_mac_ratio_vs_1x64": group[0]["counts"]["main_macs_per_sample"] / 9600,
            "test_loss": _stats([r["test"]["loss"] for r in group]),
            "test_accuracy": _stats([r["test"]["accuracy"] for r in group]),
            "validation_loss": _stats([r["training"]["history"][-1]["validation"]["loss"] for r in group]),
            "validation_accuracy": _stats([r["training"]["history"][-1]["validation"]["accuracy"] for r in group]),
            "quality_per_parameter": _stats([r["test"]["accuracy"] / r["counts"]["parameters"] for r in group]),
            "quality_per_main_mac": _stats([r["test"]["accuracy"] / r["counts"]["main_macs_per_sample"] for r in group]),
        }
    comparisons = {}
    for low, high in ((1, 2), (2, 4), (4, 8), (1, 4)):
        name = f"{high}x{64//high}_minus_{low}x{64//low}"
        comparisons[name] = {
            "test_accuracy_pp": _stats([(by_arm[high][i]["test"]["accuracy"] - by_arm[low][i]["test"]["accuracy"]) * 100 for i in range(3)]),
            "test_loss": _stats([by_arm[high][i]["test"]["loss"] - by_arm[low][i]["test"]["loss"] for i in range(3)])}
    delta = comparisons["4x16_minus_1x64"]
    acc = delta["test_accuracy_pp"]["mean"]
    loss = delta["test_loss"]["mean"]
    wins = sum(x > 0 for x in delta["test_accuracy_pp"]["values"])
    loss_wins = sum(x < 0 for x in delta["test_loss"]["values"])
    monotonic = all(arms[a]["test_accuracy"]["mean"] < arms[b]["test_accuracy"]["mean"]
                    for a, b in (("1x64", "2x32"), ("2x32", "4x16")))
    if acc >= 1.0 and wins == 3 and loss_wins == 3:
        classification = "STRONG_PRELIMINARY_SUPPORT"
    elif (0.30 <= acc < 1.0 and wins >= 2 and loss <= 0.01) or (monotonic and acc > 0 and loss <= 0.01):
        classification = "WEAK_SUPPORT"
    elif abs(acc) < 0.30 and abs(loss) < 0.01:
        classification = "NEGLIGIBLE"
    elif acc <= -0.50 and wins == 0:
        classification = "CONTRARY"
    else:
        classification = "INCONCLUSIVE"
    return {"schema": "s3_t05_summary_v1", "arms": arms, "comparisons": comparisons,
            "h011_classification": classification, "monotonic_1_2_4": monotonic,
            "ranking_by_mean_test_accuracy": sorted(arms, key=lambda x: arms[x]["test_accuracy"]["mean"], reverse=True),
            "seeds": list(SEEDS), "completed_utc": datetime.now(timezone.utc).isoformat()}


def write_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="S3-T05 nested compartment ablation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/compartment_s3_t05"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    if (base.model != "compartment_ablation" or base.dataset != "fashion_mnist"
            or base.epochs != 25 or base.batch_size != 128 or base.learning_rate != 0.001
            or base.device != "cpu" or base.train_samples is not None or base.test_samples is not None):
        parser.error("S3-T05 requires the frozen Fashion-MNIST CPU protocol")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in SEEDS:
        for branches in BRANCHES:
            path = args.output / f"{branches}x{64//branches}_seed{seed}.json"
            if path.exists() and not args.force:
                record = json.loads(path.read_text(encoding="utf-8"))
                validate_record(record, seed, branches)
                print(f"resumed {path}", flush=True)
            else:
                record = run_arm(replace(base, seed=seed, topology_seed=seed), branches)
                validate_record(record, seed, branches)
                write_json(path, record)
                print(f"completed {path}", flush=True)
            records.append(record)
    summary = summarize(records)
    write_json(args.output / "summary.json", summary)
    print(f"H-011={summary['h011_classification']}", flush=True)


if __name__ == "__main__":
    main()
