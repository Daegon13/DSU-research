"""A1-E01: preregistered PointSparse topology contrast, with resumable runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from .analysis.zero_cost import state_fingerprint
from .compartment_ablation import sha
from .config import ExperimentConfig
from .data import load_fashion_mnist
from .fixed_fanin_ablation import RecordingSampler
from .harness import evaluate, seed_everything
from .pointsparse_topology import graph_hash, make_topology, structural_metrics
from .s3_t11 import audit_anchor
from .sparse_compartment_controls import PointSparse, paired_values
from .sparse_compartment_experiment import diagnostic_evaluate

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "a1_e01"
T07 = ROOT / "runs" / "sparse_compartment_s3_t07"
ARMS = ("A1", "A2", "A3")
SEEDS = (1, 2, 3)
FROZEN_SHA256 = {
    "docs/A1_POINTSPARSE_TOPOLOGY_DESIGN.md": "4dd30164641a1f8f9985fc53fe7d1afe05df7dcdfd10d8dd4c49e201dfb7daf0",
    "src/dsu_research/pointsparse_topology.py": "42b88751cb496a9093f3b476e90869839357661abcbd09a237544d12ec3f1a5a",
    "data/a1_topology_structural.json": "8de4f3a87049d500a1ee1bcf90b47be9a4f3f465602ded9058170ad4dd20a609",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def config(seed):
    return ExperimentConfig(model="a1_e01_point_sparse", dataset="fashion_mnist",
                            seed=seed, topology_seed=seed, epochs=25, batch_size=128,
                            learning_rate=.001, device="cpu")


def preflight():
    """Stop before training if any preregistered graph or budget has drifted."""
    for relative, expected in FROZEN_SHA256.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"frozen A1 source drift: {relative}")
    artifact = json.loads((ROOT / "data" / "a1_topology_structural.json").read_text(encoding="utf-8"))
    if artifact.get("schema") != "a1_topology_structural_v1" or len(artifact["records"]) != 12:
        raise ValueError("invalid frozen structural artifact")
    rows = {(row["topology_seed"], row["arm"]): row for row in artifact["records"]}
    if len(rows) != 12:
        raise ValueError("duplicate structural record")
    for seed in SEEDS:
        degree_vector = None
        values = paired_values(seed)
        for arm in ("A0", *ARMS):
            graph, generation = make_topology(arm, seed)
            metrics = structural_metrics(graph)
            frozen = rows[seed, arm]
            model = PointSparse(graph)
            model.initialize(*values)
            if (generation != frozen["generation"] or metrics != frozen["metrics"] or
                graph_hash(graph) != generation["final_hash"] or graph.shape != (128, 64) or
                graph.numel() != 8192 or
                any(torch.unique(row).numel() != 64 for row in graph) or
                model.counts()["parameters"] != 9610 or
                model.counts()["main_macs_per_sample"] != 9472):
                raise ValueError(f"structural drift: {arm} seed {seed}")
            if arm != "A0":
                degrees = torch.bincount(graph.flatten(), minlength=784)
                if (metrics["coverage"] != 784 or int((degrees == 11).sum()) != 352 or
                    int((degrees == 10).sum()) != 432 or
                    (degree_vector is not None and not torch.equal(degrees, degree_vector))):
                    raise ValueError(f"degree drift: {arm} seed {seed}")
                degree_vector = degrees
        a1, a2, a3 = (rows[seed, arm]["metrics"] for arm in ARMS)
        if (a1["mean_jaccard"] - a2["mean_jaccard"] < .00015 or
            a3["mean_jaccard"] - a1["mean_jaccard"] < .001 or
            a1["max_jaccard"] - a2["max_jaccard"] < .02 or
            a3["max_jaccard"] - a1["max_jaccard"] < .20):
            raise ValueError(f"structural separation failed: seed {seed}")
    # The S3-T11 audit revalidates S3-T07 records, model class, weights,
    # optimizer, split, minibatch metadata and final checkpoint before reuse.
    anchor = audit_anchor()
    if anchor["decision"] != "REUSE_S3_T07_N3":
        raise ValueError("A0 reuse audit failed")
    return {"schema": "a1_e01_preflight_v1", "source_commit": "4ce1fc7",
            "frozen_sha256": FROZEN_SHA256, "structural_records": 12,
            "structural_separation": "PASS", "a0_reuse": anchor}


def build_model(arm, seed):
    graph, generation = make_topology(arm, seed)
    model = PointSparse(graph)
    initialization = model.initialize(*paired_values(seed))
    return model, generation, initialization


def run_one(arm, seed, preflight_result):
    """Test dataset is instantiated only after epoch 25 and pairing checks."""
    seed_everything(seed)
    torch.set_num_threads(1)
    model, generation, initialization = build_model(arm, seed)
    initial_state_sha = state_fingerprint(model)
    cfg = config(seed)
    train, validation, test, dataset = load_fashion_mnist(cfg, include_test=False)
    if test is not None:
        raise RuntimeError("test opened before training")
    reference = json.loads((T07 / f"PointSparse_seed{seed}.json").read_text(encoding="utf-8"))
    if any(dataset[key] != reference["dataset"][key] for key in
           ("splits", "split_rule", "train_indices_sha256", "validation_indices_sha256", "transform")):
        raise ValueError("historical split mismatch")
    recorder = RecordingSampler(train.sampler, train.dataset.indices)
    train = DataLoader(train.dataset, batch_size=128, sampler=recorder, generator=train.generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999),
                                 eps=1e-7, weight_decay=0)
    history = []
    started = time.perf_counter()
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
    duration = time.perf_counter() - started
    if (recorder.epoch_sha256 != reference["batch_order"]["epoch_sha256"] or
        recorder.combined_sha256() != reference["batch_order"]["combined_sha256"]):
        raise ValueError("historical minibatch order mismatch; test remains unopened")
    official_test = FashionMNIST(root="data", train=False, download=True,
                                 transform=transforms.ToTensor())
    test_metrics = diagnostic_evaluate(
        model, DataLoader(official_test, batch_size=128, shuffle=False))
    graph = model.G64
    return {"schema": "a1_e01_run_v1", "arm": arm, "seed": seed,
            "source_commit": "4ce1fc7", "frozen_sha256": preflight_result["frozen_sha256"],
            "seeds": {"training_seed": seed, "topology_seed": seed,
                      "python": seed, "numpy": seed, "torch": seed},
            "protocol": cfg.to_dict(), "architecture": "PointSparse_128x64",
            "topology": {"G64": graph.tolist(), "sha256": sha(graph),
                         "generation": generation, "metrics": structural_metrics(graph)},
            "initialization": initialization, "initial_state_sha256": initial_state_sha,
            "counts": model.counts(), "dataset": dataset,
            "batch_order": {"epoch_sha256": recorder.epoch_sha256,
                            "combined_sha256": recorder.combined_sha256()},
            "training": {"optimizer": "Adam", "betas": [.9, .999], "epsilon": 1e-7,
                         "weight_decay": 0, "scheduler": None, "gradient_clipping": None,
                         "mixed_precision": False, "early_stopping": False,
                         "checkpoint_selected": "final_epoch_25", "history": history,
                         "duration_seconds": duration},
            "test": test_metrics, "environment": {"device": "cpu", "precision": "float32",
            "cpu": platform.processor() or platform.uname().processor,
            "os": platform.platform(), "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads()}}


def validate_run(record, arm, seed, preflight_result):
    """Validate a completed JSON before accepting it on resume or for analysis."""
    model, generation, initialization = build_model(arm, seed)
    graph = model.G64
    old = json.loads((T07 / f"PointSparse_seed{seed}.json").read_text(encoding="utf-8"))
    training = record.get("training", {})
    order = record.get("batch_order", {})
    if (record.get("schema") != "a1_e01_run_v1" or record.get("arm") != arm or
        record.get("seed") != seed or record.get("source_commit") != "4ce1fc7" or
        record.get("frozen_sha256") != preflight_result["frozen_sha256"] or
        record.get("seeds", {}).get("training_seed") != seed or
        record.get("seeds", {}).get("topology_seed") != seed or
        record.get("protocol") != config(seed).to_dict() or
        record.get("architecture") != "PointSparse_128x64" or
        record.get("topology", {}).get("G64") != graph.tolist() or
        record["topology"].get("sha256") != sha(graph) or
        record["topology"].get("generation") != generation or
        record["topology"].get("metrics") != structural_metrics(graph) or
        record.get("initialization") != initialization or
        record.get("initial_state_sha256") != state_fingerprint(model) or
        record.get("counts") != model.counts() or
        any(record.get("dataset", {}).get(key) != old["dataset"][key] for key in
            ("splits", "split_rule", "train_indices_sha256", "validation_indices_sha256", "transform")) or
        order.get("epoch_sha256") != old["batch_order"]["epoch_sha256"] or
        order.get("combined_sha256") != old["batch_order"]["combined_sha256"] or
        training.get("optimizer") != "Adam" or training.get("betas") != [.9, .999] or
        training.get("epsilon") != 1e-7 or training.get("weight_decay") != 0 or
        training.get("checkpoint_selected") != "final_epoch_25" or
        [row.get("epoch") for row in training.get("history", [])] != list(range(1, 26)) or
        set(record.get("test", {})) != set(old["test"])):
        raise ValueError(f"invalid or unpaired A1 record: {arm} seed {seed}")
    metrics = [training["duration_seconds"], *record["test"].values()]
    for row in training["history"]:
        metrics.extend((row["train_loss"], row["validation"]["loss"],
                        row["validation"]["accuracy"]))
    if not all(math.isfinite(value) for value in metrics):
        raise ValueError(f"nonfinite A1 record: {arm} seed {seed}")


def train():
    gate = preflight()
    RUNS.mkdir(parents=True, exist_ok=True)
    write_json(RUNS / "preflight.json", gate)
    for seed in SEEDS:
        for arm in ARMS:
            path = RUNS / f"{arm}_seed{seed}.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                validate_run(record, arm, seed, gate)
                print(f"resumed {arm} seed {seed}", flush=True)
            else:
                record = run_one(arm, seed, gate)
                validate_run(record, arm, seed, gate)
                write_json(path, record)
                print(f"completed {arm} seed {seed} after 25 epochs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "train"))
    args = parser.parse_args()
    if args.phase == "preflight":
        print(json.dumps(preflight(), indent=2))
    else:
        train()


if __name__ == "__main__":
    main()
