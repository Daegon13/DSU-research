"""Prospective iso-budget PointSparse screen. No historical outcome imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .analysis.structural_cost import ArchitectureSpec, calculate_cost
from .analysis.zero_cost import collect_activations, compute_proxies, input_presence, state_fingerprint
from .analysis.zero_cost_validation import kendall_tau_b, pairwise, spearman
from .compartment_ablation import base_topology, sha
from .config import ExperimentConfig
from .data import load_fashion_mnist
from .fixed_fanin_ablation import RecordingSampler
from .harness import evaluate, seed_everything
from .s3_t06_diagnostics import train_calibration_batch
from .sparse_compartment_controls import PointSparse, paired_values

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "data" / "s3_t10_preregistered_screen.json"
RUNS = ROOT / "runs" / "s3_t10_point_sparse_iso_budget"
COHORT = ((64, 139), (80, 109), (96, 89), (120, 69),
          (128, 64), (160, 49), (192, 39), (240, 29))
METRICS = ("effective_rank_raw", "effective_rank_norm", "participation_ratio_raw",
           "participation_ratio_norm", "mean_abs_corr", "activation_std", "logit_std",
           "entropy", "near_zero_fraction", "unique_input_features", "coverage_ratio",
           "mean_references_per_feature", "reference_count_sd", "max_reference_count",
           "mean_jaccard_between_units")
RANK_METRICS = ("effective_rank_raw", "effective_rank_norm", "participation_ratio_raw",
                "participation_ratio_norm", "mean_abs_corr")
NEAR_TIE_ACC = 0.002  # 0.20 percentage points, strict less-than


def architecture_id(h, k):
    return f"PointSparse_{h}x{k}"


def topology(h, k, seed):
    if not (isinstance(h, int) and isinstance(k, int) and 1 <= k <= 784 and h > 0):
        raise ValueError("invalid H/K")
    if h == 128 and k == 64:
        return base_topology(seed)
    rng = np.random.default_rng(seed)
    return torch.from_numpy(np.stack([rng.choice(784, k, replace=False) for _ in range(h)]).astype(np.int64))


class IsoPointSparse(PointSparse):
    """Same compact PointSparse operation, variable H/K and independent RNGs."""

    def __init__(self, h, k, topology_seed, training_seed):
        nn.Module.__init__(self)
        self.h, self.k = h, k
        self.register_buffer("G64", topology(h, k, topology_seed))
        self.W_in = nn.Parameter(torch.empty(h, k))
        self.bias_s = nn.Parameter(torch.zeros(h))
        self.W_o = nn.Parameter(torch.empty(10, h))
        self.bias_o = nn.Parameter(torch.zeros(10))
        generator = torch.Generator().manual_seed(training_seed)
        with torch.no_grad():
            self.W_in.copy_(torch.randn((h, k), generator=generator) *
                            math.sqrt(2 / (1 + 0.1**2) / k))
            self.W_o.uniform_(-math.sqrt(6 / (h + 10)), math.sqrt(6 / (h + 10)),
                              generator=generator)

    def _selected(self, x):
        x = x.flatten(1)
        if x.shape[1] != 784:
            raise ValueError("expected 784 inputs")
        return x[:, self.G64.reshape(-1)].reshape(-1, self.h, self.k)


def build_model(h, k, training_seed, topology_seed):
    if (h, k) not in COHORT:
        raise ValueError("architecture outside frozen cohort")
    if (h, k) == (128, 64):
        # Exact S3-T07 anchor, including its initialization and topology.
        model = PointSparse(base_topology(topology_seed))
        model.initialize(*paired_values(training_seed))
    else:
        model = IsoPointSparse(h, k, topology_seed, training_seed)
    if sum(p.numel() for p in model.parameters()) != 9610:
        raise AssertionError("iso-budget parameter mismatch")
    return model


def structural(h, k):
    cost = calculate_cost(ArchitectureSpec(family="point_sparse", somas=h, branches=1,
                                           fan_in=k, batch_size=128))
    if cost.trainable_parameters != h * (k + 11) + 10 or cost.trainable_parameters != 9610:
        raise AssertionError("structural budget mismatch")
    if cost.main_macs_per_sample != h * (k + 10):
        raise AssertionError("MAC mismatch")
    return asdict(cost)


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def preregister():
    """Only train images, untrained forward, and structural accounting are reachable."""
    if PREREG.exists():
        raise FileExistsError("preregistration already exists; never overwrite it")
    torch.set_num_threads(1)
    batches, rows = [], []
    samples = {}
    for seed in (1, 2, 3):
        x, indices_sha = train_calibration_batch(seed, size=256)
        pixels_sha = hashlib.sha256(x.contiguous().numpy().astype("<f4", copy=False).tobytes()).hexdigest()
        samples[seed] = x
        batches.append({"seed": seed, "train_indices_sha256": indices_sha, "pixel_sha256": pixels_sha})
    for h, k in COHORT:
        recs = []
        for seed in (1, 2, 3):
            start = time.perf_counter()
            model = build_model(h, k, seed, seed)
            metrics = compute_proxies(collect_activations(model, samples[seed]), input_presence(model))
            if any(p.grad is not None for p in model.parameters()):
                raise AssertionError("proxy created gradient")
            recs.append({"seed": seed, "topology_sha256": sha(model.G64),
                         "initialization_sha256": state_fingerprint(model),
                         "sample_fingerprint": batches[seed - 1],
                         "proxy_runtime_seconds": time.perf_counter() - start,
                         "metrics": {name: metrics[name] for name in METRICS},
                         "er_per_width": metrics["effective_rank_raw"] / h})
        summary = {name: {"mean": statistics.mean(r["metrics"][name] for r in recs),
                          "sample_sd": statistics.stdev(r["metrics"][name] for r in recs)}
                   for name in METRICS}
        rows.append({"architecture_id": architecture_id(h, k), "H": h, "K": k,
                     "structural_cost": structural(h, k), "proxy_seeds": recs,
                     "proxy_summary": summary,
                     "er_per_width": {"mean": statistics.mean(r["er_per_width"] for r in recs),
                                      "sample_sd": statistics.stdev(r["er_per_width"] for r in recs)}})
    def rank(name, reverse=True):
        return [r["architecture_id"] for r in sorted(rows,
                key=lambda r: (r["proxy_summary"][name]["mean"], -r["H"]), reverse=reverse)]
    rankings = {name: rank(name, name != "mean_abs_corr") for name in RANK_METRICS}
    rankings["width_H"] = [architecture_id(h, k) for h, k in reversed(COHORT)]
    primary = rankings["effective_rank_raw"]
    data = {"schema": "s3_t10_prereg_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit_before_prereg": _git("rev-parse", "HEAD"),
            "cohort": [{"H": h, "K": k} for h, k in COHORT],
            "protocol": {"proxy_seeds": [1, 2, 3], "proxy_train_samples": 256,
                         "proxy_model_dtype": "float32", "proxy_statistics_dtype": "float64",
                         "pilot_training_seed": 1, "pilot_topology_seed": 1, "pilot_epochs": 3,
                         "pilot_batch_size": 128, "pilot_lr": 0.001,
                         "pilot_optimizer": "Adam(betas=(0.9,0.999),eps=1e-7,weight_decay=0)",
                         "pilot_loss": "CrossEntropyLoss", "activation": "LeakyReLU(0.1)",
                         "pilot_target": "validation_accuracy_epoch_3",
                         "secondary_target": "negative_validation_loss_epoch_3",
                         "accuracy_near_tie_strict_less_than": NEAR_TIE_ACC,
                         "negative_loss_near_tie_strict_less_than": 0.005,
                         "test_access": False},
            "hypotheses_before_pilot": {"H-WIDTH-FANIN": "UNTESTED", "H-ZEROCOST-ER": "UNTESTED"},
            "primary_proxy": "effective_rank_raw", "rankings": rankings,
            "predicted_top_3": primary[:3], "predicted_bottom_2": primary[-2:],
            "success_criteria": {"VALIDATED_PROMISING": "Spearman >=0.60; pairwise >0.70 excluding accuracy near-ties; ER clearly equals or exceeds width H",
                                 "PARTIAL": "positive but Spearman <0.60, or ER does not clearly exceed width",
                                 "FAILED": "near-zero/negative correlation or materially misleading ranking"},
            "batch_fingerprints": batches, "architectures": rows,
            "environment": {"cpu": platform.processor() or platform.uname().processor,
                            "os": platform.platform(), "torch": torch.__version__,
                            "torch_num_threads": torch.get_num_threads()}}
    _write(PREREG, data)
    return data


def sealed_prereg():
    """Require a committed, unchanged prereg before any pilot result is read."""
    raw = PREREG.read_bytes()
    commit = _git("log", "-1", "--format=%H", "--", "data/s3_t10_preregistered_screen.json")
    if not commit:
        raise RuntimeError("preregistration must be committed before pilot")
    committed = subprocess.check_output(["git", "show", f"{commit}:data/s3_t10_preregistered_screen.json"], cwd=ROOT)
    if raw != committed:
        raise RuntimeError("preregistration must be committed unchanged before pilot")
    data = json.loads(raw)
    if data["schema"] != "s3_t10_prereg_v1" or data["primary_proxy"] != "effective_rank_raw":
        raise ValueError("invalid preregistration")
    return data, hashlib.sha256(raw).hexdigest(), commit


def _pilot_config():
    return ExperimentConfig(model="s3_t10_point_sparse", dataset="fashion_mnist", seed=1,
                            topology_seed=1, epochs=3, batch_size=128, learning_rate=0.001,
                            device="cpu")


def run_pilot(h, k, prereg_sha, prereg_commit):
    config = _pilot_config()
    seeds = seed_everything(1)
    torch.set_num_threads(1)
    model = build_model(h, k, 1, 1)
    initial_sha = state_fingerprint(model)
    train, validation, test, dataset = load_fashion_mnist(config, include_test=False)
    if test is not None:
        raise AssertionError("test split was opened")
    recorder = RecordingSampler(train.sampler, train.dataset.indices)
    train = DataLoader(train.dataset, batch_size=128, sampler=recorder, generator=train.generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-7,
                                 weight_decay=0)
    history = []
    start = time.perf_counter()
    for epoch in range(1, 4):
        model.train()
        total = count = 0
        for inputs, targets in train:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
        history.append({"epoch": epoch, "train_loss": total / count,
                        "validation": evaluate(model, validation, torch.device("cpu"))})
    return {"schema": "s3_t10_pilot_v1", "architecture_id": architecture_id(h, k),
            "H": h, "K": k, "prereg_sha256": prereg_sha, "prereg_commit": prereg_commit,
            "seeds": {**seeds, "training_seed": 1, "topology_seed": 1},
            "dataset": dataset, "test_access": False,
            "topology_sha256": sha(model.G64), "initialization_sha256": initial_sha,
            "structural_cost": structural(h, k), "batch_order": recorder.epoch_sha256,
            "history": history, "duration_seconds": time.perf_counter() - start}


def _validate_pilot(record, row, prereg_sha, prereg_commit):
    h, k = row["H"], row["K"]
    if (record.get("schema") != "s3_t10_pilot_v1" or
        record.get("architecture_id") != architecture_id(h, k) or
        (record.get("H"), record.get("K")) != (h, k) or
        record.get("prereg_sha256") != prereg_sha or
        record.get("prereg_commit") != prereg_commit or
        record.get("test_access") is not False or "test" in record or
        record.get("structural_cost") != row["structural_cost"] or
        record.get("topology_sha256") != row["proxy_seeds"][0]["topology_sha256"] or
        record.get("initialization_sha256") != row["proxy_seeds"][0]["initialization_sha256"] or
        [r.get("epoch") for r in record.get("history", [])] != [1, 2, 3] or
        len(record.get("batch_order", [])) != 3):
        raise ValueError(f"invalid resume record: {architecture_id(h, k)}")
    for entry in record["history"]:
        if not all(math.isfinite(v) for v in (entry["train_loss"], entry["validation"]["loss"],
                                               entry["validation"]["accuracy"])):
            raise ValueError("nonfinite pilot metric")


def pilot():
    prereg, prereg_sha, commit = sealed_prereg()
    RUNS.mkdir(parents=True, exist_ok=True)
    records = []
    for row in prereg["architectures"]:
        h, k = row["H"], row["K"]
        path = RUNS / f"{architecture_id(h, k)}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = run_pilot(h, k, prereg_sha, commit)
            _validate_pilot(record, row, prereg_sha, commit)
            _write(path, record)
        _validate_pilot(record, row, prereg_sha, commit)
        records.append(record)
        print(f"{record['architecture_id']}: epoch 3 val accuracy {record['history'][-1]['validation']['accuracy']:.4f}", flush=True)
    return prereg, records, prereg_sha, commit


def analyze(prereg, records, prereg_sha, commit):
    if len(records) != len(COHORT):
        raise ValueError("all eight pilots required")
    rows = prereg["architectures"]
    for row, record in zip(rows, records):
        _validate_pilot(record, row, prereg_sha, commit)
    acc = [r["history"][-1]["validation"]["accuracy"] for r in records]
    neg_loss = [-r["history"][-1]["validation"]["loss"] for r in records]
    comparisons = {}
    for target_name, target, tolerance in (("validation_accuracy_epoch_3", acc, NEAR_TIE_ACC),
                                           ("negative_validation_loss_epoch_3", neg_loss, .005)):
        comparisons[target_name] = {}
        for name in (*RANK_METRICS[:4], "width_H"):
            values = [row["H"] if name == "width_H" else row["proxy_summary"][name]["mean"]
                      for row in rows]
            comparisons[target_name][name] = {"spearman": spearman(values, target),
                                              "kendall_tau_b": kendall_tau_b(values, target),
                                              "pairwise": pairwise(values, target, tolerance)}
    er = comparisons["validation_accuracy_epoch_3"]["effective_rank_raw"]
    width = comparisons["validation_accuracy_epoch_3"]["width_H"]
    er_pair = er["pairwise"]["accuracy_excluding_near_ties"]
    width_pair = width["pairwise"]["accuracy_excluding_near_ties"]
    er_adds = er["spearman"] > width["spearman"] and er_pair > width_pair
    if er["spearman"] is not None and er["spearman"] >= .60 and er_pair is not None and er_pair > .70 and er_adds:
        er_status = "VALIDATED_PROMISING"
    elif er["spearman"] is not None and er["spearman"] > 0 and (er_pair is None or er_pair >= .5):
        er_status = "PARTIAL"
    else:
        er_status = "FAILED"
    best = rows[max(range(len(acc)), key=lambda i: acc[i])]["architecture_id"]
    picks = list(dict.fromkeys((best, prereg["rankings"]["effective_rank_raw"][0],
                               architecture_id(128, 64))))
    spread_pp = (max(acc) - min(acc)) * 100
    width_status = ("PRELIMINARY_SIGNAL" if spread_pp >= .20 else
                    "NO_PRELIMINARY_SIGNAL" if spread_pp < .10 else "INCONCLUSIVE")
    result = {"schema": "s3_t10_analysis_v1", "prereg_sha256": prereg_sha,
              "prereg_commit": commit, "pilot_target_values": [
                  {"architecture_id": row["architecture_id"], "accuracy": a,
                   "validation_loss": -n} for row, a, n in zip(rows, acc, neg_loss)],
              "actual_accuracy_ranking": [rows[i]["architecture_id"] for i in
                                          sorted(range(len(rows)), key=lambda i: acc[i], reverse=True)],
              "comparisons": comparisons, "er_adds_over_width": er_adds,
              "H-ZEROCOST-ER": er_status, "H-WIDTH-FANIN": width_status,
              "pilot_accuracy_spread_pp": spread_pp, "stage_3_shortlist": picks,
              "epoch_equivalents": {"realized": 8 * 3, "naive": 8 * 25 * 3,
                                    "reduction_fraction": 1 - (8 * 3) / (8 * 25 * 3)}}
    _write(RUNS / "analysis.json", result)
    _write(ROOT / "data" / "s3_t10_pilot_results.json",
           {"schema": "s3_t10_results_v1", "prereg_sha256": prereg_sha,
            "prereg_commit": commit, "pilot_records": records, "analysis": result})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preregister", "pilot", "analyze"))
    args = parser.parse_args()
    if args.phase == "preregister":
        print(json.dumps(preregister()["rankings"], indent=2))
    elif args.phase == "pilot":
        prereg, records, sha256, commit = pilot()
        print(f"8 pilot records sealed to prereg commit {commit[:12]}")
    else:
        prereg, sha256, commit = sealed_prereg()
        rows = prereg["architectures"]
        records = [json.loads((RUNS / f"{row['architecture_id']}.json").read_text(encoding="utf-8"))
                   for row in rows]
        print(json.dumps(analyze(prereg, records, sha256, commit), indent=2))


if __name__ == "__main__":
    main()
