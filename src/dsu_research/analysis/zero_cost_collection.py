"""S3-T09 phase 1: collect untrained, label-free forward proxies only."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from dsu_research.compartment_ablation import (CompartmentalDSU, base_topology,
                                                initialize_paired, shared_initialization)
from dsu_research.dsu_s import DSUSv0
from dsu_research.fixed_dann import FixedDANN
from dsu_research.fixed_fanin_ablation import build_paired_models
from dsu_research.harness import seed_everything
from dsu_research.model import (CapacityMLP, DendriticANNRandom,
                                ParameterMatchedMLP, VanillaANN)
from dsu_research.s3_t06_diagnostics import train_calibration_batch
from dsu_research.sparse_compartment_controls import NeutralCompartments, PointSparse
from dsu_research.sparse_compartment_experiment import build_arm

from .zero_cost import collect_activations, compute_proxies, input_presence, state_fingerprint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "data" / "zero_cost_proxy_results.json"
SEEDS = (1, 2, 3)
ARMS = (
    ("S3-T07|PointSparse", "s3_t07", "PointSparse"),
    *((f"S3-T07|parameter-neutral {b}x{64 // b}", "s3_t07", f"{b}x{64 // b}")
      for b in (1, 2, 4, 8)),
    *((f"S3-T05|original compartment {b}x{64 // b}", "s3_t05", str(b))
      for b in (1, 2, 4, 8)),
    ("S2-T05|MLP 13->17", "s2_t05", "13"),
    ("S2-T06|MLP 20->26", "s2_t06", "20"),
    ("REPRODUCTION_001|vANN", "s1_n3", "vann"),
    ("REPRODUCTION_001|dANN-R", "s1_n3", "dann_r"),
    ("S3-T03|dANN-R", "s3_t03", "dann_r"),
    ("S3-T03|Fixed-dANN", "s3_t03", "fixed_dann"),
    ("S2-T04|DSU-S", "s2_t04", "dsu_s"),
)


def build_untrained_model(adapter: str, arm: str, seed: int, inputs: torch.Tensor):
    seed_everything(seed)
    if adapter == "s3_t07":
        model, _, _ = build_arm(seed, arm, inputs)
        return model, 3 if arm != "PointSparse" else 1
    if adapter == "s3_t05":
        model = CompartmentalDSU(int(arm), base_topology(seed))
        initialize_paired(model, *shared_initialization(seed))
        return model, 1
    if adapter == "s2_t05":
        return ParameterMatchedMLP(), 1
    if adapter == "s2_t06":
        return CapacityMLP(20), 1
    if adapter == "s1_n3":
        return (VanillaANN() if arm == "vann" else DendriticANNRandom(seed)), 1
    if adapter == "s3_t03":
        dann, fixed, _ = build_paired_models(seed, seed)
        return (dann if arm == "dann_r" else fixed), 1
    if adapter == "s2_t04":
        return DSUSv0(seed), 1
    raise ValueError(f"unknown adapter {adapter}")


def forward_temporary_max_bytes(model: torch.nn.Module, batch_size: int) -> int:
    """Largest known forward intermediate beyond the statistics workspace."""
    if isinstance(model, (PointSparse, NeutralCompartments)):
        return batch_size * model.G64.numel() * 4  # gathered float32 inputs
    if isinstance(model, (CompartmentalDSU, DSUSv0)):
        return batch_size * model.G.numel() * 4
    if isinstance(model, (FixedDANN, DendriticANNRandom)):
        return model.dendrites.weight.numel() * 4  # dense masked float32 weight
    return 0


def collect(seeds=SEEDS, arms=ARMS):
    """Cannot inspect historical outcomes: this module never loads the registry."""
    torch.set_num_threads(1)
    records = []
    batches = []
    for seed in seeds:
        started = time.perf_counter_ns()
        inputs, indices_sha = train_calibration_batch(seed, size=256)
        pixels_sha = hashlib.sha256(inputs.contiguous().numpy().astype("<f4", copy=False).tobytes()).hexdigest()
        batches.append({"seed": seed, "train_indices_sha256": indices_sha,
                        "pixel_sha256": pixels_sha,
                        "batch_loading_ms": (time.perf_counter_ns() - started) / 1e6})
        for architecture_id, adapter, arm in arms:
            start_build = time.perf_counter_ns()
            model, forward_passes = build_untrained_model(adapter, arm, seed, inputs)
            fingerprint = state_fingerprint(model)
            build_ms = (time.perf_counter_ns() - start_build) / 1e6
            start_proxy = time.perf_counter_ns()
            values = collect_activations(model, inputs)
            metrics = compute_proxies(values, input_presence(model))
            metrics["peak_tensor_size_bytes_estimate"] = max(
                metrics["peak_tensor_size_bytes_estimate"],
                forward_temporary_max_bytes(model, len(inputs)))
            proxy_ms = (time.perf_counter_ns() - start_proxy) / 1e6
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("forward proxy created parameter gradients")
            records.append({
                "architecture_id": architecture_id, "proxy_seed": seed,
                "protocol_adapter": adapter,
                "sample_fingerprint": {"train_indices_sha256": indices_sha,
                                       "pixel_sha256": pixels_sha},
                "initialization_fingerprint": fingerprint,
                "forward_passes": forward_passes,
                "build_calibration_ms": build_ms, "proxy_ms": proxy_ms,
                "runtime_ms": build_ms + proxy_ms,
                **metrics,
            })
    return {"schema_version": 1, "task": "S3-T09", "phase": "forward_proxies_only",
            "accounting_revision": "Post-result correction to largest logical tensor estimate: include compact gather or dense masked weight; all ranking proxies, fingerprints, and timings unchanged. Median absolute correlation corrected to conventional midpoint for even pair counts; other ranking fields and timing unchanged.",
            "frozen_protocol": "docs/EXPERIMENT_009_ZERO_COST_FORWARD_PROXIES.md",
            "seeds": list(seeds), "calibration_sample_count": 256,
            "device": "cpu", "model_dtype": "float32", "statistics_dtype": "float64",
            "torch_num_threads": torch.get_num_threads(), "host": platform.processor(),
            "batch_fingerprints": batches, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    times = [record["runtime_ms"] for record in result["records"]]
    print(f"{len(times)} forward-only records; mean runtime {statistics.mean(times):.1f} ms; {args.output}")


if __name__ == "__main__":
    main()
