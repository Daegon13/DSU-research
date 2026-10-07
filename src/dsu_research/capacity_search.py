"""S2-T06 validation-only capacity search; test is a separate frozen phase."""

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from .config import load_config
from .data import load_fashion_mnist
from .harness import count_parameters, evaluate, seed_everything
from .model import CapacityMLP

LADDER = (16, 19, 22, 25, 28, 32)
SEEDS = (1, 2, 3)
ACCURACY_TOLERANCE = 0.005
LOSS_TOLERANCE = 0.020


def summary(values):
    return {"values": values, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def target_from_existing(directory: Path):
    records = [json.loads((directory / f"dsu_s_seed{seed}.json").read_text(encoding="utf-8"))
               for seed in SEEDS]
    assert all(len(r["training"]["history"]) == 25 and
               r["training"]["checkpoint_selected"] == "final_epoch_25" and
               r["config"]["seed"] == seed for seed, r in zip(SEEDS, records))
    final = [r["training"]["history"][-1]["validation"] for r in records]
    return {"loss": summary([r["loss"] for r in final]),
            "accuracy": summary([r["accuracy"] for r in final])}


def matches(candidate, target):
    accuracy = candidate["accuracy"]["mean"] >= target["accuracy"]["mean"] - ACCURACY_TOLERANCE
    loss = candidate["loss"]["mean"] <= target["loss"]["mean"] + LOSS_TOLERANCE
    return {"accuracy": accuracy, "loss": loss, "full": accuracy and loss}


def train_one(base_config, hidden1, seed, output_dir):
    config = replace(base_config, seed=seed)
    seed_everything(seed)
    train_loader, val_loader, _unused_test_loader, dataset = load_fashion_mnist(config, include_test=False)
    assert _unused_test_loader is None
    model = CapacityMLP(hidden1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7)
    history = []
    start = time.perf_counter()
    for epoch in range(25):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
        history.append({"epoch": epoch + 1, "validation": evaluate(model, val_loader, torch.device("cpu"))})
    duration = time.perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"h{hidden1}_seed{seed}.pt"
    torch.save(model.state_dict(), checkpoint)
    result = {
        "hidden1": hidden1, "hidden2": model.hidden2.out_features, "seed": seed,
        "config": config.to_dict(), "dataset": dataset,
        "parameters": {**count_parameters(model), "stored": count_parameters(model)["total"],
                       **model.structural_metrics()},
        "optimizer": {"name": "Adam", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-7},
        "history": history, "checkpoint_selected": "final_epoch_25",
        "duration_train_validation_seconds": duration, "checkpoint": str(checkpoint),
    }
    (output_dir / f"h{hidden1}_seed{seed}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"h={hidden1} seed={seed} validation={history[-1]['validation']} duration={duration:.1f}s", flush=True)
    return result


def candidate(base_config, hidden1, output_dir, target):
    records = [train_one(base_config, hidden1, seed, output_dir) for seed in SEEDS]
    final = [r["history"][-1]["validation"] for r in records]
    result = {
        "hidden1": hidden1, "hidden2": records[0]["hidden2"],
        "parameters": records[0]["parameters"],
        "loss": summary([r["loss"] for r in final]),
        "accuracy": summary([r["accuracy"] for r in final]),
        "duration_train_validation_seconds": summary([r["duration_train_validation_seconds"] for r in records]),
    }
    result["match"] = matches(result, target)
    print(f"candidate h={hidden1}: {result['match']}", flush=True)
    return result


def search(config_path, output_dir, dsu_dir, baseline_dir):
    base = load_config(config_path)
    if (base.dataset, base.epochs, base.batch_size, base.learning_rate, base.device) != (
            "fashion_mnist", 25, 128, 0.001, "cpu"):
        raise ValueError("S2-T06 fixed protocol mismatch")
    if base.topology_seed is not None or base.model != "capacity_mlp":
        raise ValueError("S2-T06 requires capacity_mlp without topology seed")
    if (output_dir / "frozen_selection.json").exists():
        raise FileExistsError("Selection already frozen")
    target = target_from_existing(dsu_dir)
    baseline = [json.loads((baseline_dir / f"mlp_seed{seed}.json").read_text(encoding="utf-8"))
                for seed in SEEDS]
    base_final = [r["training"]["history"][-1]["validation"] for r in baseline]
    base_model = CapacityMLP(13)
    base_result = {"hidden1": 13, "hidden2": 17,
                   "parameters": {**count_parameters(base_model), **base_model.structural_metrics()},
                   "loss": summary([r["loss"] for r in base_final]),
                   "accuracy": summary([r["accuracy"] for r in base_final]), "reused_s2_t05": True}
    base_result["match"] = matches(base_result, target)
    if base_result["match"]["full"]:
        raise ValueError("Existing baseline unexpectedly matches; predefined refinement has no lower bound")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"phase": "validation_search", "target": target,
                "thresholds": {"accuracy_min": target["accuracy"]["mean"] - ACCURACY_TOLERANCE,
                               "loss_max": target["loss"]["mean"] + LOSS_TOLERANCE},
                "ladder": LADDER, "seeds": SEEDS, "candidates": [base_result]}
    manifest_path = output_dir / "search_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    first_pass = None
    last_fail = 13
    for h in LADDER:
        result = candidate(base, h, output_dir, target)
        manifest["candidates"].append(result)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if first_pass is None:
            if result["match"]["full"]:
                first_pass = h
            else:
                last_fail = h
        else:
            break  # exactly one next ladder candidate as sanity check
    if first_pass is None:
        manifest["phase"] = "no_full_match_within_ladder"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("No full match; test stays closed", flush=True)
        return
    for h in range(last_fail + 1, first_pass):
        result = candidate(base, h, output_dir, target)
        manifest["candidates"].append(result)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    selected = min((r for r in manifest["candidates"] if r["match"]["full"]),
                   key=lambda r: r["parameters"]["total"])
    parameters = selected["parameters"]["total"]
    macs = selected["parameters"]["theoretical_main_macs_per_sample"]
    frozen = {"selected_hidden1": selected["hidden1"], "selected_hidden2": selected["hidden2"],
              "parameters": parameters, "parameter_ratio_vs_dsu": parameters / 10634,
              "main_macs": macs, "mac_ratio_vs_dsu": macs / 9984,
              "selection_basis": "validation_only_final_epoch_25", "test_evaluated": False}
    (output_dir / "frozen_selection.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    manifest["phase"] = "frozen_before_test"
    manifest["selection"] = frozen
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"FROZEN SELECTION: {frozen}", flush=True)


def test_frozen(config_path, output_dir):
    frozen = json.loads((output_dir / "frozen_selection.json").read_text(encoding="utf-8"))
    if frozen["selection_basis"] != "validation_only_final_epoch_25":
        raise ValueError("No valid frozen selection")
    base = load_config(config_path)
    results = []
    for seed in SEEDS:
        config = replace(base, seed=seed)
        _train, _validation, test_loader, dataset = load_fashion_mnist(config, include_test=True)
        model = CapacityMLP(frozen["selected_hidden1"])
        state = torch.load(output_dir / f"h{frozen['selected_hidden1']}_seed{seed}.pt",
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        result = {"seed": seed, "test": evaluate(model, test_loader, torch.device("cpu")),
                  "dataset": dataset}
        results.append(result)
        print(f"test selected seed={seed}: {result['test']}", flush=True)
    report = {"frozen_selection": frozen, "seeds": results,
              "loss": summary([r["test"]["loss"] for r in results]),
              "accuracy": summary([r["test"]["accuracy"] for r in results])}
    (output_dir / "selected_test.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("search", "test"))
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/capacity_mlp_fmnist_s2_t06.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mlp_capacity_s2_t06"))
    args = parser.parse_args()
    if args.phase == "search":
        search(args.config, args.output_dir, Path("runs/dsu_s_v0_s2_t04"),
               Path("runs/parameter_matched_mlp_s2_t05"))
    else:
        test_frozen(args.config, args.output_dir)


if __name__ == "__main__":
    main()
