"""CLI that runs one configured experiment and persists its metrics."""

import argparse
import json
import statistics
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .data import load_fashion_mnist, load_mnist
from .dsu_s import DSUSv0
from .harness import run_experiment
from .model import DendriticANNRandom, MLPReLU, ParameterMatchedMLP, VanillaANN


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured classification experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="JSON output path; default is a timestamped file in runs/")
    parser.add_argument("--seeds", type=int, nargs="+", help="Run these trial seeds for each reproduction model")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seeds:
        if config.dataset != "fashion_mnist" or config.model != "reproduction_pair":
            parser.error("--seeds requires model='reproduction_pair' and dataset='fashion_mnist'")
        if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
            parser.error("--seeds requires distinct nonnegative integers")
        output_dir = args.output or Path("runs/reproduction_001")
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for seed in args.seeds:
            for model_name, factory in (("vann", lambda cfg: VanillaANN()),
                                        ("dann_r", lambda cfg: DendriticANNRandom(cfg.seed))):
                trial = replace(config, seed=seed, model=model_name)
                result = run_experiment(trial, model_factory=factory, dataset_loader=load_fashion_mnist)
                result["run_utc"] = datetime.now(timezone.utc).isoformat()
                result["config_source"] = str(args.config)
                path = output_dir / f"{model_name}_seed{seed}.json"
                path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                results.append(result)
                print(f"Saved {path}: loss={result['test']['loss']:.6f}, accuracy={result['test']['accuracy']:.4%}", flush=True)
        summary = {"seeds": args.seeds, "models": {}}
        for model_name in ("vann", "dann_r"):
            group = [r for r in results if r["config"]["model"] == model_name]
            summary["models"][model_name] = {}
            for metric in ("loss", "accuracy"):
                values = [r["test"][metric] for r in group]
                summary["models"][model_name][metric] = {
                    "values": values, "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved {output_dir / 'summary.json'}")
        return
    if config.model == "mlp_relu" and config.dataset == "mnist":
        factory, dataset_loader = lambda cfg: MLPReLU(hidden_dim=cfg.hidden_dim), load_mnist
    elif config.model == "dsu_s_v0" and config.dataset == "fashion_mnist":
        if config.topology_seed is None:
            parser.error("DSU-S requires an explicit topology_seed")
        factory, dataset_loader = lambda cfg: DSUSv0(cfg.topology_seed), load_fashion_mnist
    elif config.model == "parameter_matched_mlp" and config.dataset == "fashion_mnist":
        if config.topology_seed is not None:
            parser.error("The dense MLP must not have a topology_seed")
        factory, dataset_loader = lambda cfg: ParameterMatchedMLP(), load_fashion_mnist
    else:
        parser.error("Single-run CLI supports mlp_relu/MNIST, dsu_s_v0/Fashion-MNIST or parameter_matched_mlp/Fashion-MNIST; use --seeds for the reproduction pair")
    output = args.output or Path("runs") / f"{config.model}_{config.dataset}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    result = run_experiment(config, model_factory=factory, dataset_loader=dataset_loader)
    result["run_utc"] = datetime.now(timezone.utc).isoformat()
    result["config_source"] = str(args.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    print(f"Test loss: {result['test']['loss']:.4f}; accuracy: {result['test']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
