"""CLI that runs one configured experiment and persists its metrics."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .data import load_mnist
from .harness import run_experiment
from .model import MLPReLU


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Sprint 0 baseline experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="JSON output path; default is a timestamped file in runs/")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.model != "mlp_relu" or config.dataset != "mnist":
        parser.error("The CLI currently supports only model='mlp_relu' and dataset='mnist'")
    output = args.output or Path("runs") / f"{config.model}_{config.dataset}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    result = run_experiment(config, model_factory=lambda cfg: MLPReLU(hidden_dim=cfg.hidden_dim), dataset_loader=load_mnist)
    result["run_utc"] = datetime.now(timezone.utc).isoformat()
    result["config_source"] = str(args.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    print(f"Test loss: {result['test']['loss']:.4f}; accuracy: {result['test']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
