"""Read-only B1-E02 pairing audit against the historical B1-E01 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .b1_e01 import DELAYS, SEEDS

ROOT = Path(__file__).resolve().parents[2]
TRAINING_FIELDS = (
    "optimizer", "learning_rate", "betas", "epsilon", "weight_decay",
    "batch_size", "epochs", "checkpoint", "loss", "early_stopping",
    "scheduler", "gradient_clipping", "mixed_precision", "batch_order_sha256",
)


def audit() -> dict:
    checked = 0
    for delay in DELAYS:
        for seed in SEEDS:
            for arm in ("S0", "S1"):
                name = f"{arm}_d{delay}_seed{seed}.json"
                easy = json.loads((ROOT / "runs" / "b1_e01" / name).read_text(encoding="utf-8"))
                noise = json.loads((ROOT / "runs" / "b1_e02" / name).read_text(encoding="utf-8"))
                for key in ("topology", "initialization_sha256", "seeds", "cost", "lambda"):
                    if easy[key] != noise[key]:
                        raise ValueError(f"historical pairing mismatch {name}: {key}")
                for key in TRAINING_FIELDS:
                    if easy["training"][key] != noise["training"][key]:
                        raise ValueError(f"historical training mismatch {name}: {key}")
                if easy["dataset"] == noise["dataset"]:
                    raise ValueError(f"NOISE did not change dataset fingerprints: {name}")
                checked += 1
    return {"historical_pairs_checked": checked, "matching_protocol_pairs": checked,
            "different_dataset_fingerprint_pairs": checked,
            "easy_result": "data/b1_e01_results.json",
            "noise_result": "runs/b1_e02/summary.json"}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
