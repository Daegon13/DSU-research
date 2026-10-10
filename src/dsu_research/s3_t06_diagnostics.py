"""Forward-only, label-free S3-T06 initialization diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import FashionMNIST

from .sparse_compartment_controls import (
    NeutralCompartments, PointSparse, base_topology, diagnostics, paired_values,
)


def train_calibration_batch(seed, size=256, data_dir="data"):
    source = FashionMNIST(root=data_dir, train=True, download=False)
    indices = np.arange(len(source))
    np.random.default_rng(seed).shuffle(indices)
    selected = indices[:size].astype("<i8")
    inputs = source.data[selected].to(torch.float32).flatten(1) / 255.0
    return inputs, hashlib.sha256(selected.tobytes()).hexdigest()


def run(seeds=(1, 2, 3)):
    records = []
    for seed in seeds:
        x, batch_hash = train_calibration_batch(seed)
        g64 = base_topology(seed)
        values = paired_values(seed)
        point = PointSparse(g64)
        point_init = point.initialize(*values)
        records.append({"seed": seed, "arm": "PointSparse", "counts": point.counts(),
                        "initialization": point_init, "train_batch_indices_sha256": batch_hash,
                        "diagnostics": diagnostics(point, x)})
        for b in (1, 2, 4, 8):
            model = NeutralCompartments(b, g64)
            init = model.initialize(*values)
            calibration = model.calibrate(x, point)
            records.append({"seed": seed, "arm": f"{b}x{64 // b}", "counts": model.counts(),
                            "initialization": init, "calibration": calibration,
                            "train_batch_indices_sha256": batch_hash,
                            "diagnostics": diagnostics(model, x)})
    return {"task": "S3-T06", "phase": "pretraining_forward_only",
            "calibration": "first 256 shuffled train indices; no labels; ToTensor scale /255",
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
