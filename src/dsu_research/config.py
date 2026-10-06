"""Strict, small JSON configuration for the initial classification experiment."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    dataset: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    device: str
    hidden_dim: int = 128
    train_samples: int | None = None
    test_samples: int | None = None
    topology_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError("dataset must be a non-empty string")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        for name in ("seed", "epochs", "batch_size", "hidden_dim"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a valid integer")
        if self.seed > 2**32 - 1:
            raise ValueError("seed must fit NumPy's 32-bit seed range")
        if self.topology_seed is not None and (type(self.topology_seed) is not int or not 0 <= self.topology_seed <= 2**32 - 1):
            raise ValueError("topology_seed must be a nonnegative 32-bit integer or null")
        for name in ("train_samples", "test_samples"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or null")
        if type(self.learning_rate) not in (float, int) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = data.keys() - {field.name for field in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return ExperimentConfig(**data)
