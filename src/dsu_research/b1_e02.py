"""Preregistered B1-E02 interior interference; B1-E01 model is unchanged."""

from __future__ import annotations

import numpy as np
import torch

from .b1_e01 import D, DATASET_SEED, DELAYS, SPLITS, make_split, sequence_batch, sha_bytes

NOISE_CHANNELS = (0, 1, *range(3, D))


def make_noise_split(split: str, delay: int, dataset_seed: int = DATASET_SEED):
    """Generate class-independent interior noise by sample key, in time order."""
    first, labels = make_split(split, delay, dataset_seed)
    split_id, size = SPLITS[split]
    noise = np.empty((size, delay - 1, len(NOISE_CHANNELS)), dtype=np.float32)
    for index in range(size):
        rng = np.random.default_rng(np.random.SeedSequence(
            [dataset_seed, split_id, delay, index, 99]))
        noise[index] = rng.uniform(-0.1, 0.1, size=(delay - 1, 31)).astype(np.float32)
    return first, labels, torch.from_numpy(noise)


def noise_sequence_batch(first: torch.Tensor, noise: torch.Tensor, delay: int,
                         oracle: bool = False) -> torch.Tensor:
    if (first.ndim != 2 or first.shape[1] != D or delay not in DELAYS or
            noise.shape != (len(first), delay - 1, len(NOISE_CHANNELS))):
        raise ValueError("invalid NOISE batch")
    sequence = sequence_batch(first, delay, oracle=oracle)
    sequence[:, 1:-1, NOISE_CHANNELS] = noise
    return sequence


def noise_split_fingerprint(first: torch.Tensor, labels: torch.Tensor, noise: torch.Tensor,
                            split: str, delay: int, dataset_seed: int = DATASET_SEED) -> str:
    payload = f"delayed_recall_noise_v1|{dataset_seed}|{split}|{delay}|".encode("ascii")
    return sha_bytes(payload + first.numpy().tobytes() + labels.numpy().tobytes() + noise.numpy().tobytes())
