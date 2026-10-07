"""Official MNIST train/test split with deterministic prefix subsets for smoke runs."""

from pathlib import Path
import hashlib

import numpy as np
import torch

import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .config import ExperimentConfig


def load_mnist(config: ExperimentConfig, data_dir: str | Path = "data"):
    transform = transforms.ToTensor()
    train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    if config.train_samples is not None:
        if config.train_samples > len(train):
            raise ValueError("train_samples exceeds MNIST train split")
        train = Subset(train, range(config.train_samples))
    if config.test_samples is not None:
        if config.test_samples > len(test):
            raise ValueError("test_samples exceeds MNIST test split")
        test = Subset(test, range(config.test_samples))
    # DataLoader's private generator keeps the shuffle order tied to the recorded seed.
    import torch

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False)
    metadata = {
        "name": "MNIST",
        "source": "torchvision.datasets.MNIST",
        "source_version": torchvision.__version__,
        "splits": {"train": len(train), "validation": 0, "test": len(test)},
        "subset_rule": "first N examples of each official split; no subset if null",
        "transform": "ToTensor() scales uint8 pixels to float32 [0,1]",
    }
    return train_loader, test_loader, metadata


def load_fashion_mnist(config: ExperimentConfig, data_dir: str | Path = "data", *, include_test: bool = True):
    """Official train/test with the original trial-specific 54k/6k split."""
    if config.train_samples is not None or config.test_samples is not None:
        raise ValueError("The reproduction requires full Fashion-MNIST splits")
    transform = transforms.ToTensor()
    source = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test = (torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
            if include_test else None)
    indices = np.arange(len(source))
    np.random.default_rng(config.seed).shuffle(indices)
    train_indices, val_indices = indices[:-6000], indices[-6000:]
    train = Subset(source, train_indices.tolist())
    validation = Subset(source, val_indices.tolist())
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False) if test is not None else None
    metadata = {
        "name": "Fashion-MNIST", "source": "torchvision.datasets.FashionMNIST",
        "source_version": torchvision.__version__,
        "splits": {"train": len(train), "validation": len(validation), "test": len(test) if test is not None else 10000},
        "split_rule": "np.random.default_rng(seed).shuffle(arange(60000)); last 6000 validation",
        "train_indices_sha256": hashlib.sha256(train_indices.astype("<i8").tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(val_indices.astype("<i8").tobytes()).hexdigest(),
        "transform": "ToTensor() scales uint8 pixels to float32 by 255; no other normalization",
    }
    return train_loader, val_loader, test_loader, metadata
