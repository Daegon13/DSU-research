"""Official MNIST train/test split with deterministic prefix subsets for smoke runs."""

from pathlib import Path

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
