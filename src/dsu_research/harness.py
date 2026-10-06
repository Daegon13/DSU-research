"""Common metrics and execution path for the Sprint 0 baseline."""

import platform
import random
import statistics
import time
from importlib.metadata import version
from typing import Callable

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig


def seed_everything(seed: int) -> dict[str, int | None]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return {"python": seed, "numpy": seed, "torch": seed, "cuda": seed if torch.cuda.is_available() else None}


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += len(targets)
    if count == 0:
        raise ValueError("Evaluation split is empty")
    return {"loss": loss_sum / count, "accuracy": correct / count}


def measure_latency(model: nn.Module, example: torch.Tensor, device: torch.device, warmup: int = 5, repeats: int = 20) -> dict:
    model.eval()
    example = example[:1].to(device)
    measurements = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(example)
        synchronize(device)
        for _ in range(repeats):
            start = time.perf_counter()
            model(example)
            synchronize(device)
            measurements.append((time.perf_counter() - start) * 1000)
    return {
        "batch_size": 1,
        "warmup_runs": warmup,
        "repeats": repeats,
        "mean_ms_per_batch": statistics.mean(measurements),
        "std_ms_per_batch": statistics.stdev(measurements) if repeats > 1 else 0.0,
        "samples_per_second": 1000 / statistics.mean(measurements),
    }


def run_experiment(
    config: ExperimentConfig,
    model_factory: Callable[[ExperimentConfig], nn.Module],
    dataset_loader: Callable,
) -> dict:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(config.device)
    seeds = seed_everything(config.seed)
    dataset = dataset_loader(config)
    if len(dataset) == 4:
        train_loader, val_loader, test_loader, dataset_metadata = dataset
    else:
        train_loader, test_loader, dataset_metadata = dataset
        val_loader = None
    model = model_factory(config).to(device)
    reproduction = val_loader is not None
    adam_epsilon = 1e-7 if reproduction else 1e-8
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.999), eps=adam_epsilon)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    train_start = time.perf_counter()
    total_seen = 0
    for epoch in range(config.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += len(targets)
        if count == 0:
            raise ValueError("Training split is empty")
        total_seen += count
        epoch_metrics = {"epoch": epoch + 1, "loss": loss_sum / count, "accuracy": correct / count}
        if val_loader is not None:
            epoch_metrics["validation"] = evaluate(model, val_loader, device)
        history.append(epoch_metrics)
    synchronize(device)
    train_seconds = time.perf_counter() - train_start
    test_metrics = evaluate(model, test_loader, device)
    try:
        latency_example = next(iter(test_loader))[0]
    except StopIteration as exc:
        raise ValueError("Evaluation split is empty") from exc
    inference = measure_latency(model, latency_example, device)
    memory = {
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "cpu_peak_ram_bytes": None,
        "note": "CPU peak RAM is not measured in Sprint 0; CUDA peak is allocator memory, not whole-process VRAM.",
    }
    result = {
        "config": config.to_dict(),
        "seeds": seeds,
        "dataset": dataset_metadata,
        "model": {"name": config.model, "architecture": str(model)},
        "parameters": count_parameters(model),
        "training": {
            "optimizer": "Adam", "adam_betas": [0.9, 0.999], "adam_epsilon": adam_epsilon,
            "weight_decay": 0.0, "scheduler": None,
            "gradient_clipping": None, "mixed_precision": False, "early_stopping": False,
            "history": history, "duration_seconds": train_seconds,
            "samples_seen": total_seen, "samples_per_second": total_seen / train_seconds,
        },
        "test": test_metrics,
        "inference": inference,
        "memory": memory,
        "environment": {
            "device": str(device), "cpu": platform.processor() or platform.uname().processor,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "os": platform.platform(), "backend": "CUDA" if device.type == "cuda" else "CPU",
            "precision": "float32", "python": platform.python_version(),
            "torch": torch.__version__, "torchvision": version("torchvision"),
            "numpy": np.__version__,
        },
    }
    if reproduction:
        stored = result["parameters"]["total"]
        trainable = result["parameters"]["trainable"]
        result["parameters"].update({
            "stored_parameters": stored,
            "trainable_stored_parameters": trainable,
            "effective_parameters": model.effective_parameters() if hasattr(model, "effective_parameters") else trainable,
        })
        result["mask_statistics"] = model.connectivity_metrics() if hasattr(model, "connectivity_metrics") else None
        if hasattr(model, "structural_metrics"):
            result["parameters"].update(model.structural_metrics())
        result["training"]["checkpoint_selected"] = f"final_epoch_{config.epochs}"
        result["training"]["validation_best_epoch_diagnostic"] = min(
            history, key=lambda item: item["validation"]["loss"])["epoch"]
    return result
