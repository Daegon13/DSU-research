import json
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dsu_research.config import ExperimentConfig, load_config
from dsu_research.harness import count_parameters, evaluate, measure_latency, run_experiment
from dsu_research.model import MLPReLU


def test_model_forward_backward_and_parameters():
    model = MLPReLU(input_dim=4, hidden_dim=3, num_classes=2)
    inputs = torch.randn(5, 1, 2, 2)
    outputs = model(inputs)
    assert outputs.shape == (5, 2)
    outputs.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    assert count_parameters(model) == {"total": 4 * 3 + 3 + 3 * 2 + 2, "trainable": 23}


def test_config_load_and_reject_unknown_fields(tmp_path):
    data = {"model": "mlp_relu", "dataset": "mnist", "seed": 7, "epochs": 1,
            "batch_size": 4, "learning_rate": 0.001, "device": "cpu"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_config(path).seed == 7
    data["unintended_change"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_minimal_harness_without_network():
    def tiny_dataset(config):
        generator = torch.Generator().manual_seed(config.seed)
        inputs = torch.rand(8, 1, 28, 28, generator=generator)
        targets = torch.randint(0, 10, (8,), generator=generator)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=config.batch_size)
        return loader, loader, {"name": "test_fixture", "splits": {"train": 8, "validation": 0, "test": 8}}

    config = ExperimentConfig("mlp_relu", "mnist", 7, 1, 4, 0.001, "cpu")
    result = run_experiment(config, model_factory=lambda cfg: MLPReLU(hidden_dim=cfg.hidden_dim), dataset_loader=tiny_dataset)
    assert result["parameters"] == {"total": 101770, "trainable": 101770}
    assert result["training"]["samples_seen"] == 8
    assert result["training"]["duration_seconds"] > 0
    assert 0 <= result["test"]["accuracy"] <= 1
    assert result["inference"]["batch_size"] == 1
    json.dumps(result, allow_nan=False)


def test_harness_accepts_another_model_and_input_shape():
    class SmallClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, inputs):
            return self.linear(inputs)

    def tiny_dataset(config):
        inputs = torch.randn(8, 4)
        targets = torch.tensor([0, 1] * 4)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=config.batch_size)
        return loader, loader, {"name": "test_fixture", "splits": {"train": 8, "validation": 0, "test": 8}}

    config = ExperimentConfig("small_classifier", "test_fixture", 7, 1, 4, 0.001, "cpu")
    result = run_experiment(config, model_factory=lambda _: SmallClassifier(), dataset_loader=tiny_dataset)
    assert result["model"]["name"] == "small_classifier"
    assert "Linear(in_features=4, out_features=2" in result["model"]["architecture"]
    assert result["parameters"] == {"total": 10, "trainable": 10}
    assert result["inference"]["warmup_runs"] == 5
    assert result["inference"]["repeats"] == 20


def test_latency_performs_warmup_and_repeated_batch_one_forward():
    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, inputs):
            self.batch_sizes.append(len(inputs))
            return inputs

    model = CountingModel()
    result = measure_latency(model, torch.ones(4, 3), torch.device("cpu"), warmup=3, repeats=7)
    assert model.batch_sizes == [1] * 10
    assert result["warmup_runs"] == 3
    assert result["repeats"] == 7
    assert result["mean_ms_per_batch"] > 0


def test_evaluation_uses_all_examples_for_loss_and_accuracy():
    class FixedPredictions(torch.nn.Module):
        def forward(self, inputs):
            return torch.tensor([[2.0, 0.0]]).expand(len(inputs), -1)

    loader = DataLoader(TensorDataset(torch.zeros(2, 4), torch.tensor([0, 1])), batch_size=1)
    result = evaluate(FixedPredictions(), loader, torch.device("cpu"))
    assert result["accuracy"] == 0.5
    assert result["loss"] == pytest.approx((math.log1p(math.exp(-2)) + math.log1p(math.exp(2))) / 2)
