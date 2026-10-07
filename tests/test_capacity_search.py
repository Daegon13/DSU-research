"""Small structural and protocol checks for S2-T06."""

import json
from pathlib import Path

import pytest
import torch

from dsu_research.capacity_search import LADDER, SEEDS, matches
from dsu_research.config import load_config
from dsu_research.data import load_fashion_mnist
from dsu_research.model import CapacityMLP


@pytest.mark.parametrize("hidden1,hidden2,parameters", [
    (13, 17, 10623), (16, 21, 13137), (19, 25, 15675),
    (22, 29, 18237), (25, 33, 20823), (28, 37, 23433), (32, 42, 26936),
])
def test_family_dimensions_and_parameter_counts(hidden1, hidden2, parameters):
    model = CapacityMLP(hidden1)
    assert [(layer.in_features, layer.out_features) for layer in
            (model.hidden1, model.hidden2, model.output)] == [
                (784, hidden1), (hidden1, hidden2), (hidden2, 10)]
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert model.structural_metrics()["theoretical_main_macs_per_sample"] == (
        784 * hidden1 + hidden1 * hidden2 + hidden2 * 10)


def test_logits_have_no_softmax():
    model = CapacityMLP(16)
    with torch.no_grad():
        for layer in (model.hidden1, model.hidden2, model.output):
            layer.weight.zero_()
            layer.bias.fill_(2)
    torch.testing.assert_close(model(torch.zeros(2, 784)), torch.full((2, 10), 2.0))


def test_fixed_search_config_and_thresholds():
    config = json.loads(Path("experiments/configs/capacity_mlp_fmnist_s2_t06.json").read_text())
    assert (config["epochs"], config["batch_size"], config["learning_rate"], config["device"]) == (
        25, 128, 0.001, "cpu")
    assert LADDER == (16, 19, 22, 25, 28, 32)
    assert SEEDS == (1, 2, 3)
    target = {"accuracy": {"mean": 0.885}, "loss": {"mean": 0.320}}
    assert matches({"accuracy": {"mean": 0.880}, "loss": {"mean": 0.340}}, target)["full"]
    assert not matches({"accuracy": {"mean": 0.879}, "loss": {"mean": 0.339}}, target)["full"]


def test_search_data_path_does_not_open_test_dataset(monkeypatch):
    calls = []

    class FakeFashionMNIST:
        def __init__(self, *, train, **_kwargs):
            calls.append(train)
            if not train:
                raise AssertionError("test dataset opened during search")

        def __len__(self):
            return 60000

    monkeypatch.setattr("dsu_research.data.torchvision.datasets.FashionMNIST", FakeFashionMNIST)
    config = load_config("experiments/configs/capacity_mlp_fmnist_s2_t06.json")
    train, validation, test, metadata = load_fashion_mnist(config, include_test=False)
    assert calls == [True]
    assert (len(train.dataset), len(validation.dataset), test) == (54000, 6000, None)
    assert metadata["splits"] == {"train": 54000, "validation": 6000, "test": 10000}
