"""Structural and mathematical checks for compact DSU-S v0."""

import io
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dsu_research.config import ExperimentConfig, load_config
from dsu_research.dsu_s import DSUSv0
from dsu_research.harness import run_experiment, seed_everything


def test_shapes_storage_and_topology():
    model = DSUSv0(17)
    assert model(torch.randn(7, 784)).shape == (7, 10)
    assert model.dendrite_pre_activations(torch.randn(7, 784)).shape == (7, 128, 4)
    assert model.G.shape == (128, 4, 16)
    assert set(dict(model.named_parameters())) == {"W_d", "bias_d", "W_s", "bias_s", "W_o", "bias_o"}
    assert {name: tuple(p.shape) for name, p in model.named_parameters()} == {
        "W_d": (128, 4, 16), "bias_d": (128, 4), "W_s": (128, 4),
        "bias_s": (128,), "W_o": (10, 128), "bias_o": (10,),
    }
    stored = sum(p.numel() for p in model.parameters())
    assert stored == model.effective_parameters() == 10634
    assert all(p.requires_grad for p in model.parameters())
    assert not model.G.requires_grad
    assert all(t.numel() < 512 * 784 for t in list(model.parameters()) + list(model.buffers()))
    assert max(t.numel() for t in model.parameters()) == 8192
    assert model.G.min() >= 0 and model.G.max() < 784
    assert torch.all(torch.diff(model.G.sort(dim=-1).values, dim=-1) > 0)
    metrics = model.structural_metrics()
    assert metrics["trainable_parameter_bytes"] == 42536
    assert metrics["topology_index_count"] == 8192
    assert metrics["topology_index_dtype"] == str(model.G.dtype)
    assert metrics["topology_index_bytes"] == model.G.numel() * model.G.element_size()
    assert metrics["theoretical_main_macs_per_sample"] == 8192 + 512 + 1280 == 9984


def test_topology_seed_is_independent_of_training_seed():
    seed_everything(1)
    first = DSUSv0(19)
    seed_everything(2)
    second = DSUSv0(19)
    third = DSUSv0(20)
    assert torch.equal(first.G, second.G)
    assert not torch.equal(first.G, third.G)
    assert first.structural_metrics()["topology_sha256"] == second.structural_metrics()["topology_sha256"]
    assert not torch.equal(first.W_d, second.W_d)


def test_forward_math_and_semantic_connectivity():
    model = DSUSv0(23)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.W_d[0, 0, 0] = 2
        model.bias_d[0, 0] = -1
        model.W_s[0, 0] = 3
        model.bias_s[0] = 1
        model.W_o[0, 0] = 4
        model.bias_o[0] = 5
    observed = int(model.G[0, 0, 0])
    unseen = next(i for i in range(784) if i not in model.G[0].flatten().tolist())
    x = torch.zeros(1, 784)
    x[0, observed] = 2
    pre = model.dendrite_pre_activations(x)
    assert pre[0, 0, 0].item() == pytest.approx(3)
    assert model(x)[0, 0].item() == pytest.approx(45)
    assert model(torch.zeros_like(x))[0, 0].item() == pytest.approx(7.8)
    changed_unseen = x.clone()
    changed_unseen[0, unseen] = 99
    torch.testing.assert_close(model.dendrite_pre_activations(changed_unseen)[0, 0], pre[0, 0])
    changed_observed = x.clone()
    changed_observed[0, observed] += 1
    assert model.dendrite_pre_activations(changed_observed)[0, 0, 0] > pre[0, 0, 0]


def test_backward_and_serialization():
    model = DSUSv0(29)
    original_topology = model.G.clone()
    x = torch.randn(5, 784)
    torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2, 3, 4])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.G.grad is None
    torch.optim.Adam(model.parameters(), lr=0.001).step()
    assert torch.equal(model.G, original_topology)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = DSUSv0(30)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert torch.equal(restored.G, model.G)
    for name, param in model.named_parameters():
        torch.testing.assert_close(dict(restored.named_parameters())[name], param)
    torch.testing.assert_close(restored(x), model(x))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_device_forward_backward(device):
    model = DSUSv0(31).to(device)
    assert model.G.device.type == device
    model(torch.randn(2, 784, device=device)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_common_harness_synthetic_batch():
    def tiny_data(config):
        x = torch.rand(8, 1, 28, 28)
        y = torch.arange(8)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        return loader, loader, loader, {"name": "synthetic", "splits": {"train": 8, "validation": 8, "test": 8}}

    config = ExperimentConfig("dsu_s_v0", "fashion_mnist", 3, 1, 4, 0.001, "cpu", topology_seed=41)
    result = run_experiment(config, lambda cfg: DSUSv0(cfg.topology_seed), tiny_data)
    counts = result["parameters"]
    assert counts["stored_parameters"] == counts["trainable_stored_parameters"] == counts["effective_parameters"] == 10634
    assert counts["trainable_parameter_bytes"] == 42536
    assert counts["topology_index_count"] == 8192
    assert counts["theoretical_main_macs_per_sample"] == 9984
    assert result["config"]["topology_seed"] == 41
    assert result["seeds"]["torch"] == 3
    assert result["training"]["samples_seen"] == 8
    assert result["training"]["checkpoint_selected"] == "final_epoch_1"


def test_config_records_and_validates_topology_seed(tmp_path):
    path = tmp_path / "dsu.json"
    values = {"model": "dsu_s_v0", "dataset": "fashion_mnist", "seed": 3,
              "topology_seed": 41, "epochs": 1, "batch_size": 4,
              "learning_rate": 0.001, "device": "cpu"}
    path.write_text(json.dumps(values), encoding="utf-8")
    assert load_config(path).topology_seed == 41
    values["topology_seed"] = -1
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="topology_seed"):
        load_config(path)
