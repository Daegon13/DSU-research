import torch
from torch.utils.data import DataLoader, TensorDataset

from dsu_research.config import ExperimentConfig
from dsu_research.harness import run_experiment
from dsu_research.model import DendriticANNRandom, VanillaANN


def test_vann_logits_backward_and_parameter_count():
    model = VanillaANN()
    assert (model.dendrites.in_features, model.dendrites.out_features) == (784, 512)
    assert (model.somata.in_features, model.somata.out_features) == (512, 128)
    assert (model.output.in_features, model.output.out_features) == (128, 10)
    logits = model(torch.randn(3, 1, 28, 28))
    assert logits.shape == (3, 10)
    logits.sum().backward()
    assert all(param.grad is not None for param in model.parameters())
    assert sum(param.numel() for param in model.parameters()) == 468874


def test_dann_masks_and_counts():
    model = DendriticANNRandom(1)
    same = DendriticANNRandom(1)
    different = DendriticANNRandom(2)
    assert torch.equal(model.input_mask, same.input_mask)
    assert not torch.equal(model.input_mask, different.input_mask)
    assert not model.input_mask.requires_grad
    assert not model.cable_mask.requires_grad
    assert model.input_mask.sum().item() == 8192
    assert model.cable_mask.sum().item() == 512
    assert torch.all(model.cable_mask.sum(dim=1) == 4)
    for soma in range(128):
        assert model.cable_mask[soma].nonzero().flatten().tolist() == list(range(4 * soma, 4 * soma + 4))
    counts = model.input_mask.sum(dim=1)
    assert counts.float().mean().item() == 16
    assert counts.min() < 16 < counts.max()
    assert sum(param.numel() for param in model.parameters()) == 468874
    assert model.effective_parameters() == 10634
    assert model.connectivity_metrics()["mask_active_connections"] == 8704


def test_dann_inactive_connections_never_affect_forward_or_gradient():
    model = DendriticANNRandom(3)
    original_input_mask = model.input_mask.clone()
    original_cable_mask = model.cable_mask.clone()
    assert torch.all(model.dendrites.weight[~model.input_mask] == 0)
    assert torch.all(model.somata.weight[~model.cable_mask] == 0)
    inputs = torch.randn(2, 784)
    before = model(inputs).detach().clone()
    with torch.no_grad():
        model.dendrites.weight[~model.input_mask] = 999
        model.somata.weight[~model.cable_mask] = 999
    torch.testing.assert_close(model(inputs), before)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1]))
    loss.backward()
    assert torch.all(model.dendrites.weight.grad[~model.input_mask] == 0)
    assert torch.all(model.somata.weight.grad[~model.cable_mask] == 0)
    optimizer.step()
    assert model(inputs).shape == (2, 10)
    assert torch.equal(model.input_mask, original_input_mask)
    assert torch.equal(model.cable_mask, original_cable_mask)


def test_common_harness_records_validation_and_effective_parameters():
    def tiny_data(config):
        inputs = torch.rand(8, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)
        return loader, loader, loader, {"name": "tiny", "splits": {"train": 8, "validation": 8, "test": 8}}

    for name, factory in (("vann", lambda cfg: VanillaANN()),
                          ("dann_r", lambda cfg: DendriticANNRandom(cfg.seed))):
        config = ExperimentConfig(name, "fashion_mnist", 1, 1, 4, 0.001, "cpu")
        result = run_experiment(config, factory, tiny_data)
        assert result["parameters"]["stored_parameters"] == 468874
        assert result["parameters"]["effective_parameters"] == (10634 if name == "dann_r" else 468874)
        assert "validation" in result["training"]["history"][0]
        assert result["training"]["checkpoint_selected"] == "final_epoch_1"
        assert (result["mask_statistics"] is None) == (name == "vann")
