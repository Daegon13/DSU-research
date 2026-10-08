"""Paired structural, functional and optimizer checks for Fixed-dANN and DSU-S."""

import io

import pytest
import torch
from torch.nn import functional as F

from dsu_research.dsu_s import DSUSv0
from dsu_research.fixed_dann import FixedDANN, fixed_dann_from_dsu_s
from dsu_research.harness import seed_everything


RTOL = 1e-5
ATOL = 1e-6


def _active_fixed_weights(fixed):
    input_rows = torch.arange(512, device=fixed.G.device).repeat_interleave(16)
    soma_rows = torch.arange(128, device=fixed.G.device).repeat_interleave(4)
    return (
        fixed.dendrites.weight[input_rows, fixed.G.reshape(-1)].reshape(128, 4, 16),
        fixed.somata.weight[soma_rows, torch.arange(512, device=fixed.G.device)].reshape(128, 4),
    )


def _paired_parameter_deltas(dsu, fixed):
    input_weights, soma_weights = _active_fixed_weights(fixed)
    pairs = (
        (dsu.W_d, input_weights), (dsu.bias_d, fixed.dendrites.bias.reshape(128, 4)),
        (dsu.W_s, soma_weights), (dsu.bias_s, fixed.somata.bias),
        (dsu.W_o, fixed.output.weight), (dsu.bias_o, fixed.output.bias),
    )
    return [float((left.detach() - right.detach()).abs().max()) for left, right in pairs]


def _intermediates_dsu(dsu, x):
    dendrite_pre = dsu.dendrite_pre_activations(x)
    dendrite_activation = F.leaky_relu(dendrite_pre, negative_slope=0.1)
    soma_pre = (dendrite_activation * dsu.W_s).sum(dim=-1) + dsu.bias_s
    soma_activation = F.leaky_relu(soma_pre, negative_slope=0.1)
    logits = F.linear(soma_activation, dsu.W_o, dsu.bias_o)
    return dendrite_pre, dendrite_activation, soma_pre, soma_activation, logits


def _intermediates_fixed(fixed, x):
    dendrite_pre = fixed.dendrite_pre_activations(x)
    dendrite_activation = fixed.dendrite_activations(x)
    soma_pre = fixed.soma_pre_activations(x)
    soma_activation = fixed.soma_activations(x)
    logits = fixed(x)
    return dendrite_pre, dendrite_activation, soma_pre, soma_activation, logits


def test_topology_masks_parameter_accounting_and_serialization():
    dsu = DSUSv0(101)
    fixed = fixed_dann_from_dsu_s(dsu)
    assert torch.equal(fixed.G, dsu.G)
    expected_input_mask = torch.zeros_like(fixed.input_mask)
    expected_input_mask[torch.arange(512).repeat_interleave(16), dsu.G.reshape(-1)] = True
    expected_soma_mask = torch.zeros_like(fixed.soma_mask)
    expected_soma_mask[torch.arange(128).repeat_interleave(4), torch.arange(512)] = True
    assert torch.equal(fixed.input_mask, expected_input_mask)
    assert torch.equal(fixed.soma_mask, expected_soma_mask)
    assert fixed.input_mask.sum().item() == 8192
    assert fixed.soma_mask.sum().item() == 512
    assert torch.all(fixed.input_mask.sum(dim=1) == 16)
    assert torch.all(fixed.soma_mask.sum(dim=1) == 4)
    assert not any(buffer.requires_grad for buffer in fixed.buffers())
    assert fixed.structural_metrics() == {
        "stored_parameters": 468874,
        "effective_parameters": 10634,
        "input_mask_connections": 8192,
        "soma_mask_connections": 512,
    }
    assert sum(p.numel() for p in dsu.parameters()) == dsu.effective_parameters() == 10634
    assert torch.all(fixed.dendrites.weight[~fixed.input_mask] == 0)
    assert torch.all(fixed.somata.weight[~fixed.soma_mask] == 0)

    stream = io.BytesIO()
    torch.save(fixed.state_dict(), stream)
    stream.seek(0)
    restored = FixedDANN(DSUSv0(102).G)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert torch.equal(restored.G, fixed.G)
    assert torch.equal(restored.input_mask, fixed.input_mask)
    assert torch.equal(restored.soma_mask, fixed.soma_mask)


def test_rejects_invalid_topologies():
    valid = DSUSv0(103).G
    with pytest.raises(ValueError, match="shape"):
        FixedDANN(valid[:, :, :-1])
    duplicate = valid.clone()
    duplicate[0, 0, 1] = duplicate[0, 0, 0]
    with pytest.raises(ValueError, match="distinct"):
        FixedDANN(duplicate)
    out_of_range = valid.clone()
    out_of_range[0, 0, 0] = 784
    with pytest.raises(ValueError, match="indices"):
        FixedDANN(out_of_range)


def test_exact_weight_mapping_and_masks_are_semantically_required():
    seed_everything(104)
    dsu = DSUSv0(105)
    fixed = fixed_dann_from_dsu_s(dsu)
    assert _paired_parameter_deltas(dsu, fixed) == [0.0] * 6
    x = torch.randn(3, 784)
    before = fixed(x).detach().clone()
    with torch.no_grad():
        fixed.dendrites.weight[~fixed.input_mask] = 12345
        fixed.somata.weight[~fixed.soma_mask] = -12345
    torch.testing.assert_close(fixed(x), before, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 7])
def test_paired_intermediates_logits_and_loss(batch_size):
    seed_everything(106 + batch_size)
    dsu = DSUSv0(107)
    fixed = fixed_dann_from_dsu_s(dsu)
    x = torch.randn(batch_size, 784)
    targets = torch.arange(batch_size) % 10
    observed_errors = []
    for compact, dense in zip(_intermediates_dsu(dsu, x), _intermediates_fixed(fixed, x)):
        error = (compact - dense).abs()
        observed_errors.append({"max": float(error.detach().max()),
                                "mean": float(error.detach().mean())})
        torch.testing.assert_close(compact, dense, rtol=RTOL, atol=ATOL)
    compact_loss = F.cross_entropy(dsu(x), targets)
    dense_loss = F.cross_entropy(fixed(x), targets)
    torch.testing.assert_close(compact_loss, dense_loss, rtol=RTOL, atol=ATOL)
    assert all(error["max"] < 1e-5 and error["mean"] < 1e-6 for error in observed_errors)


def test_paired_gradients_and_inactive_gradients():
    seed_everything(109)
    dsu = DSUSv0(110)
    fixed = fixed_dann_from_dsu_s(dsu)
    x_dsu = torch.randn(8, 784, requires_grad=True)
    x_fixed = x_dsu.detach().clone().requires_grad_()
    targets = torch.arange(8)
    F.cross_entropy(dsu(x_dsu), targets).backward()
    F.cross_entropy(fixed(x_fixed), targets).backward()

    input_rows = torch.arange(512).repeat_interleave(16)
    soma_rows = torch.arange(128).repeat_interleave(4)
    paired_gradients = (
        (dsu.W_d.grad, fixed.dendrites.weight.grad[input_rows, fixed.G.reshape(-1)].reshape(128, 4, 16)),
        (dsu.bias_d.grad, fixed.dendrites.bias.grad.reshape(128, 4)),
        (dsu.W_s.grad, fixed.somata.weight.grad[soma_rows, torch.arange(512)].reshape(128, 4)),
        (dsu.bias_s.grad, fixed.somata.bias.grad),
        (dsu.W_o.grad, fixed.output.weight.grad),
        (dsu.bias_o.grad, fixed.output.bias.grad),
    )
    max_gradient_error = 0.0
    for compact, dense in paired_gradients:
        max_gradient_error = max(max_gradient_error, float((compact - dense).abs().max()))
        torch.testing.assert_close(compact, dense, rtol=RTOL, atol=ATOL)
    assert max_gradient_error < 1e-5
    torch.testing.assert_close(x_dsu.grad, x_fixed.grad, rtol=RTOL, atol=ATOL)
    assert torch.all(fixed.dendrites.weight.grad[~fixed.input_mask] == 0)
    assert torch.all(fixed.somata.weight.grad[~fixed.soma_mask] == 0)


def test_single_adam_step_and_five_step_trajectory():
    seed_everything(111)
    dsu = DSUSv0(112)
    fixed = fixed_dann_from_dsu_s(dsu)
    compact_optimizer = torch.optim.Adam(dsu.parameters(), lr=0.001, weight_decay=0)
    dense_optimizer = torch.optim.Adam(fixed.parameters(), lr=0.001, weight_decay=0)
    generator = torch.Generator().manual_seed(113)
    batches = [(torch.randn(6, 784, generator=generator),
                torch.randint(0, 10, (6,), generator=generator)) for _ in range(5)]

    max_parameter_delta = 0.0
    for inputs, targets in batches:
        compact_optimizer.zero_grad()
        dense_optimizer.zero_grad()
        F.cross_entropy(dsu(inputs), targets).backward()
        F.cross_entropy(fixed(inputs), targets).backward()
        compact_optimizer.step()
        dense_optimizer.step()
        max_parameter_delta = max(max_parameter_delta, max(_paired_parameter_deltas(dsu, fixed)))
        torch.testing.assert_close(dsu(inputs), fixed(inputs), rtol=RTOL, atol=ATOL)

    assert max_parameter_delta < 1e-5
    assert torch.all(fixed.dendrites.weight[~fixed.input_mask] == 0)
    assert torch.all(fixed.somata.weight[~fixed.soma_mask] == 0)
    for parameter, mask in ((fixed.dendrites.weight, fixed.input_mask),
                            (fixed.somata.weight, fixed.soma_mask)):
        state = dense_optimizer.state[parameter]
        assert torch.all(state["exp_avg"][~mask] == 0)
        assert torch.all(state["exp_avg_sq"][~mask] == 0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_device_forward_backward(device):
    dsu = DSUSv0(114).to(device)
    fixed = fixed_dann_from_dsu_s(dsu)
    assert all(buffer.device.type == device for buffer in fixed.buffers())
    x = torch.randn(2, 784, device=device)
    torch.testing.assert_close(dsu(x), fixed(x), rtol=RTOL, atol=ATOL)
    fixed(x).sum().backward()
    assert all(parameter.grad is not None for parameter in fixed.parameters())
