"""Focused structural checks for the S2-T05 baseline."""

import torch

from dsu_research.model import ParameterMatchedMLP


def test_architecture_logits_and_counts():
    model = ParameterMatchedMLP()
    assert [(m.in_features, m.out_features) for m in
            (model.hidden1, model.hidden2, model.output)] == [(784, 13), (13, 17), (17, 10)]
    assert [m.bias.abs().sum().item() for m in
            (model.hidden1, model.hidden2, model.output)] == [0, 0, 0]
    with torch.no_grad():
        for layer in (model.hidden1, model.hidden2, model.output):
            layer.weight.zero_()
            layer.bias.fill_(1)
        model.output.bias.fill_(3)
    logits = model(torch.zeros(2, 1, 28, 28))
    assert logits.shape == (2, 10)
    torch.testing.assert_close(logits, torch.full((2, 10), 3.0))
    assert sum(p.numel() for p in model.parameters()) == 10623
    assert all(p.requires_grad for p in model.parameters())
    assert model.structural_metrics() == {
        "trainable_parameter_bytes": 42492,
        "theoretical_main_macs_per_sample": 10583,
    }


def test_both_hidden_activations_use_leaky_relu_point_one():
    model = ParameterMatchedMLP()
    with torch.no_grad():
        model.hidden1.weight.zero_()
        model.hidden1.bias.fill_(-1)
        model.hidden2.weight.fill_(1)
        model.hidden2.bias.zero_()
        model.output.weight.fill_(1)
        model.output.bias.zero_()
    # -1 -> -0.1 in hidden1; 13 * -0.1 -> -0.13 in hidden2;
    # 17 * -0.13 = -2.21 logits (no softmax).
    torch.testing.assert_close(model(torch.zeros(1, 784)), torch.full((1, 10), -2.21))
