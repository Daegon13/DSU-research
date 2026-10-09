"""S3-E02 accounting and pairing guards; timing values are deliberately unasserted."""

import torch

from dsu_research.physical_benchmark import (
    check_equivalence, make_inputs, model_accounting, optimizer_accounting,
    paired_models, serialized_bytes, train_step,
)


def test_paired_storage_and_adam_state():
    dense, compact = paired_models()
    assert torch.equal(dense.G, compact.G)
    assert dense.effective_parameters() == compact.effective_parameters() == 10634
    x, target = make_inputs()[32]
    assert check_equivalence(dense, compact, x) < 1e-6
    for model in (dense, compact):
        account = model_accounting(model)
        assert account["parameter_bytes"] == sum(p.numel() * p.element_size() for p in model.parameters())
        assert account["buffer_bytes"] == sum(b.numel() * b.element_size() for b in model.buffers())
        assert account["model_tensor_payload_bytes"] == account["parameter_bytes"] + account["buffer_bytes"]
        assert serialized_bytes(model.state_dict()) > account["model_tensor_payload_bytes"]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_step(model, optimizer, x, target, True)
        state = optimizer_accounting(optimizer)
        assert state["exp_avg"] == state["exp_avg_sq"] == account["parameter_bytes"]
        assert state["total"] == sum(value.numel() * value.element_size()
                                     for entry in optimizer.state.values() for value in entry.values()
                                     if isinstance(value, torch.Tensor))
        assert serialized_bytes({"model": model.state_dict(), "optimizer": optimizer.state_dict()}) > 0
