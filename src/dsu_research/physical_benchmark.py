"""S3-E02: paired physical accounting and CPU microbenchmarks, without training data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import statistics
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.benchmark import Timer

from .dsu_s import DSUSv0
from .fixed_dann import fixed_dann_from_dsu_s


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def model_accounting(model: torch.nn.Module) -> dict:
    parameters = {name: tensor_bytes(p) for name, p in model.named_parameters() if p.requires_grad}
    buffers = {name: tensor_bytes(b) for name, b in model.named_buffers() if name in model.state_dict()}
    return {
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_bytes": sum(parameters.values()),
        "parameters": parameters,
        "buffer_count": sum(model.state_dict()[name].numel() for name in buffers),
        "buffer_bytes": sum(buffers.values()),
        "buffers": buffers,
        "model_tensor_payload_bytes": sum(parameters.values()) + sum(buffers.values()),
    }


def optimizer_accounting(optimizer: torch.optim.Optimizer) -> dict:
    totals = {"exp_avg": 0, "exp_avg_sq": 0, "other": 0}
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                totals[name if name in totals else "other"] += tensor_bytes(value)
    return {**totals, "total": sum(totals.values())}


def serialized_bytes(value: object) -> int:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.tell()


def paired_models(seed: int = 20261009):
    torch.manual_seed(seed)
    compact = DSUSv0(topology_seed=seed)
    dense = fixed_dann_from_dsu_s(compact)
    assert torch.equal(dense.G, compact.G)
    assert dense.effective_parameters() == compact.effective_parameters()
    return dense, compact


def check_equivalence(dense, compact, x) -> float:
    assert torch.equal(dense.G, compact.G)
    dense.eval()
    compact.eval()
    with torch.inference_mode():
        a, b = dense(x), compact(x)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        return float((a - b).abs().max())


def make_inputs(seed: int = 104, sizes=(1, 32, 128, 512)):
    generator = torch.Generator().manual_seed(seed)
    return {size: (torch.randn(size, 784, generator=generator),
                   torch.randint(0, 10, (size,), generator=generator)) for size in sizes}


def train_step(model, optimizer, x, target, do_step: bool):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), target)
    loss.backward()
    if do_step:
        optimizer.step()


def timing(call, batch: int, threads: int, min_run_time: float = 0.15) -> dict:
    for _ in range(5):
        call()
    measurements = [Timer(stmt="call()", globals={"call": call}, num_threads=threads)
                    .blocked_autorange(min_run_time=min_run_time) for _ in range(3)]
    medians = [m.median * 1000 for m in measurements]
    median = statistics.median(medians)
    pooled = sorted(t * 1000 for m in measurements for t in m.times)
    q1 = pooled[len(pooled) // 4]
    q3 = pooled[(3 * len(pooled)) // 4]
    return {"median_ms_batch": median, "iqr_ms_batch": q3 - q1,
            "median_ms_sample": median / batch, "samples_per_second": 1000 * batch / median,
            "replicate_medians_ms": medians, "measurements": len(pooled),
            "timer_iterations_per_measurement": [m.number_per_run for m in measurements]}


def measure_pair(mode: str, batch: int, threads: int, inputs) -> dict:
    dense, compact = paired_models()
    x, target = inputs[batch]
    before = check_equivalence(dense, compact, x)
    results = {}
    for name, model in (("fixed", dense), ("dsu", compact)):
        if mode == "forward":
            model.eval()
            def call(model=model):
                with torch.inference_mode():
                    model(x)
        else:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            if mode == "adam":
                train_step(model, optimizer, x, target, True)  # initialize Adam state before timing
            def call(model=model, optimizer=optimizer):
                train_step(model, optimizer, x, target, mode == "adam")
        results[name] = timing(call, batch, threads)
    after = check_equivalence(dense, compact, x) if mode != "adam" else None
    results["max_logit_error_before"] = before
    results["max_logit_error_after_nonmutating"] = after
    results["fixed_over_dsu_latency_ratio"] = (results["fixed"]["median_ms_batch"] /
                                                results["dsu"]["median_ms_batch"])
    return results


def run() -> dict:
    original_threads = torch.get_num_threads()
    dense, compact = paired_models()
    inputs = make_inputs()
    fingerprint = {
        "os": platform.platform(), "cpu": platform.processor(),
        "logical_cores": os.cpu_count(), "python": sys.version.split()[0],
        "torch": torch.__version__, "device": "cpu", "dtype": "torch.float32",
        "original_threads": original_threads, "interop_threads": torch.get_num_interop_threads(),
        "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
    }
    topology_hash = hashlib.sha256(compact.G.numpy().astype("<i8", copy=False).tobytes()).hexdigest()
    storage = {}
    for name, model in (("fixed", dense), ("dsu", compact)):
        x, target = inputs[128]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model_only = serialized_bytes(model.state_dict())
        train_step(model, optimizer, x, target, True)
        account = model_accounting(model)
        state = optimizer_accounting(optimizer)
        assert state["exp_avg"] == state["exp_avg_sq"] == account["parameter_bytes"]
        storage[name] = {**account, "model_checkpoint_bytes": model_only,
                         "optimizer_state_bytes": state,
                         "known_training_tensor_payload_bytes": account["model_tensor_payload_bytes"] + state["total"],
                         "training_checkpoint_bytes": serialized_bytes({"model": model.state_dict(),
                                                                        "optimizer": optimizer.state_dict()})}
    regimes = {}
    try:
        for label, threads in (("single", 1), ("default", original_threads)):
            torch.set_num_threads(threads)
            regimes[label] = {"threads": threads, "forward": {}, "forward_backward": {}, "adam": {}}
            for batch in (1, 32, 128, 512):
                regimes[label]["forward"][str(batch)] = measure_pair("forward", batch, threads, inputs)
            for batch in (32, 128):
                regimes[label]["forward_backward"][str(batch)] = measure_pair("forward_backward", batch, threads, inputs)
            regimes[label]["adam"]["128"] = measure_pair("adam", 128, threads, inputs)
    finally:
        torch.set_num_threads(original_threads)
    return {"fingerprint": fingerprint, "topology_sha256": topology_hash,
            "initial_logit_max_error": check_equivalence(*paired_models(), inputs[128][0]),
            "storage": storage, "timings": regimes,
            "macs_per_sample": {"effective_both": 9984, "dense_matmul_fixed": 468224,
                                "compact_main_dsu": 9984, "dense_over_compact": 468224 / 9984},
            "activation_bytes": {str(batch): {"dsu_gather": batch * 128 * 4 * 16 * 4,
                                                "fixed_dendrites": batch * 512 * 4,
                                                "fixed_somata": batch * 128 * 4}
                                 for batch in inputs},
            "cpu_peak_runtime_memory": "NOT MEASURED RELIABLY"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
