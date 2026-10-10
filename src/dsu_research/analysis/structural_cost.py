"""Exact tensor accounting for the implemented static architecture families.

Counts describe logical tensors, not allocator peak, FLOPs, latency or energy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureSpec:
    family: str
    input_dim: int = 784
    outputs: int = 10
    hidden: tuple[int, ...] = ()
    somas: int = 128
    branches: int = 4
    fan_in: int = 16
    compartment_mode: str = "original"
    index_bytes: int = 8
    batch_size: int = 1
    parameter_bytes: int = 4
    state_scalars_per_unit: int = 0


@dataclass(frozen=True)
class StructuralCost:
    trainable_parameters: int
    stored_parameters: int
    effective_parameters: int
    effective_connections: int
    stored_parameter_bytes: int
    topology_bytes: int
    persistent_buffer_bytes: int
    optimizer_state_bytes_estimate: int
    model_checkpoint_tensor_payload_bytes: int
    main_macs_per_sample: int
    executed_dense_main_macs_per_sample: int | None
    temporary_activation_tensor_bytes: dict[str, int]
    state_bytes: int
    gather_indices_per_sample: int
    fixed_aggregate_additions_per_sample: int


def calculate_cost(spec: ArchitectureSpec) -> StructuralCost:
    """Account for current float parameter tensors and persistent buffers.

    Adam estimate counts two parameter-sized moments and excludes scalar steps,
    gradients and serialization headers. Activations are logical forward tensors.
    """
    for name in ("input_dim", "outputs", "somas", "branches", "fan_in",
                 "index_bytes", "batch_size", "parameter_bytes"):
        value = getattr(spec, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(spec.state_scalars_per_unit) is not int or spec.state_scalars_per_unit < 0:
        raise ValueError("state_scalars_per_unit must be a nonnegative integer")
    if spec.family == "dense_mlp":
        if not spec.hidden or any(type(h) is not int or h < 1 for h in spec.hidden):
            raise ValueError("dense_mlp needs positive hidden widths")
        widths = (spec.input_dim, *spec.hidden, spec.outputs)
        macs = sum(a * b for a, b in zip(widths, widths[1:]))
        params = macs + sum(widths[1:])
        effective = stored = params
        topology = 0
        buffers = 0
        gather = 0
        additions = 0
        dense_macs = macs
        activations = {f"hidden_{i}": spec.batch_size * h * spec.parameter_bytes
                       for i, h in enumerate(spec.hidden, 1)}
    elif spec.family in ("point_sparse", "dsu_s", "compartmental_sparse", "dann_r", "fixed_dann"):
        h, b, k, d, o = spec.somas, spec.branches, spec.fan_in, spec.input_dim, spec.outputs
        if k > d or (spec.family == "point_sparse" and b != 1):
            raise ValueError("invalid fan-in or point branches")
        inputs = h * b * k
        output_weights = h * o
        if spec.family == "point_sparse":
            effective = inputs + h + output_weights + o
            macs = inputs + output_weights
            additions = 0
            activations = {"selected_inputs": inputs, "soma_preactivation": h,
                           "soma_activation": h}
        elif spec.family == "compartmental_sparse" and spec.compartment_mode == "neutral":
            effective = inputs + h + output_weights + o
            macs = inputs + output_weights
            additions = h * (b - 1)
            activations = {"selected_inputs": inputs, "branch_preactivation": h * b,
                           "branch_activation": h * b, "aggregate": h,
                           "soma_activation": h}
        elif spec.family in ("dsu_s", "compartmental_sparse", "dann_r", "fixed_dann"):
            effective = inputs + h * b + h * b + h + output_weights + o
            macs = inputs + h * b + output_weights
            additions = 0
            activations = {"selected_inputs": inputs, "branch_preactivation": h * b,
                           "branch_activation": h * b, "soma_preactivation": h,
                           "soma_activation": h}
        else:
            raise ValueError("unknown compartment mode")
        if spec.family in ("dann_r", "fixed_dann"):
            # Dense parameters include inactive weights. bool masks are one byte/entry.
            dense_macs = d * h * b + h * b * h + output_weights
            stored = dense_macs + h * b + h + o
            mask_bytes = d * h * b + h * b * h
            topology = mask_bytes + (inputs * spec.index_bytes if spec.family == "fixed_dann" else 0)
            buffers = topology
            gather = 0
            activations.pop("selected_inputs")
        else:
            dense_macs = None
            stored = effective
            topology = inputs * spec.index_bytes
            buffers = topology + (2 * spec.parameter_bytes if spec.family == "compartmental_sparse" and spec.compartment_mode == "neutral" else 0)
            gather = inputs
        activations = {name: count * spec.batch_size * spec.parameter_bytes
                       for name, count in activations.items()}
    else:
        raise ValueError("unknown architecture family")
    activations["logits"] = spec.batch_size * spec.outputs * spec.parameter_bytes
    return StructuralCost(
        trainable_parameters=stored, stored_parameters=stored,
        effective_parameters=effective, effective_connections=macs,
        stored_parameter_bytes=stored * spec.parameter_bytes,
        topology_bytes=topology, persistent_buffer_bytes=buffers,
        optimizer_state_bytes_estimate=2 * stored * spec.parameter_bytes,
        model_checkpoint_tensor_payload_bytes=stored * spec.parameter_bytes + buffers,
        main_macs_per_sample=macs,
        executed_dense_main_macs_per_sample=dense_macs,
        temporary_activation_tensor_bytes=activations,
        state_bytes=spec.somas * spec.state_scalars_per_unit * spec.batch_size * spec.parameter_bytes,
        gather_indices_per_sample=gather,
        fixed_aggregate_additions_per_sample=additions)
