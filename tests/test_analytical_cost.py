"""S3-T08 accounting against the measured structural controls."""

import json
from pathlib import Path

import pytest

from dsu_research.analysis import ArchitectureSpec, calculate_cost
from dsu_research.analysis.experiment_registry import build_registry


def test_known_structures():
    point = calculate_cost(ArchitectureSpec("point_sparse", branches=1, fan_in=64))
    assert (point.stored_parameters, point.main_macs_per_sample) == (9610, 9472)
    assert point.topology_bytes == 65536
    dsu = calculate_cost(ArchitectureSpec("dsu_s"))
    assert (dsu.stored_parameters, dsu.main_macs_per_sample, dsu.topology_bytes) == (10634, 9984, 65536)
    assert dsu.model_checkpoint_tensor_payload_bytes == 108072
    fixed = calculate_cost(ArchitectureSpec("fixed_dann"))
    assert (fixed.stored_parameters, fixed.effective_parameters,
            fixed.executed_dense_main_macs_per_sample) == (468874, 10634, 468224)
    assert fixed.model_checkpoint_tensor_payload_bytes == 2407976
    assert fixed.topology_bytes == 532480
    assert fixed.optimizer_state_bytes_estimate == 3750992
    vanilla = calculate_cost(ArchitectureSpec("dense_mlp", hidden=(512, 128)))
    assert (vanilla.stored_parameters, vanilla.main_macs_per_sample) == (468874, 468224)
    assert calculate_cost(ArchitectureSpec("point_sparse", branches=1, fan_in=64,
                                           batch_size=2, state_scalars_per_unit=1)).state_bytes == 1024
    physical_path = Path(__file__).parents[1] / "runs" / "physical_s3_t04.json"
    if physical_path.exists():
        measured = json.loads(physical_path.read_text(encoding="utf-8"))["storage"]
        for key, model in (("fixed", fixed), ("dsu", dsu)):
            assert model.stored_parameter_bytes == measured[key]["parameter_bytes"]
            assert model.persistent_buffer_bytes == measured[key]["buffer_bytes"]
            assert model.model_checkpoint_tensor_payload_bytes == measured[key]["model_tensor_payload_bytes"]
            assert model.optimizer_state_bytes_estimate == (measured[key]["optimizer_state_bytes"]["exp_avg"]
                                                            + measured[key]["optimizer_state_bytes"]["exp_avg_sq"])


@pytest.mark.parametrize("b", (1, 2, 4, 8))
def test_compartment_families(b):
    original = calculate_cost(ArchitectureSpec("compartmental_sparse", branches=b, fan_in=64 // b))
    neutral = calculate_cost(ArchitectureSpec("compartmental_sparse", branches=b,
                                              fan_in=64 // b, compartment_mode="neutral"))
    assert original.stored_parameters == 9610 + 256 * b
    assert original.main_macs_per_sample == 9472 + 128 * b
    assert neutral.stored_parameters == 9610
    assert neutral.main_macs_per_sample == 9472
    assert neutral.fixed_aggregate_additions_per_sample == 128 * (b - 1)


def test_historical_registry_is_reproducible_and_keeps_test_closed():
    saved = json.loads((Path(__file__).parents[1] / "data" / "historical_experiments.json").read_text(encoding="utf-8"))
    if (Path(__file__).parents[1] / "runs" / "physical_s3_t04.json").exists():
        assert build_registry() == saved
    rows = saved["records"]
    assert len(rows) == 30
    assert all(row["experiment_id"] and row["protocol_family"] and
               row["paired_comparison_group"] for row in rows)
    assert all(row["test_accuracy_mean"] is None for row in rows
               if row["experiment_id"] == "S2-T06" and row["architecture"] != "MLP 20->26")
