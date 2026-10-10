"""Rebuild the S3-T08 historical registry from local, already measured JSONs."""

import json
import statistics
from pathlib import Path

from .structural_cost import ArchitectureSpec, calculate_cost


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "data" / "historical_experiments.json"


def _read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _mean(values):
    return statistics.mean(values) if values else None


def _row(name, experiment_id, protocol_family, pair, paths, spec,
         test_paths=None, validation_only=False, initialization=None):
    records = [_read(p) for p in paths]
    cost = calculate_cost(spec)
    tests = [_read(p) for p in (test_paths or paths)] if not validation_only else []
    test_accuracy = [r["test"]["accuracy"] for r in tests if "test" in r]
    test_loss = [r["test"]["loss"] for r in tests if "test" in r]
    validation = []
    for r in records:
        if "history" in r:
            validation.append(r["history"][-1]["validation"])
        elif "training" in r and "history" in r["training"]:
            last = r["training"]["history"][-1]
            if "validation" in last:
                validation.append(last["validation"])
    first = records[0]
    topology = first.get("topology", {})
    if not isinstance(topology, dict):
        topology = {}
    diagnostics = first.get("initial_diagnostics", {})
    return {
        "architecture": name, "experiment_id": experiment_id,
        "protocol_family": protocol_family, "paired_comparison_group": pair,
        "source_files": paths + (test_paths or []), "seed_count": len(records),
        "H": spec.somas if spec.family != "dense_mlp" else None,
        "B": spec.branches if spec.family != "dense_mlp" else None,
        "K": spec.fan_in if spec.family not in ("dense_mlp", "dann_r") else None,
        "mean_K": spec.fan_in if spec.family == "dann_r" else None,
        "hidden_widths": list(spec.hidden) if spec.family == "dense_mlp" else None,
        "params_stored": cost.stored_parameters,
        "params_effective": cost.effective_parameters,
        "main_macs_per_sample": cost.main_macs_per_sample,
        "stored_parameter_bytes": cost.stored_parameter_bytes,
        "topology_bytes": cost.topology_bytes,
        "model_tensor_payload_bytes_estimate": cost.model_checkpoint_tensor_payload_bytes,
        "topology_characteristics": topology or ({"fan_in": "variable; mean 16"}
                                                 if spec.family == "dann_r" else None),
        "test_accuracy_mean": _mean(test_accuracy),
        "test_nll_mean": _mean(test_loss),
        "validation_accuracy_mean": _mean([v["accuracy"] for v in validation]),
        "validation_nll_mean": _mean([v["loss"] for v in validation]),
        "latency": None, "activation_stats": diagnostics or None,
        "initialization_scheme": initialization or first.get("initialization", {}).get("policy"),
    }


def build_registry():
    rows = []
    def add(name, exp, protocol, pair, directory, stem, seeds, spec, **kwargs):
        paths = [f"runs/{directory}/{stem}_seed{s}.json" for s in seeds]
        rows.append(_row(name, exp, protocol, pair, paths, spec, **kwargs))

    dense_512 = ArchitectureSpec("dense_mlp", hidden=(512, 128))
    for phase, filename in (("smoke", "s0_mlp_mnist_smoke.json"),
                            ("convergence", "s0_gate_convergence.json")):
        path = f"runs/{filename}"
        rows.append(_row("S0 MLP/ReLU 128", f"S0_{phase.upper()}",
                         "s0_mnist_diagnostic", f"s0_{phase}", [path],
                         ArchitectureSpec("dense_mlp", hidden=(128,)),
                         initialization="PyTorch Linear default"))
    add("vANN", "REPRODUCTION_001", "s1_reproduction", "s1_vann_dannr",
        "reproduction_001", "vann", range(1, 4), dense_512,
        initialization="Glorot uniform; zero biases")
    sparse_dense = ArchitectureSpec("dann_r", branches=4, fan_in=16)
    add("dANN-R", "REPRODUCTION_001", "s1_reproduction", "s1_vann_dannr",
        "reproduction_001", "dann_r", range(1, 4), sparse_dense,
        initialization="Glorot uniform; global random mask")
    add("vANN", "REPRODUCTION_001_N5", "s1_reproduction_n5", "s1_n5_vann_dannr",
        "reproduction_001", "vann", range(1, 6), dense_512,
        initialization="Glorot uniform; zero biases")
    add("dANN-R", "REPRODUCTION_001_N5", "s1_reproduction_n5", "s1_n5_vann_dannr",
        "reproduction_001", "dann_r", range(1, 6), sparse_dense,
        initialization="Glorot uniform; global random mask")
    add("dANN-R", "S3-T03", "s3_fixed_fanin", "s3_t03_ab",
        "fixed_fanin_s3_t03", "dann_r", range(1, 6), sparse_dense,
        initialization="paired dense-mask baseline")
    add("Fixed-dANN", "S3-T03", "s3_fixed_fanin", "s3_t03_ab",
        "fixed_fanin_s3_t03", "fixed_dann", range(1, 6),
        ArchitectureSpec("fixed_dann"), initialization="paired dense-mask control")
    add("DSU-S", "S2-T04", "s2_dsu_learning", "s2_t04_dannr_dsu",
        "dsu_s_v0_s2_t04", "dsu_s", range(1, 4), ArchitectureSpec("dsu_s"),
        initialization="Glorot uniform; zero biases")
    add("MLP 13->17", "S2-T05", "s2_parameter_matched", "s2_t05_mlp_dsu",
        "parameter_matched_mlp_s2_t05", "mlp", range(1, 4),
        ArchitectureSpec("dense_mlp", hidden=(13, 17)),
        initialization="Glorot uniform; zero biases")
    selected = _read("runs/mlp_capacity_s2_t06/selected_test.json")
    for h1, h2 in ((16, 21), (19, 25), (20, 26), (21, 27), (22, 29), (25, 33)):
        add(f"MLP {h1}->{h2}", "S2-T06", "s2_capacity_validation",
            "s2_t06_capacity", "mlp_capacity_s2_t06", f"h{h1}", range(1, 4),
            ArchitectureSpec("dense_mlp", hidden=(h1, h2)), validation_only=True,
            initialization="Glorot uniform; zero biases")
        if h1 == 20:
            rows[-1]["test_accuracy_mean"] = selected["accuracy"]["mean"]
            rows[-1]["test_nll_mean"] = selected["loss"]["mean"]
            rows[-1]["source_files"].append("runs/mlp_capacity_s2_t06/selected_test.json")
            rows[-1]["paired_comparison_group"] = "s2_t06_selected_dsu"
    for branches in (1, 2, 4, 8):
        k = 64 // branches
        name = f"{branches}x{k}"
        add(f"original compartment {name}", "S3-T05", "s3_compartment_original",
            "s3_t05_nested", "compartment_s3_t05", name, range(1, 4),
            ArchitectureSpec("compartmental_sparse", branches=branches, fan_in=k))
    add("PointSparse", "S3-T07", "s3_parameter_neutral", "s3_t07_neutral",
        "sparse_compartment_s3_t07", "PointSparse", range(1, 4),
        ArchitectureSpec("point_sparse", branches=1, fan_in=64),
        initialization="paired Kaiming input; Xavier output")
    for branches in (1, 2, 4, 8):
        k = 64 // branches
        name = f"{branches}x{k}"
        add(f"parameter-neutral {name}", "S3-T07", "s3_parameter_neutral",
            "s3_t07_neutral", "sparse_compartment_s3_t07", name, range(1, 4),
            ArchitectureSpec("compartmental_sparse", branches=branches,
                             fan_in=k, compartment_mode="neutral"),
            initialization="paired Kaiming input; Xavier output; frozen label-free calibration")
    diagnostics_path = "runs/s3_t06_initial_diagnostics.json"
    diagnostics = _read(diagnostics_path)["records"]
    for arm in ("PointSparse", "1x64", "2x32", "4x16", "8x8"):
        matching = [record for record in diagnostics if record["arm"] == arm]
        b = 1 if arm == "PointSparse" else int(arm.split("x")[0])
        spec = (ArchitectureSpec("point_sparse", branches=1, fan_in=64) if arm == "PointSparse"
                else ArchitectureSpec("compartmental_sparse", branches=b, fan_in=64 // b,
                                      compartment_mode="neutral"))
        cost = calculate_cost(spec)
        rows.append({
            "architecture": f"{arm} untrained diagnostics", "experiment_id": "S3-T06",
            "protocol_family": "s3_forward_calibration", "paired_comparison_group": "s3_t06_forward",
            "source_files": [diagnostics_path], "seed_count": len(matching),
            "H": 128, "B": b, "K": 64 // b, "mean_K": None, "hidden_widths": None,
            "params_stored": cost.stored_parameters, "params_effective": cost.effective_parameters,
            "main_macs_per_sample": cost.main_macs_per_sample,
            "stored_parameter_bytes": cost.stored_parameter_bytes,
            "topology_bytes": cost.topology_bytes,
            "model_tensor_payload_bytes_estimate": cost.model_checkpoint_tensor_payload_bytes,
            "topology_characteristics": {"G64": "nested, shared across arms by seed"},
            "test_accuracy_mean": None, "test_nll_mean": None,
            "validation_accuracy_mean": None, "validation_nll_mean": None,
            "latency": None, "activation_stats": [record["diagnostics"] for record in matching],
            "initialization_scheme": "paired Kaiming input; Xavier output; frozen label-free calibration",
        })
    physical = _read("runs/physical_s3_t04.json")
    for row in rows:
        if row["architecture"] in ("Fixed-dANN", "DSU-S"):
            key = "fixed" if row["architecture"] == "Fixed-dANN" else "dsu"
            row["latency"] = {mode: {batch: values[key]["median_ms_batch"]
                                     for batch, values in physical["timings"][mode]["forward"].items()}
                              for mode in ("single", "default")}
            row["latency_source"] = "runs/physical_s3_t04.json"
    return {"schema_version": 1, "units": {"accuracy": "fraction", "nll": "cross_entropy",
              "latency": "median_ms_per_batch", "bytes": "bytes"},
            "missing": "null means not measured or not reported for this protocol",
            "records": rows}


def main():
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(build_registry(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
