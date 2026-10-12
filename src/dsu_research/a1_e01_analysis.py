"""A1-E01: apply the frozen paired-contrast rules after all runs finish."""

from __future__ import annotations

import json
import statistics

from .a1_e01 import ARMS, ROOT, RUNS, SEEDS, T07, digest, preflight, validate_run, write_json
from .pointsparse_topology import make_topology, structural_metrics

CONTRASTS = (("C1", "A0", "A1"), ("C2", "A1", "A2"),
             ("C3", "A1", "A3"), ("C4", "A2", "A3"))


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "by_seed": dict(zip(SEEDS, values))}


def classify(delta_accuracy_pp, delta_nll):
    """Exact A1-DESIGN thresholds; report NLL compatibility separately."""
    mean_acc = statistics.mean(delta_accuracy_pp)
    mean_nll = statistics.mean(delta_nll)
    same_direction = sum(value * mean_acc > 0 for value in delta_accuracy_pp) >= 2
    nll_compatible = (abs(mean_nll) < .005 or mean_acc * mean_nll <= 0)
    if abs(mean_acc) >= .50 and same_direction and nll_compatible:
        status = "MATERIAL"
    elif .20 <= abs(mean_acc) < .50 and same_direction:
        status = "SMALL_BUT_REAL_SIGNAL"
    elif abs(mean_acc) < .20 and abs(mean_nll) < .005:
        status = "NEGLIGIBLE"
    else:
        status = "INCONCLUSIVE"
    return {"status": status, "same_direction_at_least_2_of_3": same_direction,
            "nll_compatible": nll_compatible}


def summarize():
    paths = { (arm, seed): RUNS / f"{arm}_seed{seed}.json"
             for seed in SEEDS for arm in ARMS }
    if any(not path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(f"A1-E01 incomplete; do not inspect test: {missing}")
    gate = preflight()
    records = {}
    for seed in SEEDS:
        old_path = T07 / f"PointSparse_seed{seed}.json"
        old = json.loads(old_path.read_text(encoding="utf-8"))
        a0_graph, a0_generation = make_topology("A0", seed)
        records["A0", seed] = {"arm": "A0", "seed": seed, "source_file": str(old_path.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": digest(old_path), "test": old["test"],
            "validation": old["training"]["history"][-1]["validation"],
            "history": old["training"]["history"], "duration_seconds": old["training"]["duration_seconds"],
            "topology_sha256": old["topology"]["g64_sha256"],
            "topology_generation": a0_generation,
            "topology_metrics": structural_metrics(a0_graph),
            "initialization": old["initialization"],
            "counts": old["counts"], "environment": old["environment"],
            "dataset": old["dataset"], "batch_order": old["batch_order"]}
        for arm in ARMS:
            path = paths[arm, seed]
            run = json.loads(path.read_text(encoding="utf-8"))
            validate_run(run, arm, seed, gate)
            records[arm, seed] = {"arm": arm, "seed": seed,
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "source_sha256": digest(path), "test": run["test"],
                "validation": run["training"]["history"][-1]["validation"],
                "history": run["training"]["history"],
                "duration_seconds": run["training"]["duration_seconds"],
                "topology_sha256": run["topology"]["sha256"],
                "topology_generation": run["topology"]["generation"],
                "topology_metrics": run["topology"]["metrics"],
                "initial_state_sha256": run["initial_state_sha256"],
                "initialization": run["initialization"],
                "counts": run["counts"], "environment": run["environment"],
                "dataset": run["dataset"], "batch_order": run["batch_order"]}
        paired = [records[arm, seed] for arm in ("A0", *ARMS)]
        for field in ("base_input_sha256", "soma_bias_sha256", "output_sha256", "output_bias_sha256"):
            if len({row["initialization"][field] for row in paired}) != 1:
                raise ValueError(f"unpaired initial weights: seed {seed}, {field}")
        for section, field in (("dataset", "train_indices_sha256"),
                               ("dataset", "validation_indices_sha256"),
                               ("batch_order", "combined_sha256")):
            if len({row[section][field] for row in paired}) != 1:
                raise ValueError(f"unpaired data or minibatches: seed {seed}, {section}.{field}")
    aggregates = {}
    for arm in ("A0", *ARMS):
        rows = [records[arm, seed] for seed in SEEDS]
        aggregates[arm] = {
            "test_accuracy_percent": stats([row["test"]["accuracy"] * 100 for row in rows]),
            "test_nll": stats([row["test"]["loss"] for row in rows]),
            "validation_accuracy_percent": stats([row["validation"]["accuracy"] * 100 for row in rows]),
            "validation_nll": stats([row["validation"]["loss"] for row in rows]),
        }
    contrasts = {}
    for label, first, second in CONTRASTS:
        acc = [(records[second, seed]["test"]["accuracy"] - records[first, seed]["test"]["accuracy"]) * 100
               for seed in SEEDS]
        nll = [records[second, seed]["test"]["loss"] - records[first, seed]["test"]["loss"]
               for seed in SEEDS]
        contrasts[label] = {"first": first, "second": second,
            "delta_test_accuracy_pp": stats(acc), "delta_test_nll": stats(nll),
            "classification": classify(acc, nll)}
    statuses = [row["classification"]["status"] for row in contrasts.values()]
    controlled_degree_statuses = [contrasts[label]["classification"]["status"]
                                  for label in ("C2", "C3", "C4")]
    if "MATERIAL" in controlled_degree_statuses:
        hypothesis = "MATERIAL_EFFECT"
    elif "INCONCLUSIVE" in controlled_degree_statuses:
        hypothesis = "INCONCLUSIVE"
    elif "SMALL_BUT_REAL_SIGNAL" in controlled_degree_statuses:
        hypothesis = "SMALL_SIGNAL"
    else:
        hypothesis = "NEGLIGIBLE"
    if "MATERIAL" in statuses:
        track = "ONE_MECHANISM_SPECIFIC_FOLLOWUP"
    elif "INCONCLUSIVE" in statuses:
        track = "BACKLOG_WITH_AT_MOST_ONE_DIAGNOSTIC"
    else:
        track = "BACKLOG_DEPRIORITIZED"
    return {"schema": "a1_e01_summary_v1", "source_commit": "4ce1fc7",
        "preflight": gate, "seeds": list(SEEDS), "arms": list(("A0", *ARMS)),
        "records": [records[arm, seed] for seed in SEEDS for arm in ("A0", *ARMS)],
        "aggregates": aggregates, "contrasts": contrasts,
        "H_A1_TOPOLOGY": hypothesis, "track_A_decision": track,
        "new_runs": 9, "new_epoch_equivalents": 225,
        "historical_A0_epoch_equivalents_reused": 75,
        "new_training_seconds": sum(records[arm, seed]["duration_seconds"]
                                    for seed in SEEDS for arm in ARMS)}


def main():
    result = summarize()
    path = ROOT / "data" / "a1_e01_results.json"
    write_json(path, result)
    print({label: row["classification"]["status"] for label, row in result["contrasts"].items()})
    print(f"H-A1-TOPOLOGY={result['H_A1_TOPOLOGY']}; Track A={result['track_A_decision']}")


if __name__ == "__main__":
    main()
