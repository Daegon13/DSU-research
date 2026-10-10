"""S3-T09 phase 2: retrospective ranking after forward proxies are sealed."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROXIES = ROOT / "data" / "zero_cost_proxy_results.json"
REGISTRY = ROOT / "data" / "historical_experiments.json"
OUTPUT = ROOT / "data" / "zero_cost_validation.json"

# Sign and diagnostic status were fixed before reading historical outcomes.
METRICS = {
    "effective_rank_raw": (1, False), "effective_rank_norm": (1, False),
    "participation_ratio_raw": (1, False), "participation_ratio_norm": (1, False),
    "mean_abs_corr": (-1, False), "median_abs_corr": (-1, False),
    "near_zero_fraction": (-1, False),
    "preactivation_scale.std": (1, True), "activation_std": (1, True),
    "logit_std": (1, True), "logit_norm": (1, True), "entropy": (1, True),
    "coverage_ratio": (1, False), "mean_jaccard_between_units": (-1, False),
    "activation_pattern_logdet": (1, False),
}
TARGETS = {"accuracy": ("test_accuracy_mean", 0.001),
           "negative_nll": ("test_nll_mean", 0.005)}


def _value(row, dotted):
    result = row
    for key in dotted.split("."):
        result = result[key]
    return result


def _ranks(values):
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    for start in range(len(order)):
        if start and values[order[start]] == values[order[start - 1]]:
            continue
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for i in order[start:end]:
            result[i] = rank
    return result


def spearman(x, y):
    if len(x) != len(y) or len(x) < 2:
        return None
    a, b = _ranks(x), _ranks(y)
    ma, mb = statistics.mean(a), statistics.mean(b)
    numerator = sum((i - ma) * (j - mb) for i, j in zip(a, b))
    denominator = math.sqrt(sum((i - ma) ** 2 for i in a) * sum((j - mb) ** 2 for j in b))
    return numerator / denominator if denominator else None


def kendall_tau_b(x, y):
    if len(x) != len(y) or len(x) < 3:
        return None
    concordant = discordant = tie_x = tie_y = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tie_x += 1
        elif dy == 0:
            tie_y += 1
        elif dx * dy > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt((concordant + discordant + tie_x) *
                            (concordant + discordant + tie_y))
    return (concordant - discordant) / denominator if denominator else None


def pairwise(x, y, target_tolerance):
    if len(x) != len(y) or len(x) < 2:
        return None
    correct = near_ties = scored = proxy_ties = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        delta_target = y[i] - y[j]
        if abs(delta_target) < target_tolerance:
            near_ties += 1
            continue
        scored += 1
        delta_proxy = x[i] - x[j]
        if abs(delta_proxy) <= 1e-12:
            correct += 0.5
            proxy_ties += 1
        elif delta_proxy * delta_target > 0:
            correct += 1
    total = scored + near_ties
    return {"correct_pairs": correct, "scored_pairs": scored,
            "near_tie_pairs": near_ties, "proxy_tie_pairs": proxy_ties,
            "accuracy_excluding_near_ties": correct / scored if scored else None,
            "accuracy_including_near_ties": (correct + 0.5 * near_ties) / total if total else None}


def aggregate_records(proxy_data):
    by_architecture = defaultdict(list)
    expected_seeds = proxy_data["seeds"]
    for record in proxy_data["records"]:
        by_architecture[record["architecture_id"]].append(record)
    aggregated = {}
    for architecture_id, records in by_architecture.items():
        records.sort(key=lambda r: r["proxy_seed"])
        if [r["proxy_seed"] for r in records] != expected_seeds:
            raise ValueError(f"missing or duplicate proxy seed: {architecture_id}")
        metrics = {}
        for name in METRICS:
            values = [_value(r, name) for r in records]
            metrics[name] = {"mean": statistics.mean(values) if all(v is not None for v in values) else None,
                             "sample_sd": statistics.stdev(values) if len(values) > 1 and
                             all(v is not None for v in values) else None}
        aggregated[architecture_id] = {
            "architecture_id": architecture_id, "seed_count": len(records), "metrics": metrics,
            "runtime_ms_mean": statistics.mean(r["runtime_ms"] for r in records),
            "runtime_ms_sample_sd": statistics.stdev(r["runtime_ms"] for r in records),
            "forward_passes": records[0]["forward_passes"],
            "representation_units": records[0]["representation_units"],
            "coverage_ratio_mean": statistics.mean(r["coverage_ratio"] for r in records),
        }
    return aggregated


def validate(proxy_data, registry_data):
    aggregated = aggregate_records(proxy_data)
    historical = {f"{row['experiment_id']}|{row['architecture']}": row
                  for row in registry_data["records"]}
    if any(key not in historical for key in aggregated):
        raise ValueError("proxy architecture not found in historical registry")
    groups = defaultdict(list)
    for key, proxy in aggregated.items():
        historical_row = historical[key]
        groups[(historical_row["protocol_family"],
                historical_row["paired_comparison_group"])].append((key, proxy, historical_row))
    comparisons = []
    for (family, pair), rows in sorted(groups.items()):
        rows.sort(key=lambda triple: triple[0])
        for target_name, (field, tolerance) in TARGETS.items():
            usable = [triple for triple in rows if triple[2][field] is not None]
            if len(usable) < 2:
                continue
            target = [(-row[field] if target_name == "negative_nll" else row[field])
                      for _, _, row in usable]
            proxy_results = {}
            for name, (direction, diagnostic_only) in METRICS.items():
                values = [proxy["metrics"][name]["mean"] for _, proxy, _ in usable]
                if any(value is None for value in values):
                    proxy_results[name] = None
                    continue
                oriented = [direction * value for value in values]
                proxy_results[name] = {
                    "direction": direction, "diagnostic_only": diagnostic_only,
                    "spearman_raw": spearman(values, target),
                    "spearman_oriented": spearman(oriented, target),
                    "kendall_tau_b_raw": kendall_tau_b(values, target),
                    "kendall_tau_b_oriented": kendall_tau_b(oriented, target),
                    "pairwise": pairwise(oriented, target, tolerance),
                }
            comparisons.append({"protocol_family": family, "paired_comparison_group": pair,
                                "target": target_name, "near_tie_tolerance": tolerance,
                                "architecture_ids": [key for key, _, _ in usable],
                                "target_values": target, "metric_results": proxy_results})
    return {"schema_version": 1, "phase": "retrospective_validation",
            "target_source": "data/historical_experiments.json",
            "aggregation": "mean and sample SD over zero-cost seeds 1,2,3",
            "architectures": aggregated, "groups": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxies", type=Path, default=PROXIES)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    proxy_bytes = args.proxies.read_bytes()
    proxy_data = json.loads(proxy_bytes)
    registry_data = json.loads(args.registry.read_text(encoding="utf-8"))
    result = validate(proxy_data, registry_data)
    result["proxy_file_sha256"] = hashlib.sha256(proxy_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{len(result['architectures'])} architectures; {len(result['groups'])} target/group comparisons; {args.output}")


if __name__ == "__main__":
    main()
