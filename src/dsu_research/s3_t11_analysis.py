"""S3-T11 post-training analysis; rules were sealed in the preregistration."""

from __future__ import annotations

import argparse
import json
import statistics

from .analysis.zero_cost_validation import kendall_tau_b, pairwise, spearman
from .s3_t11 import (EPOCHS_EARLY, HISTORICAL_GROUPS, NEAR_TIE_ACC, PREREG,
                     RESULTS, ROOT, RUNS, SEEDS, SHORTLIST, T07, T10, _sha,
                     _validate_new, _write, audit_anchor, sealed_prereg)


def _history(record):
    return record.get("history", record.get("training", {}).get("history"))


def _metrics(predictor, target):
    return {"spearman": spearman(predictor, target),
            "kendall_tau_b": kendall_tau_b(predictor, target),
            "pairwise": pairwise(predictor, target, NEAR_TIE_ACC)}


def _mean_sd(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "by_seed": values}


def _historical_records(prereg):
    result = {}
    for group, (directory, arms) in HISTORICAL_GROUPS.items():
        reference = prereg["historical_source_files"][group]
        by_arm = {}
        index = 0
        for arm in arms:
            rows = []
            for seed in SEEDS:
                path = ROOT / directory / f"{arm}_seed{seed}.json"
                if reference[index]["file"] != str(path.relative_to(ROOT)).replace("\\", "/") or _sha(path) != reference[index]["sha256"]:
                    raise ValueError(f"historical source drift: {path}")
                row = json.loads(path.read_text(encoding="utf-8"))
                history = _history(row)
                if len(history) != 25 or [h["epoch"] for h in history] != list(range(1, 26)):
                    raise ValueError(f"incomplete historical curve: {path}")
                rows.append(row)
                index += 1
            by_arm[arm] = rows
        result[group] = by_arm
    return result


def _historical_proxy_means():
    data = json.loads((ROOT / "data" / "zero_cost_proxy_results.json").read_text(encoding="utf-8"))
    sums = {}
    for row in data["records"]:
        sums.setdefault(row["architecture_id"], []).append(row["effective_rank_raw"])
    return {key: statistics.mean(values) for key, values in sums.items() if len(values) == 3}


def historical_analysis(prereg):
    groups = _historical_records(prereg)
    proxies = _historical_proxy_means()
    output = {}
    for group, arms in groups.items():
        names = list(arms)
        final_val = [statistics.mean(_history(row)[24]["validation"]["accuracy"] for row in arms[arm])
                     for arm in names]
        has_test = all(row.get("test") and row["test"].get("accuracy") is not None
                       for records in arms.values() for row in records)
        targets = {"final_validation_accuracy": final_val}
        if has_test:
            targets["final_test_accuracy"] = [statistics.mean(row["test"]["accuracy"] for row in arms[arm])
                                              for arm in names]
        epochs = {}
        per_seed = {}
        for epoch in EPOCHS_EARLY:
            early = [statistics.mean(_history(row)[epoch - 1]["validation"]["accuracy"]
                                     for row in arms[arm]) for arm in names]
            early_loss = [statistics.mean(_history(row)[epoch - 1]["validation"]["loss"]
                                          for row in arms[arm]) for arm in names]
            epochs[str(epoch)] = {"early_validation_accuracy": early,
                                  "early_validation_loss": early_loss,
                                  "vs_final": {target: _metrics(early, values)
                                               for target, values in targets.items()},
                                  "negative_loss_vs_final": {target: _metrics([-v for v in early_loss], values)
                                                             for target, values in targets.items()}}
            per_seed[str(epoch)] = {}
            for target in targets:
                per_seed[str(epoch)][target] = []
                for seed_index, seed in enumerate(SEEDS):
                    x = [_history(arms[arm][seed_index])[epoch - 1]["validation"]["accuracy"]
                         for arm in names]
                    loss_x = [-_history(arms[arm][seed_index])[epoch - 1]["validation"]["loss"]
                              for arm in names]
                    y = [(_history(arms[arm][seed_index])[24]["validation"]["accuracy"]
                          if target == "final_validation_accuracy" else arms[arm][seed_index]["test"]["accuracy"])
                         for arm in names]
                    per_seed[str(epoch)][target].append({"seed": seed,
                                                         "accuracy": _metrics(x, y),
                                                         "negative_loss": _metrics(loss_x, y)})
        er = None
        if group == "s3_t07_parameter_neutral":
            keys = [f"S3-T07|{arm if arm == 'PointSparse' else 'parameter-neutral ' + arm}" for arm in names]
        elif group == "s3_t05_original_compartments":
            keys = [f"S3-T05|original compartment {arm}" for arm in names]
        else:
            keys = None
        if keys is not None:
            if any(key not in proxies for key in keys):
                raise ValueError(f"missing ER proxies for {group}")
            values = [proxies[key] for key in keys]
            er = {"er_raw_mean": values,
                  "vs_final": {target: _metrics(values, y) for target, y in targets.items()}}
        output[group] = {"architecture_ids": names, "targets": targets,
                         "epochs": epochs, "per_seed": per_seed,
                         "er_raw": er, "test_comparable": has_test}
    return output


def classify_early(groups):
    stats = [group["epochs"]["3"]["vs_final"]["final_validation_accuracy"] for group in groups.values()]
    systematic_inverse = any(row["spearman"] is not None and row["spearman"] <= -.5 and
                             row["pairwise"]["accuracy_excluding_near_ties"] is not None and
                             row["pairwise"]["accuracy_excluding_near_ties"] <= .3 for row in stats)
    strong = sum(row["spearman"] is not None and row["spearman"] >= .5 and
                 row["pairwise"]["accuracy_excluding_near_ties"] is not None and
                 row["pairwise"]["accuracy_excluding_near_ties"] > .7 for row in stats)
    positive = sum(row["spearman"] is not None and row["spearman"] > 0 and
                   row["pairwise"]["accuracy_excluding_near_ties"] is not None and
                   row["pairwise"]["accuracy_excluding_near_ties"] > .5 for row in stats)
    if systematic_inverse:
        return "FAILED"
    if strong >= 2 and all(row["spearman"] is not None and row["spearman"] >= 0 and
                           row["pairwise"]["accuracy_excluding_near_ties"] is not None and
                           row["pairwise"]["accuracy_excluding_near_ties"] >= .5 for row in stats):
        return "PROMISING"
    if positive >= 2:
        return "WEAK"
    return "FAILED"


def classify_width(arms):
    anchor = arms["PointSparse_128x64"]
    comparisons = {}
    status = "NOT_SUPPORTED"
    for key in ("PointSparse_80x109", "PointSparse_160x49"):
        candidate = arms[key]
        dv = candidate["validation_accuracy"]["mean"] - anchor["validation_accuracy"]["mean"]
        dt = candidate["test_accuracy"]["mean"] - anchor["test_accuracy"]["mean"]
        dnll = candidate["test_loss"]["mean"] - anchor["test_loss"]["mean"]
        aligned = dv * dt > 0
        direction = 1 if dv > 0 else -1
        val_wins = sum(direction * (a - b) > 0 for a, b in zip(candidate["validation_accuracy"]["by_seed"],
                                                              anchor["validation_accuracy"]["by_seed"]))
        test_wins = sum(direction * (a - b) > 0 for a, b in zip(candidate["test_accuracy"]["by_seed"],
                                                               anchor["test_accuracy"]["by_seed"]))
        strong = (aligned and abs(dv) >= .005 and abs(dt) >= .005 and val_wins >= 2 and
                  test_wins >= 2 and direction * dnll <= .005)
        weak = aligned and abs(dv) >= .002 and abs(dt) >= .002 and val_wins >= 2 and test_wins >= 2
        flat = abs(dv) < .002 and abs(dt) < .002 and abs(dnll) < .005
        comparisons[key] = {"delta_validation_accuracy_pp": dv * 100,
                            "delta_test_accuracy_pp": dt * 100, "delta_test_nll": dnll,
                            "paired_validation_direction_seeds": val_wins,
                            "paired_test_direction_seeds": test_wins,
                            "strong_rule": strong, "weak_rule": weak, "flat_rule": flat}
        if strong:
            status = "SUPPORTED_WITHIN_TESTED_RANGE"
        elif weak and status != "SUPPORTED_WITHIN_TESTED_RANGE":
            status = "WEAK_SUPPORT"
        elif not flat and status == "NOT_SUPPORTED":
            status = "INCONCLUSIVE"
    return status, comparisons


def _all_shortlist_records(prereg, prereg_sha, commit):
    if audit_anchor() != prereg["anchor_audit"]:
        raise ValueError("anchor source changed")
    old = {}
    new = {}
    for seed in SEEDS:
        old[seed] = json.loads((T07 / f"PointSparse_seed{seed}.json").read_text(encoding="utf-8"))
    for h, k in SHORTLIST[:2]:
        key = f"PointSparse_{h}x{k}"
        new[key] = []
        for seed in SEEDS:
            path = RUNS / f"{key}_seed{seed}.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            _validate_new(row, h, k, seed, prereg_sha, commit)
            new[key].append(row)
    return old, new


def analyze():
    prereg, prereg_sha, commit = sealed_prereg()
    if _sha(T10) != prereg["t10_results_sha256"]:
        raise ValueError("T10 pilot source changed")
    old, new = _all_shortlist_records(prereg, prereg_sha, commit)
    arms = {}
    all_records = {}
    for key, rows in new.items():
        all_records[key] = rows
    all_records["PointSparse_128x64"] = [old[seed] for seed in SEEDS]
    for key, rows in all_records.items():
        def val(row, name):
            if name == "train_loss":
                return _history(row)[24]["train_loss"]
            if name.startswith("validation_"):
                return _history(row)[24]["validation"][name.removeprefix("validation_")]
            if name.startswith("test_"):
                return row["test"][name.removeprefix("test_")]
            raise KeyError(name)
        metrics = ("train_loss", "validation_loss", "validation_accuracy", "test_loss", "test_accuracy",
                   "test_mean_logit_l2", "test_predictive_entropy", "test_correct_class_margin",
                   "test_fraction_positive_margins", "test_ece_15")
        h, k = ((80, 109) if key.endswith("80x109") else
                (160, 49) if key.endswith("160x49") else (128, 64))
        arms[key] = {"H": h, "K": k, "structural_cost": next(r["structural_cost"] for r in prereg["shortlist"] if (r["H"], r["K"]) == (h, k)),
                     "source": "S3-T07 reused" if key.endswith("128x64") else "S3-T11 new",
                     **{name: _mean_sd([val(row, name) for row in rows]) for name in metrics},
                     "duration_seconds": _mean_sd([row["training"]["duration_seconds"] if "training" in row else row["duration_seconds"] for row in rows])}
    names = [f"PointSparse_{h}x{k}" for h, k in SHORTLIST]
    final_val = [arms[name]["validation_accuracy"]["mean"] for name in names]
    final_test = [arms[name]["test_accuracy"]["mean"] for name in names]
    t10 = json.loads(T10.read_text(encoding="utf-8"))
    pilot = {r["architecture_id"]: r["history"][2]["validation"]["accuracy"] for r in t10["pilot_records"]}
    t10_proxy = json.loads((ROOT / "data" / "s3_t10_preregistered_screen.json").read_text(encoding="utf-8"))
    er = {r["architecture_id"]: r["proxy_summary"]["effective_rank_raw"]["mean"] for r in t10_proxy["architectures"]}
    pilot_values = [pilot[name] for name in names]
    er_values = [er[name] for name in names]
    seed1_val = [_history(all_records[name][0])[24]["validation"]["accuracy"] for name in names]
    seed1_test = [all_records[name][0]["test"]["accuracy"] for name in names]
    historical = historical_analysis(prereg)
    early_status = classify_early(historical)
    width_status, width_comparisons = classify_width(arms)
    rank = lambda target: [names[i] for i in sorted(range(3), key=lambda i: target[i], reverse=True)]
    prospect = {"t10_epoch3_ranking": prereg["t10_shortlist_epoch3_ranking"],
                "final_validation_seed1_ranking": rank(seed1_val),
                "final_test_seed1_ranking": rank(seed1_test),
                "final_validation_mean_ranking": rank(final_val),
                "final_test_mean_ranking": rank(final_test),
                "predictors": {"t10_epoch3_validation_accuracy": pilot_values,
                               "er_raw_mean": er_values},
                "targets": {"final_validation_seed1": seed1_val, "final_test_seed1": seed1_test,
                            "final_validation_mean": final_val, "final_test_mean": final_test},
                "comparisons": {target: {"t10_epoch3": _metrics(pilot_values, values),
                                         "er_raw": _metrics(er_values, values)}
                                for target, values in (("final_validation_seed1", seed1_val),
                                                       ("final_test_seed1", seed1_test),
                                                       ("final_validation_mean", final_val),
                                                       ("final_test_mean", final_test))}}
    result = {"schema": "s3_t11_results_v1", "prereg_commit": commit,
              "prereg_sha256": prereg_sha, "anchor_audit": prereg["anchor_audit"],
              "shortlist": arms, "new_run_records": new,
              "anchor_reused_records": [old[seed] for seed in SEEDS],
              "prospective_t10_to_t11": prospect,
              "historical_early_pilot": historical,
              "EARLY-PILOT-3E": early_status,
              "H-WIDTH-FANIN": width_status, "width_comparisons": width_comparisons,
              "compute": {"t10_epoch_equivalents": 24, "t11_new_epoch_equivalents": 150,
                          "incremental_total": 174, "naive_all_eight": 600,
                          "saved_fraction": 1 - 174 / 600,
                          "anchor_historical_epoch_equivalents_reused": 75}}
    _write(RESULTS, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("analyze",))
    parser.parse_args()
    result = analyze()
    print(json.dumps({"EARLY-PILOT-3E": result["EARLY-PILOT-3E"],
                      "H-WIDTH-FANIN": result["H-WIDTH-FANIN"],
                      "prospective": result["prospective_t10_to_t11"],
                      "compute": result["compute"]}, indent=2))


if __name__ == "__main__":
    main()
