"""Evidence summary for the residual-strength-gated AttnRes screen."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONTROL = "F0_reused_attnres_h1"
CANDIDATE = "S1_rsg_h1"
SCREEN_SEED = 259123
EXPECTED_ADDED_PARAMETERS = 12294


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def paired_interval(candidate, baseline, key, *, replicates, seed):
    if len(candidate) != len(baseline):
        raise ValueError("paired arms have different sample counts")
    for left, right in zip(candidate, baseline):
        identity_left = (
            left["index"], left["source"], left["sequence_length"], left["tokens"]
        )
        identity_right = (
            right["index"], right["source"], right["sequence_length"], right["tokens"]
        )
        if identity_left != identity_right:
            raise ValueError(f"paired sample mismatch: {identity_left} != {identity_right}")
    numerator = f"{key}_sum"
    candidate_num = np.asarray([row[numerator] for row in candidate], dtype=np.float64)
    baseline_num = np.asarray([row[numerator] for row in baseline], dtype=np.float64)
    tokens = np.asarray([row["tokens"] for row in baseline], dtype=np.float64)

    def estimate(indices):
        return (candidate_num[indices].sum() - baseline_num[indices].sum()) / tokens[
            indices
        ].sum()

    observed = estimate(np.arange(len(candidate)))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(candidate), size=len(candidate))
        draws[index] = estimate(sampled)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"delta": float(observed), "ci95": [float(low), float(high)]}


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left, right):
    if len(left) < 3:
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _feature_risk_summary(rows, feature):
    values = np.asarray([row[feature] for row in rows], dtype=np.float64)
    kl = np.asarray([row["full_to_skip_kl"] for row in rows], dtype=np.float64)
    flip = np.asarray([row["top1_flip"] for row in rows], dtype=np.float64)
    if values.size == 0:
        return None
    lower, upper = np.quantile(values, [0.25, 0.75])
    low_mask = values <= lower
    high_mask = values >= upper
    return {
        "count": int(values.size),
        "feature_mean": float(values.mean()),
        "feature_std": float(values.std()),
        "feature_min": float(values.min()),
        "feature_max": float(values.max()),
        "spearman_with_skip_kl": _spearman(values, kl),
        "spearman_with_top1_flip": _spearman(values, flip),
        "low_quartile_mean_skip_kl": float(kl[low_mask].mean()),
        "high_quartile_mean_skip_kl": float(kl[high_mask].mean()),
        "low_quartile_top1_flip_rate": float(flip[low_mask].mean()),
        "high_quartile_top1_flip_rate": float(flip[high_mask].mean()),
    }


def _skip_relevance(rows):
    result = {
        "overall": {
            feature: _feature_risk_summary(rows, feature)
            for feature in (
                "w_recent",
                "historical_mass",
                "gate_factor",
                "gate_deviation",
                "correction_ratio",
            )
        },
        "by_block": {},
        "by_action": {},
    }
    grouped_block = defaultdict(list)
    grouped_action = defaultdict(list)
    for row in rows:
        grouped_block[int(row["block"])].append(row)
        grouped_action[(int(row["block"]), int(row["layer_offset"]))].append(row)
    for block, records in sorted(grouped_block.items()):
        result["by_block"][str(block)] = {
            feature: _feature_risk_summary(records, feature)
            for feature in ("w_recent", "gate_factor", "gate_deviation", "correction_ratio")
        }
    for (block, offset), records in sorted(grouped_action.items()):
        result["by_action"][f"b{block}_o{offset}"] = {
            feature: _feature_risk_summary(records, feature)
            for feature in ("w_recent", "gate_factor", "gate_deviation", "correction_ratio")
        }
    return result


def _full_effect_label(ce, kl):
    if ce["delta"] < 0.0 and kl["delta"] <= 0.0:
        if ce["ci95"][1] < 0.0 or kl["ci95"][1] < 0.0:
            return "improved_with_paired_evidence"
        return "slightly_improved_point_estimate"
    if ce["ci95"][0] <= 0.0 <= ce["ci95"][1] and kl["ci95"][0] <= 0.0 <= kl["ci95"][1]:
        return "statistically_indistinguishable"
    if ce["delta"] + kl["delta"] < 0.0 and ce["delta"] <= 0.003:
        return "small_tradeoff_with_joint_objective_improvement"
    if ce["delta"] > 0.005 or kl["delta"] > 0.01:
        return "meaningfully_degraded"
    return "mixed_or_small_effect"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=151731)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    summaries = {}
    rows = {}
    for arm in (CONTROL, CANDIDATE):
        heldout = root / f"{arm}_screen_seed{SCREEN_SEED}_n128"
        summaries[arm] = json.loads(
            (heldout / "HELDOUT_FULLPATH_RSG_SUMMARY.json").read_text()
        )
        rows[arm] = load_rows(heldout / "heldout_rows.jsonl")
    candidate_config = json.loads((root / CANDIDATE / "run_config.json").read_text())
    candidate_train = json.loads((root / CANDIDATE / "train_summary.json").read_text())
    risk_rows = load_rows(
        root / f"{CANDIDATE}_screen_seed{SCREEN_SEED}_n128" / "skip_risk_rows.jsonl"
    )

    ce = paired_interval(
        rows[CANDIDATE], rows[CONTROL], "ce",
        replicates=args.replicates, seed=args.seed,
    )
    kl = paired_interval(
        rows[CANDIDATE], rows[CONTROL], "kl",
        replicates=args.replicates, seed=args.seed + 1,
    )
    candidate = summaries[CANDIDATE]
    control = summaries[CONTROL]
    audit = candidate_config["parameter_audit"]
    integrity = bool(
        candidate_config["residual_strength_gate"]
        and audit["additional_trainable_parameters"] == EXPECTED_ADDED_PARAMETERS
        and audit["skip_specific_parameters"] == 0
        and audit["external_controller_parameters"] == 0
        and candidate_train["skip_training_forwards"] == 0
        and candidate_train["identity_max_abs_logit_delta"] <= 1.0e-4
        and candidate_config["fresh_pretrained_base"]
        and candidate_config["loaded_training_checkpoint"] is None
        and not candidate_config["warm_started_from_prior_experiment"]
        and not candidate_config["skip_branch_in_training"]
        and not candidate_config["skip_supervision"]
        and not candidate_config["coverage_or_compute_target"]
        and not candidate_config["second_stage_training"]
    )

    gate_ablation = candidate["residual_strength_gate_interventions"]
    identity_gate_ce_delta = gate_ablation["identity_gate"][
        "delta_ce_vs_learned_gate"
    ]
    identity_gate_kl_delta = gate_ablation["identity_gate"][
        "delta_kl_vs_learned_gate"
    ]
    cyclic_gate_ce_delta = gate_ablation["cyclic_gate_permutation"][
        "delta_ce_vs_learned_gate"
    ]
    cyclic_gate_kl_delta = gate_ablation["cyclic_gate_permutation"][
        "delta_kl_vs_learned_gate"
    ]
    factors = candidate["learned_residual_strength_factors"]
    factor_ranges = {
        block: record["max"] - record["min"] for block, record in factors.items()
    }
    max_factor_range = max(factor_ranges.values())
    mean_factor_std = float(np.mean([record["std"] for record in factors.values()]))
    full_effect = _full_effect_label(ce, kl)
    skip_relevance = _skip_relevance(risk_rows)

    if full_effect == "meaningfully_degraded":
        suggested_next_action = "stop_this_component"
    elif full_effect in {
        "improved_with_paired_evidence",
        "slightly_improved_point_estimate",
        "small_tradeoff_with_joint_objective_improvement",
    }:
        suggested_next_action = "run_seed_replication_then_vlm_and_reskip"
    else:
        suggested_next_action = "retain_as_simple_candidate_only_if_skip_relevance_is_useful"

    record = {
        "candidate": CANDIDATE,
        "control": CONTROL,
        "additional_trainable_parameters": audit["additional_trainable_parameters"],
        "candidate_student_ce": candidate["overall"]["student_ce"],
        "control_student_ce": control["overall"]["student_ce"],
        "ce_delta": ce["delta"],
        "ce_ci95_low": ce["ci95"][0],
        "ce_ci95_high": ce["ci95"][1],
        "candidate_teacher_kl": candidate["overall"]["teacher_kl"],
        "control_teacher_kl": control["overall"]["teacher_kl"],
        "kl_delta": kl["delta"],
        "kl_ci95_low": kl["ci95"][0],
        "kl_ci95_high": kl["ci95"][1],
        "native_adaptation_objective_delta": ce["delta"] + kl["delta"],
        "identity_gate_ce_delta": identity_gate_ce_delta,
        "identity_gate_kl_delta": identity_gate_kl_delta,
        "identity_gate_objective_delta": identity_gate_ce_delta + identity_gate_kl_delta,
        "cyclic_gate_ce_delta": cyclic_gate_ce_delta,
        "cyclic_gate_kl_delta": cyclic_gate_kl_delta,
        "cyclic_gate_objective_delta": cyclic_gate_ce_delta + cyclic_gate_kl_delta,
        "mean_gate_factor_std": mean_factor_std,
        "max_gate_factor_range": max_factor_range,
        "gate_factor_spearman_skip_kl": skip_relevance["overall"]["gate_factor"]["spearman_with_skip_kl"],
        "w_recent_spearman_skip_kl": skip_relevance["overall"]["w_recent"]["spearman_with_skip_kl"],
        "gate_deviation_spearman_skip_kl": skip_relevance["overall"]["gate_deviation"]["spearman_with_skip_kl"],
        "integrity_pass": integrity,
        "full_effect": full_effect,
        "suggested_next_action": suggested_next_action,
    }
    result = {
        "screen_seed": SCREEN_SEED,
        "paired_bootstrap_replicates": args.replicates,
        "interpretation_policy": "continuous_evidence_not_mechanical_pass_fail",
        "primary_question": "does_the_module_slightly_improve_full_attnres",
        "secondary_question": "does_its_native_token_signal_help_posthoc_reskip_analysis",
        "record": record,
        "gate_factor_by_block": factors,
        "skip_relevance": skip_relevance,
    }
    with open(output / "SCREEN_EVIDENCE.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    with open(
        output / "SCREEN_COMPARISON.csv", "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
