"""Calibrate one simple token-level native ReSkip policy from forced actions.

The policy remains a set of scalar bounds. No selector module is trained. The
first 16 samples choose thresholds and the last 16 samples are reported once
as validation. RSG is used only at block 1, where the cross-fitted diagnostic
showed the clearest consistent increment across all three micro-actions.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ACTIONS = {
    1: (1, 2, 3),
    2: (1,),
    3: (0, 1),
    4: (1,),
    5: (0, 3),
}
GATE_ELIGIBLE_BLOCKS = {1}


def _group(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["block"]), int(row["sample_index"]), int(row["token_position"]))].append(row)
    result = defaultdict(list)
    for (block, sample, token), records in grouped.items():
        result[block].append(
            {
                "sample": sample,
                "token": token,
                "w_recent": float(records[0]["w_recent"]),
                "gate_factor": float(records[0]["gate_factor"]),
                "risk": max(float(row["full_to_skip_kl"]) for row in records)
                + 0.25 * float(any(bool(row["top1_flip"]) for row in records)),
                "max_kl": max(float(row["full_to_skip_kl"]) for row in records),
                "any_flip": float(any(bool(row["top1_flip"]) for row in records)),
            }
        )
    return result


def _mask(records, rule):
    recent = np.asarray([row["w_recent"] for row in records])
    factor = np.asarray([row["gate_factor"] for row in records])
    selected = recent >= rule["w_recent_lower"]
    if "gate_factor_upper" in rule:
        selected &= factor < rule["gate_factor_upper"]
    return selected


def _summary(records, rule):
    selected = _mask(records, rule)
    if not selected.any():
        return {
            "coverage": 0.0,
            "mean_risk": None,
            "mean_max_kl": None,
            "any_flip_rate": None,
        }
    return {
        "coverage": float(selected.mean()),
        "mean_risk": float(np.mean([row["risk"] for row, keep in zip(records, selected) if keep])),
        "mean_max_kl": float(np.mean([row["max_kl"] for row, keep in zip(records, selected) if keep])),
        "any_flip_rate": float(np.mean([row["any_flip"] for row, keep in zip(records, selected) if keep])),
    }


def _native_rule(records, target_coverage):
    recent = np.asarray([row["w_recent"] for row in records])
    candidates = []
    for quantile in np.linspace(0.0, 1.0, 101):
        rule = {"w_recent_lower": float(np.quantile(recent, quantile))}
        summary = _summary(records, rule)
        candidates.append(
            (
                abs(summary["coverage"] - target_coverage),
                summary["mean_risk"],
                rule,
                summary,
            )
        )
    return min(candidates, key=lambda item: (item[0], item[1]))[2:]


def _gate_rule(records, target_coverage):
    recent = np.asarray([row["w_recent"] for row in records])
    factor = np.asarray([row["gate_factor"] for row in records])
    candidates = []
    for recent_quantile in np.arange(0.0, 0.81, 0.05):
        recent_threshold = float(np.quantile(recent, recent_quantile))
        for factor_quantile in np.arange(0.1, 1.0, 0.1):
            # Add a tiny margin because deployment uses a strict upper bound
            # and the learned factors are quantized to bfloat16 values.
            factor_threshold = float(np.quantile(factor, factor_quantile) + 1.0e-6)
            rule = {
                "w_recent_lower": recent_threshold,
                "gate_factor_upper": factor_threshold,
            }
            summary = _summary(records, rule)
            if not 0.38 <= summary["coverage"] <= 0.52:
                continue
            objective = summary["mean_risk"] + 0.03 * abs(
                summary["coverage"] - target_coverage
            )
            candidates.append((objective, rule, summary))
    if not candidates:
        raise RuntimeError("no gate rule met the calibration coverage window")
    _, rule, summary = min(candidates, key=lambda item: item[0])
    return rule, summary


def _config(rules):
    upper = {}
    lower = {}
    for block, rule in rules.items():
        lower[str(block)] = {"w_recent": rule["w_recent_lower"]}
        upper[str(block)] = {}
        if "gate_factor_upper" in rule:
            upper[str(block)]["residual_strength_factor"] = rule[
                "gate_factor_upper"
            ]
    return {
        "feature_upper_bounds": upper,
        "feature_lower_bounds": lower,
        "eligible_blocks": sorted(ACTIONS),
        "action_by_block": {str(block): "layer" for block in ACTIONS},
        "layer_offsets_by_block": {
            str(block): list(offsets) for block, offsets in ACTIONS.items()
        },
        "max_skips": len(ACTIONS),
        "decode_only": True,
        "selector_description": "native rectangular bounds; no learned skip controller",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-coverage", type=float, default=0.45)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with args.rows.open() as file:
        rows = [json.loads(line) for line in file]
    grouped = _group(rows)
    native_rules = {}
    gate_rules = {}
    evidence = {}
    for block in sorted(ACTIONS):
        calibration = [row for row in grouped[block] if row["sample"] < 16]
        validation = [row for row in grouped[block] if row["sample"] >= 16]
        native_rule, native_calibration = _native_rule(
            calibration, args.target_coverage
        )
        native_rules[block] = native_rule
        if block in GATE_ELIGIBLE_BLOCKS:
            gate_rule, gate_calibration = _gate_rule(
                calibration, args.target_coverage
            )
        else:
            gate_rule, gate_calibration = native_rule, native_calibration
        gate_rules[block] = gate_rule
        evidence[str(block)] = {
            "layer_offsets": list(ACTIONS[block]),
            "native_rule": native_rule,
            "gate_augmented_rule": gate_rule,
            "calibration_native": native_calibration,
            "calibration_gate_augmented": gate_calibration,
            "validation_native": _summary(validation, native_rule),
            "validation_gate_augmented": _summary(validation, gate_rule),
        }

    for name, rules in (
        ("native_only", native_rules),
        ("gate_augmented", gate_rules),
    ):
        config = _config(rules)
        (args.output_dir / f"{name}_policy.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
    block_evidence = list(evidence.items())
    for split in ("calibration", "validation"):
        for variant in ("native", "gate_augmented"):
            expected_layers = sum(
                len(ACTIONS[int(block)])
                * record[f"{split}_{variant}"]["coverage"]
                for block, record in block_evidence
            )
            evidence[f"{split}_{variant}_layer_skips_per_token"] = expected_layers
            evidence[f"{split}_{variant}_block_equivalents_per_token"] = expected_layers / 4.0
    result = {
        "status": "posthoc_exploratory_policy_screen",
        "calibration_samples": list(range(16)),
        "validation_samples": list(range(16, 32)),
        "target_per_block_coverage": args.target_coverage,
        "gate_eligible_blocks_fixed_from_crossfit_diagnostic": sorted(GATE_ELIGIBLE_BLOCKS),
        "risk_definition": "max micro-action KL within block plus 0.25 times any top-1 flip",
        "evidence": evidence,
    }
    (args.output_dir / "POLICY_CALIBRATION.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
