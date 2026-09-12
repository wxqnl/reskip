"""Cross-fitted diagnostic of RSG's incremental skip-risk information.

This is an analysis probe, not a proposed trained deployment controller. It
asks whether the new Full-path gate contains information beyond two existing
native AttnRes scalars on unseen samples.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _crossfit(records, feature_names, unsafe_kl, folds):
    sample_ids = np.asarray([row["sample_index"] for row in records], dtype=int)
    features = np.asarray(
        [[row[name] for name in feature_names] for row in records],
        dtype=np.float64,
    )
    target = np.asarray(
        [row["top1_flip"] or row["full_to_skip_kl"] > unsafe_kl for row in records],
        dtype=int,
    )
    prediction = np.zeros(len(records), dtype=np.float64)
    for fold in range(folds):
        test = sample_ids % folds == fold
        train = ~test
        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0)
        std[std < 1.0e-8] = 1.0
        model = LogisticRegression(
            C=0.2,
            max_iter=1000,
            class_weight="balanced",
            random_state=0,
        )
        model.fit((features[train] - mean) / std, target[train])
        prediction[test] = model.predict_proba(
            (features[test] - mean) / std
        )[:, 1]
    return sample_ids, target, prediction


def _selection(records, prediction, coverage):
    count = max(int(round(len(records) * coverage)), 1)
    selected = np.argsort(prediction, kind="mergesort")[:count]
    kl = np.asarray([row["full_to_skip_kl"] for row in records])
    flip = np.asarray([row["top1_flip"] for row in records], dtype=np.float64)
    return {
        "coverage": float(count / len(records)),
        "mean_skip_kl": float(kl[selected].mean()),
        "top1_flip_rate": float(flip[selected].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()
    with args.rows.open() as file:
        rows = [json.loads(line) for line in file]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["block"]), int(row["layer_offset"]))].append(row)

    native_features = ("w_recent", "correction_ratio")
    augmented_features = native_features + ("gate_factor", "gate_deviation")
    result = {
        "purpose": "diagnostic_upper_bound_not_a_deployment_controller",
        "split": f"sample_index_mod_{args.folds}",
        "native_features": list(native_features),
        "augmented_features": list(augmented_features),
        "thresholds": {},
    }
    for unsafe_kl in (0.01, 0.05, 0.10):
        action_results = {}
        native_auc = []
        augmented_auc = []
        for action, records in sorted(grouped.items()):
            _, target, native_prediction = _crossfit(
                records, native_features, unsafe_kl, args.folds
            )
            _, _, augmented_prediction = _crossfit(
                records, augmented_features, unsafe_kl, args.folds
            )
            native_value = float(roc_auc_score(target, native_prediction))
            augmented_value = float(roc_auc_score(target, augmented_prediction))
            native_auc.append(native_value)
            augmented_auc.append(augmented_value)
            action_results[f"b{action[0]}_o{action[1]}"] = {
                "unsafe_prevalence": float(target.mean()),
                "native_auc": native_value,
                "augmented_auc": augmented_value,
                "auc_delta": augmented_value - native_value,
                "native_safe_selection": {
                    str(coverage): _selection(records, native_prediction, coverage)
                    for coverage in (0.25, 0.50)
                },
                "augmented_safe_selection": {
                    str(coverage): _selection(records, augmented_prediction, coverage)
                    for coverage in (0.25, 0.50)
                },
            }
        native_macro = float(np.mean(native_auc))
        augmented_macro = float(np.mean(augmented_auc))
        result["thresholds"][str(unsafe_kl)] = {
            "macro_native_auc": native_macro,
            "macro_augmented_auc": augmented_macro,
            "macro_auc_delta": augmented_macro - native_macro,
            "actions_improved": int(
                sum(right > left for left, right in zip(native_auc, augmented_auc))
            ),
            "actions_total": len(native_auc),
            "macro_safe_selection": {
                str(coverage): {
                    "native_mean_skip_kl": float(
                        np.mean(
                            [
                                record["native_safe_selection"][str(coverage)]["mean_skip_kl"]
                                for record in action_results.values()
                            ]
                        )
                    ),
                    "augmented_mean_skip_kl": float(
                        np.mean(
                            [
                                record["augmented_safe_selection"][str(coverage)]["mean_skip_kl"]
                                for record in action_results.values()
                            ]
                        )
                    ),
                    "native_top1_flip_rate": float(
                        np.mean(
                            [
                                record["native_safe_selection"][str(coverage)]["top1_flip_rate"]
                                for record in action_results.values()
                            ]
                        )
                    ),
                    "augmented_top1_flip_rate": float(
                        np.mean(
                            [
                                record["augmented_safe_selection"][str(coverage)]["top1_flip_rate"]
                                for record in action_results.values()
                            ]
                        )
                    ),
                }
                for coverage in (0.25, 0.50)
            },
            "by_action": action_results,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
