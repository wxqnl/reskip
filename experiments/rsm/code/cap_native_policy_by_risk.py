"""Cap a native policy to the lowest-risk heldout AttnRes blocks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--native-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eligible-blocks", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    calibration = json.loads(args.calibration.read_text())
    policy = json.loads(args.native_policy.read_text())
    candidates = []
    for block in policy["eligible_blocks"]:
        evidence = calibration["evidence"][str(block)]
        validation = evidence["validation_native"]
        candidates.append(
            {
                "block": int(block),
                "validation_mean_risk": float(validation["mean_risk"]),
                "validation_mean_max_kl": float(validation["mean_max_kl"]),
                "validation_flip_rate": float(validation["any_flip_rate"]),
                "validation_coverage": float(validation["coverage"]),
                "layer_offsets": list(evidence["layer_offsets"]),
            }
        )
    ranking = sorted(
        candidates,
        key=lambda item: (
            item["validation_mean_risk"],
            item["validation_mean_max_kl"],
            item["block"],
        ),
    )
    chosen = sorted(item["block"] for item in ranking[: args.eligible_blocks])
    chosen_keys = {str(block) for block in chosen}
    capped = {
        **policy,
        "eligible_blocks": chosen,
        "feature_upper_bounds": {
            key: value
            for key, value in policy["feature_upper_bounds"].items()
            if key in chosen_keys
        },
        "feature_lower_bounds": {
            key: value
            for key, value in policy["feature_lower_bounds"].items()
            if key in chosen_keys
        },
        "action_by_block": {
            key: value
            for key, value in policy["action_by_block"].items()
            if key in chosen_keys
        },
        "layer_offsets_by_block": {
            key: value
            for key, value in policy["layer_offsets_by_block"].items()
            if key in chosen_keys
        },
        "max_skips": len(chosen),
        "selector_description": (
            "native w_recent bounds on the three lowest-heldout-risk AttnRes blocks; no learned skip controller"
        ),
    }
    receipt = {
        "eligible_block_count": len(chosen),
        "selected_blocks": chosen,
        "risk_ranking": ranking,
        "selection_uses_training_heldout_only": True,
        "policy": capped,
    }
    (args.output_dir / "capped_native_policy.json").write_text(
        json.dumps(capped, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "CAPPED_POLICY_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
