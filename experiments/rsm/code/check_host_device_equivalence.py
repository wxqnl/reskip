"""Require exact host/device token and skip-action equivalence before timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_action_stats(value):
    if isinstance(value, dict):
        return {
            key: normalize_action_stats(item)
            for key, item in value.items()
            if not key.startswith("native_device_conditional_")
        }
    if isinstance(value, list):
        normalized = [normalize_action_stats(item) for item in value]
        while normalized and normalized[-1] == 0:
            normalized.pop()
        return normalized
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=18)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.raw_rows.read_text().splitlines()]
    selected = {
        (row["task"], row["doc_id"], row["repetition"], row["variant"]): row
        for row in rows
        if row["variant"] in {"rsm_reskip", "host_rsm_reskip"}
    }
    keys = sorted(
        {
            (task, doc_id, repetition)
            for task, doc_id, repetition, _ in selected
        }
    )
    comparisons = []
    for task, doc_id, repetition in keys:
        device = selected.get((task, doc_id, repetition, "rsm_reskip"))
        host = selected.get((task, doc_id, repetition, "host_rsm_reskip"))
        if device is None or host is None:
            comparisons.append(
                {
                    "task": task,
                    "doc_id": doc_id,
                    "repetition": repetition,
                    "pair_complete": False,
                }
            )
            continue
        device_action_stats = normalize_action_stats(device["skip_stats"])
        host_action_stats = normalize_action_stats(host["skip_stats"])
        comparisons.append(
            {
                "task": task,
                "doc_id": doc_id,
                "repetition": repetition,
                "pair_complete": True,
                "generated_token_ids_exact": (
                    device["generated_token_ids"] == host["generated_token_ids"]
                ),
                "decoded_response_exact": (
                    device["decoded_response"] == host["decoded_response"]
                ),
                "block_equivalents_exact": (
                    device["block_equivalents_per_decode_token"]
                    == host["block_equivalents_per_decode_token"]
                ),
                "skip_action_stats_exact": device_action_stats == host_action_stats,
            }
        )

    exact_fields = (
        "generated_token_ids_exact",
        "decoded_response_exact",
        "block_equivalents_exact",
        "skip_action_stats_exact",
    )
    summary = {
        "expected_pairs": args.expected_pairs,
        "observed_pairs": len(comparisons),
        "complete_pairs": sum(item.get("pair_complete", False) for item in comparisons),
        **{
            field: sum(item.get(field, False) for item in comparisons)
            for field in exact_fields
        },
    }
    summary["passed"] = (
        summary["observed_pairs"] == args.expected_pairs
        and summary["complete_pairs"] == args.expected_pairs
        and all(summary[field] == args.expected_pairs for field in exact_fields)
    )
    payload = {
        "protocol": "same-loaded-model host/device equivalence on six official benchmark requests with three repetitions",
        "normalization": (
            "Ignore device-only native_device_conditional_* diagnostics and "
            "trim trailing zeros from per-block vectors before exact comparison."
        ),
        "summary": summary,
        "comparisons": comparisons,
        "source": str(args.raw_rows),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
