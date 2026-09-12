"""Summarize Full and two native ReSkip policy screens."""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data/Minko/experiments/attnres_residual_strength_gate_2b_20260831")
TASKS = (
    "ai2d", "mmbench_en_dev_static", "mmmu_val", "mmstar", "ocrbench",
    "realworldqa",
)


def _result(arm):
    paths = sorted(glob.glob(str(ROOT / "vlm_screen_n256" / arm / "models__*" / "*_results.json")))
    if not paths:
        raise FileNotFoundError(arm)
    return json.load(open(paths[-1])), paths[-1]


def _score(payload, task):
    record = payload["results"][task]
    keys = {
        "ai2d": ("exact_match,flexible-extract", 100.0),
        "mmbench_en_dev_static": ("gpt_eval_score,none", 1.0),
        "mmmu_val": ("mmmu_acc,none", 100.0),
        "mmstar": ("average,none", 100.0),
        "ocrbench": ("ocrbench_accuracy,none", 100.0),
        "realworldqa": ("exact_match,none", 100.0),
    }
    key, scale = keys[task]
    return scale * float(record[key])


def _skip(path):
    raw = json.load(open(path))
    decode = int(raw.get("decode_forwards", 0))
    layers = int(raw.get("partial_decoder_layer_skip_events", 0))
    return {
        "decode_forwards": decode,
        "layer_skips": layers,
        "layers_per_decode_token": layers / decode if decode else None,
        "block_equivalents_per_decode_token": layers / (4.0 * decode) if decode else None,
        "raw": raw,
    }


def main():
    arms = ("candidate", "reskip_native_only", "reskip_gate_augmented")
    payloads = {}
    paths = {}
    for arm in arms:
        payloads[arm], paths[arm] = _result(arm)
    rows = []
    for arm in arms:
        scores = {task: _score(payloads[arm], task) for task in TASKS}
        rows.append(
            {
                "arm": arm,
                **scores,
                "macro_average": float(np.mean(list(scores.values()))),
            }
        )
    full_macro = rows[0]["macro_average"]
    for row in rows:
        row["delta_vs_full_pp"] = row["macro_average"] - full_macro
    result = {
        "screen_only": True,
        "limit_per_task": 256,
        "systems": rows,
        "skip": {
            "reskip_native_only": _skip(ROOT / "vlm_screen_n256/reskip_native_only/skip_stats.json"),
            "reskip_gate_augmented": _skip(ROOT / "vlm_screen_n256/reskip_gate_augmented/skip_stats.json"),
        },
        "result_jsons": paths,
    }
    output = ROOT / "vlm_screen_n256"
    (output / "VLM_POLICY_SCREEN_SUMMARY.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    fields = ("arm", *TASKS, "macro_average", "delta_vs_full_pp")
    with (output / "VLM_POLICY_SCREEN_TABLE.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
