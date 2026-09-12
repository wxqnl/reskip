"""Build the canonical full-six-task comparison after RSM-ReSkip finishes."""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data/Minko/experiments/attnres_residual_strength_gate_2b_20260831")
PAPER = Path("/data/Minko/experiments/reskip_paper_vlm_completion_20260824/full6")
TASKS = (
    "ai2d",
    "mmbench_en_dev_static",
    "mmmu_val",
    "mmstar",
    "ocrbench",
    "realworldqa",
)
DISPLAY = {
    "ai2d": "AI2D",
    "mmbench_en_dev_static": "MMBench",
    "mmmu_val": "MMMU",
    "mmstar": "MMStar",
    "ocrbench": "OCRBench",
    "realworldqa": "RealWorldQA",
}
SCORE_KEYS = {
    "ai2d": ("exact_match,flexible-extract", 100.0),
    "mmbench_en_dev_static": ("gpt_eval_score,none", 1.0),
    "mmmu_val": ("mmmu_acc,none", 100.0),
    "mmstar": ("average,none", 100.0),
    "ocrbench": ("ocrbench_accuracy,none", 100.0),
    "realworldqa": ("exact_match,none", 100.0),
}


def _latest(pattern: str) -> Path:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return Path(paths[-1])


def _row(system: str, result_path: Path) -> dict:
    with result_path.open() as file:
        results = json.load(file)["results"]
    scores = {}
    for task in TASKS:
        key, scale = SCORE_KEYS[task]
        scores[DISPLAY[task]] = float(results[task][key]) * scale
    return {
        "system": system,
        **scores,
        "macro_average": float(np.mean(list(scores.values()))),
        "result_json": str(result_path),
    }


def _skip_stats(path: Path) -> dict:
    with path.open() as file:
        raw = json.load(file)
    decode = int(raw["decode_forwards"])
    layers = int(raw["partial_decoder_layer_skip_events"])
    return {
        "decode_forwards": decode,
        "decoder_layer_skips": layers,
        "layers_per_decode_token": layers / decode,
        "block_equivalents_per_decode_token": layers / (4.0 * decode),
        "path": str(path),
    }


def main() -> None:
    result_paths = {
        "Base (paper)": _latest(str(PAPER / "base/models__*/*_results.json")),
        "Full AttnRes (paper)": _latest(
            str(PAPER / "fullar/models__*/*_results.json")
        ),
        "ReSkip (paper)": _latest(
            str(PAPER / "reskip/models__*/*_results.json")
        ),
        "Full RSM-AttnRes": _latest(
            str(ROOT / "full6/candidate/models__*/*_results.json")
        ),
        "RSM-ReSkip": _latest(
            str(ROOT / "full6/reskip_conservative/models__*/*_results.json")
        ),
    }
    rows = [_row(system, path) for system, path in result_paths.items()]
    full_rsm = next(row for row in rows if row["system"] == "Full RSM-AttnRes")
    paper_attnres = next(row for row in rows if row["system"] == "Full AttnRes (paper)")
    for row in rows:
        row["delta_vs_full_rsm_pp"] = (
            row["macro_average"] - full_rsm["macro_average"]
        )
    result = {
        "formal_full_six_task_evaluation": True,
        "tasks": [DISPLAY[task] for task in TASKS],
        "systems": rows,
        "full_rsm_minus_paper_attnres_pp": (
            full_rsm["macro_average"] - paper_attnres["macro_average"]
        ),
        "rsm_reskip": _skip_stats(
            ROOT / "full6/reskip_conservative/skip_stats.json"
        ),
        "paper_reskip": _skip_stats(PAPER / "reskip/skip_stats.json"),
        "policy_path": str(
            ROOT / "policy_screen_seed259123/conservative_gate_policy.json"
        ),
        "failed_aggressive_policy": {
            "macro_average": 64.35039132050339,
            "delta_vs_full_rsm_pp": -3.6198302455921834,
            "block_equivalents_per_decode_token": 1.8048528058877644,
            "status": "archived_negative_result",
        },
        "runtime_note": (
            "This accuracy run uses the exact token-level host selector. "
            "Its wall time is not a speed benchmark."
        ),
    }
    output = ROOT / "full6"
    with (output / "FULL6_FINAL_TABLE.json").open("w") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    fields = (
        "system",
        *[DISPLAY[task] for task in TASKS],
        "macro_average",
        "delta_vs_full_rsm_pp",
        "result_json",
    )
    with (output / "FULL6_FINAL_TABLE.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
