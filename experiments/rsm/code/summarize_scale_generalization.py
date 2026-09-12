"""Aggregate official six-task lmms-eval artifacts across VLM families."""
from __future__ import annotations

import csv
import glob
import json
import statistics
from pathlib import Path


ROOT = Path("/data/Minko")
EXPERIMENT = ROOT / "experiments/attnres_rsm_vlm_scale_generalization_20260831"
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


def latest(pattern: Path | str) -> Path:
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def row(system: str, family: str, seed: int | None, path: Path) -> dict:
    results = json.loads(path.read_text())["results"]
    scores = {}
    for task in TASKS:
        key, scale = SCORE_KEYS[task]
        scores[DISPLAY[task]] = float(results[task][key]) * scale
    return {
        "system": system,
        "family": family,
        "seed": seed,
        **scores,
        "macro_average": statistics.fmean(scores.values()),
        "result_json": str(path),
    }


def main() -> None:
    rows = []
    for seed in range(3):
        if seed == 0:
            path = latest(
                ROOT
                / "experiments/attnres_residual_strength_gate_2b_20260831/full6/candidate/models__*/*_results.json"
            )
        else:
            path = latest(
                EXPERIMENT
                / f"eval/full_rsm/qwen3vl_2b_rsm_seed{seed}/models__*/*_results.json"
            )
        rows.append(row("Full RSM-AttnRes", "Qwen3-VL-2B", seed, path))
    for seed in range(3):
        path = latest(
            EXPERIMENT
            / f"eval/full_rsm/qwen3vl_4b_rsm_seed{seed}/models__*/*_results.json"
        )
        rows.append(row("Full RSM-AttnRes", "Qwen3-VL-4B", seed, path))
    rows.append(
        row(
            "Full RSM-AttnRes",
            "SmolVLM2-2.2B",
            0,
            latest(
                EXPERIMENT
                / "eval/full_rsm/smolvlm2_2p2b_rsm_lr2e4_5k_seed0/models__*/*_results.json"
            ),
        )
    )
    rows.append(
        row(
            "Full RSM-AttnRes",
            "InternVL3.5-2B",
            0,
            latest(
                EXPERIMENT
                / "eval/full_rsm/internvl3p5_2b_rsm_lr2e4_5k_seed0/models__*/*_results.json"
            ),
        )
    )

    family_summaries = []
    for family in sorted({record["family"] for record in rows}):
        members = [record for record in rows if record["family"] == family]
        macros = [record["macro_average"] for record in members]
        family_summaries.append(
            {
                "family": family,
                "n_seeds": len(members),
                "macro_average_mean": statistics.fmean(macros),
                "macro_average_sample_std": (
                    statistics.stdev(macros) if len(macros) > 1 else None
                ),
            }
        )

    payload = {
        "formal_full_six_task_evaluation": True,
        "tasks": [DISPLAY[task] for task in TASKS],
        "rows": rows,
        "family_summaries": family_summaries,
        "note": (
            "Each row uses the official six-task protocol. Cross-family raw "
            "metrics are reported separately and are not pooled."
        ),
    }
    reports = EXPERIMENT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "FULL_RSM_SCALE_GENERALIZATION.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    fields = (
        "system",
        "family",
        "seed",
        *[DISPLAY[task] for task in TASKS],
        "macro_average",
        "result_json",
    )
    with (reports / "FULL_RSM_SCALE_GENERALIZATION.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
