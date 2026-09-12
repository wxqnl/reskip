"""Create the formal six-task comparison against frozen paper baselines."""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data/Minko")
EXPERIMENT = ROOT / "experiments/attnres_residual_strength_gate_2b_20260831"
TASKS = (
    "ai2d",
    "mmbench_en_dev_static",
    "mmmu_val",
    "mmstar",
    "ocrbench",
    "realworldqa",
)
LABELS = {
    "ai2d": "AI2D",
    "mmbench_en_dev_static": "MMBench",
    "mmmu_val": "MMMU",
    "mmstar": "MMStar",
    "ocrbench": "OCRBench",
    "realworldqa": "RealWorldQA",
}


def _latest(pattern):
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def _score(payload, task):
    record = payload["results"][task]
    if task == "ai2d":
        return 100.0 * float(record["exact_match,flexible-extract"])
    if task == "mmbench_en_dev_static":
        return float(record["gpt_eval_score,none"])
    if task == "mmmu_val":
        return 100.0 * float(record["mmmu_acc,none"])
    if task == "mmstar":
        return 100.0 * float(record["average,none"])
    if task == "ocrbench":
        return 100.0 * float(record["ocrbench_accuracy,none"])
    if task == "realworldqa":
        return 100.0 * float(record["exact_match,none"])
    raise KeyError(task)


def main():
    paths = {
        "Base (paper)": _latest(
            ROOT / "experiments/reskip_paper_vlm_completion_20260824/full6/base/models__*/*_results.json"
        ),
        "Full AttnRes (paper)": _latest(
            ROOT / "experiments/reskip_paper_vlm_completion_20260824/full6/fullar/models__*/*_results.json"
        ),
        "Full RSG-AttnRes": _latest(
            EXPERIMENT / "full6/candidate/models__*/*_results.json"
        ),
    }
    systems = []
    for name, path in paths.items():
        payload = json.loads(path.read_text())
        scores = {LABELS[task]: _score(payload, task) for task in TASKS}
        systems.append(
            {
                "system": name,
                **scores,
                "macro_average": float(np.mean(list(scores.values()))),
                "result_json": str(path),
            }
        )
    paper_attnres = next(
        row["macro_average"] for row in systems if row["system"] == "Full AttnRes (paper)"
    )
    candidate = next(
        row["macro_average"] for row in systems if row["system"] == "Full RSG-AttnRes"
    )
    result = {
        "tasks": [LABELS[task] for task in TASKS],
        "systems": systems,
        "candidate_minus_paper_attnres_pp": candidate - paper_attnres,
        "candidate_slightly_better_than_paper_attnres": bool(candidate > paper_attnres),
        "note": "All rows use the same official six-task lmms-eval protocol; the new candidate is Full-path only and contains no ReSkip execution.",
    }
    output = EXPERIMENT / "full6"
    (output / "FULL6_COMPARISON.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    columns = (
        "system", "MMBench", "MMMU", "MMStar", "AI2D", "OCRBench",
        "RealWorldQA", "macro_average", "result_json",
    )
    with (output / "FULL6_COMPARISON.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(systems)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
