"""Summarize the paired six-task lmms-eval screen."""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np


ROOT = Path("/data/Minko/experiments/attnres_residual_strength_gate_2b_20260831")
TASKS = (
    "ai2d",
    "mmbench_en_dev_static",
    "mmmu_val",
    "mmstar",
    "ocrbench",
    "realworldqa",
)


def result_path(arm):
    matches = sorted(glob.glob(str(ROOT / "vlm_screen_n256" / arm / "models__*" / "*_results.json")))
    if not matches:
        raise FileNotFoundError(arm)
    return Path(matches[-1])


def score(payload, task):
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
    payloads = {
        arm: json.loads(result_path(arm).read_text())
        for arm in ("control", "candidate")
    }
    rows = []
    for task in TASKS:
        control = score(payloads["control"], task)
        candidate = score(payloads["candidate"], task)
        rows.append(
            {
                "task": task,
                "control": control,
                "candidate": candidate,
                "delta_pp": candidate - control,
            }
        )
    control_macro = float(np.mean([row["control"] for row in rows]))
    candidate_macro = float(np.mean([row["candidate"] for row in rows]))
    result = {
        "screen_only": True,
        "limit_per_task": 256,
        "paired_request_order": True,
        "control_result": str(result_path("control")),
        "candidate_result": str(result_path("candidate")),
        "tasks": rows,
        "macro": {
            "control": control_macro,
            "candidate": candidate_macro,
            "delta_pp": candidate_macro - control_macro,
            "tasks_improved": sum(row["delta_pp"] > 0 for row in rows),
            "tasks_tied": sum(row["delta_pp"] == 0 for row in rows),
            "tasks_degraded": sum(row["delta_pp"] < 0 for row in rows),
        },
        "interpretation": "This bounded screen decides whether a full six-task run is warranted; it is not a paper result.",
    }
    output = ROOT / "vlm_screen_n256"
    (output / "VLM_SCREEN_SUMMARY.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (output / "VLM_SCREEN_TABLE.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("task", "control", "candidate", "delta_pp"))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
