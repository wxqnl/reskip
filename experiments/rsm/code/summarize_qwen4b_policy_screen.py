"""Apply the frozen Qwen3-VL-4B policy-screen continuation rule."""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path


ROOT = Path("/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831")
SCREEN = ROOT / "reskip/qwen3vl_4b_seed0_screen"
POLICY = ROOT / "reskip/qwen3vl_4b_seed0_policy_seed259123"
TASKS = (
    "ai2d",
    "mmbench_en_dev_static",
    "mmmu_val",
    "mmstar",
    "ocrbench",
    "realworldqa",
)
SCORE_KEYS = {
    "ai2d": ("exact_match,flexible-extract", 100.0),
    "mmbench_en_dev_static": ("gpt_eval_score,none", 1.0),
    "mmmu_val": ("mmmu_acc,none", 100.0),
    "mmstar": ("average,none", 100.0),
    "ocrbench": ("ocrbench_accuracy,none", 100.0),
    "realworldqa": ("exact_match,none", 100.0),
}


def latest(variant: str) -> Path:
    matches = sorted(glob.glob(str(SCREEN / variant / "models__*/*_results.json")))
    if not matches:
        raise FileNotFoundError(variant)
    return Path(matches[-1])


def score(variant: str) -> dict:
    path = latest(variant)
    results = json.loads(path.read_text())["results"]
    task_scores = {
        task: float(results[task][key]) * scale
        for task, (key, scale) in SCORE_KEYS.items()
    }
    result = {
        "variant": variant,
        "task_scores": task_scores,
        "macro_average": statistics.fmean(task_scores.values()),
        "result_json": str(path),
    }
    if variant != "full":
        stats_path = SCREEN / variant / "skip_stats.json"
        stats = json.loads(stats_path.read_text())
        decode = int(stats["decode_forwards"])
        layers = int(stats["partial_decoder_layer_skip_events"])
        result.update(
            {
                "decode_forwards": decode,
                "decoder_layer_skip_events": layers,
                "block_equivalents_per_decode_token": (
                    layers / (4.0 * decode) if decode else 0.0
                ),
                "skip_stats_json": str(stats_path),
                "policy_json": str(POLICY / f"{variant}_policy.json"),
            }
        )
    return result


def main() -> None:
    rows = [score(name) for name in ("full", "native_only", "gate_augmented")]
    full = rows[0]["macro_average"]
    for row in rows:
        row["delta_vs_full_pp"] = row["macro_average"] - full
    candidates = [
        row for row in rows[1:]
        if row["delta_vs_full_pp"] >= -0.30
        and row["block_equivalents_per_decode_token"] > 0.0
    ]
    selected = (
        max(
            candidates,
            key=lambda row: (
                row["block_equivalents_per_decode_token"],
                row["macro_average"],
            ),
        )
        if candidates
        else None
    )
    payload = {
        "status": "continue_to_formal_full6" if selected else "stop_no_safe_policy",
        "quality_floor_delta_pp": -0.30,
        "rows": rows,
        "selected_variant": selected["variant"] if selected else None,
        "selected_policy_json": selected["policy_json"] if selected else None,
        "selection_uses_only_screen_data": True,
        "formal_full6_is_not_used_for_policy_selection": True,
    }
    output = SCREEN / "POLICY_SCREEN_SELECTION.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
