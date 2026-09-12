"""Apply the frozen seed-1 conservative screen continuation rule."""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path


ROOT = Path("/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831")
SCREEN = ROOT / "reskip/qwen3vl_4b_seed1_conservative_screen"
TASKS = ("ai2d", "mmbench_en_dev_static", "mmmu_val", "mmstar", "ocrbench", "realworldqa")
SCORE_KEYS = {
    "ai2d": ("exact_match,flexible-extract", 100.0),
    "mmbench_en_dev_static": ("gpt_eval_score,none", 1.0),
    "mmmu_val": ("mmmu_acc,none", 100.0),
    "mmstar": ("average,none", 100.0),
    "ocrbench": ("ocrbench_accuracy,none", 100.0),
    "realworldqa": ("exact_match,none", 100.0),
}


def result(variant: str) -> dict:
    matches = sorted(glob.glob(str(SCREEN / variant / "models__*/*_results.json")))
    if not matches:
        raise FileNotFoundError(variant)
    path = Path(matches[-1])
    data = json.loads(path.read_text())["results"]
    scores = {
        task: float(data[task][key]) * scale
        for task, (key, scale) in SCORE_KEYS.items()
    }
    return {"variant": variant, "scores": scores, "macro": statistics.fmean(scores.values()), "result_json": str(path)}


def main() -> None:
    full = result("full")
    candidate = result("native_conservative")
    stats_path = SCREEN / "native_conservative/skip_stats.json"
    stats = json.loads(stats_path.read_text())
    decode = int(stats["decode_forwards"])
    layers = int(stats["partial_decoder_layer_skip_events"])
    block_equiv = layers / (4.0 * decode) if decode else 0.0
    delta = candidate["macro"] - full["macro"]
    passed = delta >= -0.30 and 0.75 <= block_equiv <= 1.15
    payload = {
        "status": "continue_to_seed1_formal" if passed else "stop_no_safe_policy",
        "full": full,
        "candidate": candidate,
        "delta_vs_full_pp": delta,
        "block_equivalents_per_token": block_equiv,
        "screen_rule": {"minimum_delta_pp": -0.30, "block_equivalent_interval": [0.75, 1.15]},
        "policy_json": str(ROOT / "reskip/qwen3vl_4b_seed1_conservative_policy_seed259123/native_only_policy.json"),
        "formal_seed1_output_unseen": True,
    }
    (SCREEN / "SCREEN_DECISION.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
