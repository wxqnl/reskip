"""Apply the frozen formal quality gate to the final 4B seed-2 policy."""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path


ROOT = Path("/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831")
TASKS = ("ai2d", "mmbench_en_dev_static", "mmmu_val", "mmstar", "ocrbench", "realworldqa")
SCORE_KEYS = {
    "ai2d": ("exact_match,flexible-extract", 100.0),
    "mmbench_en_dev_static": ("gpt_eval_score,none", 1.0),
    "mmmu_val": ("mmmu_acc,none", 100.0),
    "mmstar": ("average,none", 100.0),
    "ocrbench": ("ocrbench_accuracy,none", 100.0),
    "realworldqa": ("exact_match,none", 100.0),
}


def latest(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def score(path: Path) -> tuple[dict, float]:
    data = json.loads(path.read_text())["results"]
    values = {
        task: float(data[task][key]) * scale
        for task, (key, scale) in SCORE_KEYS.items()
    }
    return values, statistics.fmean(values.values())


def main() -> None:
    full_path = latest(str(ROOT / "eval/full_rsm/qwen3vl_4b_rsm_seed2/models__*/*_results.json"))
    reskip_path = latest(str(ROOT / "eval/reskip/qwen3vl_4b_seed2_capped/models__*/*_results.json"))
    stats_path = ROOT / "eval/reskip/qwen3vl_4b_seed2_capped/skip_stats.json"
    full_tasks, full_macro = score(full_path)
    reskip_tasks, reskip_macro = score(reskip_path)
    stats = json.loads(stats_path.read_text())
    block_equiv = stats["avg_partial_decoder_layer_skips_per_decode_forward"] / 4.0
    delta = reskip_macro - full_macro
    passed = delta >= -0.30 and block_equiv > 0.0
    payload = {
        "status": "pass_runtime_allowed" if passed else "fail_runtime_forbidden",
        "full_rsm_seed2": {"macro": full_macro, "tasks": full_tasks, "result_json": str(full_path)},
        "rsm_reskip_seed2_capped": {"macro": reskip_macro, "tasks": reskip_tasks, "result_json": str(reskip_path)},
        "delta_vs_full_pp": delta,
        "block_equivalents_per_token": block_equiv,
        "quality_floor_delta_pp": -0.30,
        "policy_changed_after_formal_result": False,
    }
    output = ROOT / "eval/reskip/qwen3vl_4b_seed2_capped/FORMAL_QUALITY_GATE.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
