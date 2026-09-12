"""Build paper-facing VLM quality tables from preserved official artifacts."""
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


def latest(pattern: str | Path) -> Path | None:
    matches = sorted(glob.glob(str(pattern)))
    return Path(matches[-1]) if matches else None


def make_row(
    family: str,
    system: str,
    path: Path | None,
    *,
    seed: int | None = None,
    provenance: str,
) -> dict | None:
    if path is None or not path.is_file():
        return None
    results = json.loads(path.read_text())["results"]
    scores = {}
    for task in TASKS:
        key, scale = SCORE_KEYS[task]
        scores[DISPLAY[task]] = float(results[task][key]) * scale
    return {
        "family": family,
        "system": system,
        "seed": seed,
        **scores,
        "macro_average": statistics.fmean(scores.values()),
        "provenance": provenance,
        "result_json": str(path),
    }


def add(rows: list[dict], value: dict | None) -> None:
    if value is not None:
        rows.append(value)


def main() -> None:
    rows: list[dict] = []
    paper2 = ROOT / "experiments/reskip_paper_vlm_completion_20260824/full6"
    rsm2 = ROOT / "experiments/attnres_residual_strength_gate_2b_20260831/full6"
    archive = (
        ROOT
        / "experiments/_archive/reskip_precanonical_20260824"
    )

    add(rows, make_row("Qwen3-VL-2B", "Base", latest(paper2 / "base/models__*/*_results.json"), provenance="canonical paper completion"))
    add(rows, make_row("Qwen3-VL-2B", "Full AttnRes", latest(paper2 / "fullar/models__*/*_results.json"), seed=0, provenance="canonical paper completion"))
    for seed in (1, 2):
        add(rows, make_row("Qwen3-VL-2B", "Full AttnRes", latest(EXPERIMENT / f"eval/controls/qwen3vl_2b_full_attnres_seed{seed}/models__*/*_results.json"), seed=seed, provenance="current matched-seed control"))
    add(rows, make_row("Qwen3-VL-2B", "Full RSM-AttnRes", latest(rsm2 / "candidate/models__*/*_results.json"), seed=0, provenance="canonical accepted RSM seed 0"))
    for seed in (1, 2):
        add(rows, make_row("Qwen3-VL-2B", "Full RSM-AttnRes", latest(EXPERIMENT / f"eval/full_rsm/qwen3vl_2b_rsm_seed{seed}/models__*/*_results.json"), seed=seed, provenance="current formal run"))
    add(rows, make_row("Qwen3-VL-2B", "RSM-ReSkip", latest(rsm2 / "reskip_conservative/models__*/*_results.json"), seed=0, provenance="canonical device-policy quality run"))

    qwen4_archive = archive / "paper_vlm_exact_repro_20260819/accuracy"
    add(rows, make_row("Qwen3-VL-4B", "Base", latest(EXPERIMENT / "eval/controls/qwen3vl_4b_base/models__*/*_results.json"), provenance="current matched control"))
    add(rows, make_row("Qwen3-VL-4B", "Full AttnRes", latest(qwen4_archive / "4b_full_ar/models__*/*_results.json"), seed=0, provenance="preserved exact-protocol AttnRes control"))
    for seed in (1, 2):
        add(rows, make_row("Qwen3-VL-4B", "Full AttnRes", latest(EXPERIMENT / f"eval/controls/qwen3vl_4b_full_attnres_seed{seed}/models__*/*_results.json"), seed=seed, provenance="current matched-seed control"))
    for seed in range(3):
        add(rows, make_row("Qwen3-VL-4B", "Full RSM-AttnRes", latest(EXPERIMENT / f"eval/full_rsm/qwen3vl_4b_rsm_seed{seed}/models__*/*_results.json"), seed=seed, provenance="current formal run"))
    add(rows, make_row("Qwen3-VL-4B", "RSM-ReSkip aggressive (failed)", latest(EXPERIMENT / "eval/reskip/qwen3vl_4b_seed0/models__*/*_results.json"), seed=0, provenance="preserved failed formal policy; exceeded quality-loss gate"))
    add(rows, make_row("Qwen3-VL-4B", "RSM-ReSkip", latest(EXPERIMENT / "eval/reskip/qwen3vl_4b_seed2_capped/models__*/*_results.json"), seed=2, provenance="final capped-policy confirmatory run"))

    generic_archive = archive / "ar_retrofit_v2_20260814/evaluation"
    add(rows, make_row("SmolVLM2-2.2B", "Base", latest(generic_archive / "smolvlm2_transfer_screen_5k/Frozen_Base/vlm/models__*/*_results.json"), provenance="preserved matched six-task control"))
    for seed in range(3):
        add(rows, make_row("SmolVLM2-2.2B", "Full AttnRes", latest(EXPERIMENT / f"eval/controls/smolvlm2_2p2b_full_attnres_lr1e3_noent_5k_seed{seed}/models__*/*_results.json"), seed=seed, provenance="Amendment-009 accepted matched-seed control"))
        add(rows, make_row("SmolVLM2-2.2B", "Full RSM-AttnRes", latest(EXPERIMENT / f"eval/full_rsm_lr1e3_noent/smolvlm2_2p2b_rsm_lr1e3_noent_5k_seed{seed}/models__*/*_results.json"), seed=seed, provenance="Amendment-009 accepted three-seed RSM confirmation"))
    add(rows, make_row("InternVL3.5-2B", "Base", latest(generic_archive / "internvl35_transfer_screen_5k/Frozen_Base/vlm/models__*/*_results.json"), provenance="preserved matched six-task control"))
    add(rows, make_row("InternVL3.5-2B", "Full AttnRes", latest(EXPERIMENT / "eval/controls/internvl3p5_2b_full_attnres_seed0/models__*/*_results.json"), seed=0, provenance="current clean matched control"))
    add(rows, make_row("InternVL3.5-2B", "Full RSM-AttnRes", latest(EXPERIMENT / "eval/full_rsm/internvl3p5_2b_rsm_lr2e4_5k_seed0/models__*/*_results.json"), seed=0, provenance="current formal run"))

    summaries = []
    groups = sorted({(row["family"], row["system"]) for row in rows})
    for family, system in groups:
        members = [
            row for row in rows
            if row["family"] == family and row["system"] == system
        ]
        macros = [row["macro_average"] for row in members]
        summaries.append(
            {
                "family": family,
                "system": system,
                "n": len(members),
                "macro_average_mean": statistics.fmean(macros),
                "macro_average_sample_std": (
                    statistics.stdev(macros) if len(macros) > 1 else None
                ),
            }
        )

    paired_deltas = []
    for family in (
        "Qwen3-VL-2B",
        "Qwen3-VL-4B",
        "SmolVLM2-2.2B",
        "InternVL3.5-2B",
    ):
        full_by_seed = {
            row["seed"]: row
            for row in rows
            if row["family"] == family
            and row["system"] == "Full AttnRes"
            and row["seed"] is not None
        }
        rsm_by_seed = {
            row["seed"]: row
            for row in rows
            if row["family"] == family
            and row["system"] == "Full RSM-AttnRes"
            and row["seed"] is not None
        }
        for seed in sorted(full_by_seed.keys() & rsm_by_seed.keys()):
            paired_deltas.append(
                {
                    "family": family,
                    "seed": seed,
                    "full_attnres_macro": full_by_seed[seed]["macro_average"],
                    "full_rsm_attnres_macro": rsm_by_seed[seed]["macro_average"],
                    "rsm_minus_attnres_pp": (
                        rsm_by_seed[seed]["macro_average"]
                        - full_by_seed[seed]["macro_average"]
                    ),
                }
            )

    paired_summaries = []
    for family in sorted({row["family"] for row in paired_deltas}):
        deltas = [
            row["rsm_minus_attnres_pp"]
            for row in paired_deltas
            if row["family"] == family
        ]
        paired_summaries.append(
            {
                "family": family,
                "n_matched_seeds": len(deltas),
                "rsm_minus_attnres_pp_mean": statistics.fmean(deltas),
                "rsm_minus_attnres_pp_sample_std": (
                    statistics.stdev(deltas) if len(deltas) > 1 else None
                ),
            }
        )

    payload = {
        "formal_tasks": [DISPLAY[task] for task in TASKS],
        "rows": rows,
        "summaries": summaries,
        "matched_seed_deltas": paired_deltas,
        "matched_seed_summaries": paired_summaries,
        "cross_family_pooling": False,
        "caveat": (
            "Preserved controls are reused only when model, task set, and "
            "official scoring protocol match; each source path is retained."
        ),
    }
    reports = EXPERIMENT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "VLM_FINAL_QUALITY_TABLES.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    fields = (
        "family", "system", "seed",
        *[DISPLAY[task] for task in TASKS],
        "macro_average", "provenance", "result_json",
    )
    with (reports / "VLM_FINAL_QUALITY_ROWS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
