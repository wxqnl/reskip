"""Aggregate the frozen generic-VLM learning-rate and specialization experiment."""
from __future__ import annotations

import csv
import glob
import json
import statistics
from pathlib import Path


ROOT = Path("/data/Minko")
EXP = ROOT / "experiments/attnres_rsm_vlm_scale_generalization_20260831"
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


def latest(pattern: Path) -> Path:
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def scores_from_result(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text())["results"]
    scores = {}
    for task in TASKS:
        key, scale = SCORE_KEYS[task]
        scores[DISPLAY[task]] = float(raw[task][key]) * scale
    return scores


def training_diagnostics(path: Path | None) -> dict:
    if path is None:
        return {
            "route_entropy_ema": None,
            "gate_mean": None,
            "gate_min": None,
            "gate_max": None,
            "identity_max_abs_logit_delta": None,
            "skip_training_forwards": None,
        }
    summary = json.loads(path.read_text())
    gates = summary.get("last_batch_residual_strength_factors", {})
    count = sum(item["count"] for item in gates.values())
    return {
        "route_entropy_ema": summary.get("route_entropy_ema"),
        "gate_mean": (
            sum(item["count"] * item["mean"] for item in gates.values()) / count
            if count
            else 1.0
        ),
        "gate_min": min((item["min"] for item in gates.values()), default=1.0),
        "gate_max": max((item["max"] for item in gates.values()), default=1.0),
        "identity_max_abs_logit_delta": summary.get("identity_max_abs_logit_delta"),
        "skip_training_forwards": summary.get("skip_training_forwards"),
    }


def evaluated_row(
    family: str,
    condition: str,
    result_path: Path,
    train_summary: Path | None,
) -> dict:
    scores = scores_from_result(result_path)
    return {
        "family": family,
        "condition": condition,
        **scores,
        "macro_average": statistics.fmean(scores.values()),
        **training_diagnostics(train_summary),
        "result_json": str(result_path),
        "train_summary": str(train_summary) if train_summary else None,
    }


def main() -> None:
    prior = json.loads((EXP / "reports/VLM_FINAL_QUALITY_TABLES.json").read_text())
    prior_by_key = {
        (row["family"], row["system"]): row
        for row in prior["rows"]
        if row["family"] in {"SmolVLM2-2.2B", "InternVL3.5-2B"}
    }
    specs = {
        "SmolVLM2-2.2B": {
            "prefix": "smolvlm2_2p2b",
            "old_rsm_train": EXP / "train/smolvlm2_2p2b_rsm_lr2e4_5k_seed0/train_summary.json",
            "old_full_train": EXP / "train/controls/smolvlm2_2p2b_full_attnres_seed0/train_summary.json",
        },
        "InternVL3.5-2B": {
            "prefix": "internvl3p5_2b",
            "old_rsm_train": EXP / "train/internvl3p5_2b_rsm_lr2e4_5k_seed0/train_summary.json",
            "old_full_train": EXP / "train/controls/internvl3p5_2b_full_attnres_seed0/train_summary.json",
        },
    }

    rows = []
    decisions = []
    for family, spec in specs.items():
        for system, condition, train_path in (
            ("Base", "Frozen Base", None),
            ("Full AttnRes", "Full AttnRes, lr=2e-4, entropy=0.02", spec["old_full_train"]),
            ("Full RSM-AttnRes", "RSM, lr=2e-4, entropy=0.02", spec["old_rsm_train"]),
        ):
            source = prior_by_key[(family, system)]
            rows.append(
                evaluated_row(
                    family,
                    condition,
                    Path(source["result_json"]),
                    train_path,
                )
            )

        prefix = spec["prefix"]
        new_specs = (
            (
                "Full AttnRes, lr=1e-3, entropy=0.02",
                EXP / f"eval/controls/{prefix}_full_attnres_lr1e3_5k_seed0/models__*/*_results.json",
                EXP / f"train/controls/{prefix}_full_attnres_lr1e3_5k_seed0/train_summary.json",
            ),
            (
                "RSM, lr=1e-3, entropy=0.02",
                EXP / f"eval/full_rsm_lr1e3/{prefix}_rsm_lr1e3_5k_seed0/models__*/*_results.json",
                EXP / f"train/lr1e3/{prefix}_rsm_lr1e3_5k_seed0/train_summary.json",
            ),
            (
                "Full AttnRes, lr=1e-3, entropy=0",
                EXP / f"eval/controls/{prefix}_full_attnres_lr1e3_noent_5k_seed0/models__*/*_results.json",
                EXP / f"train/controls/{prefix}_full_attnres_lr1e3_noent_5k_seed0/train_summary.json",
            ),
            (
                "RSM, lr=1e-3, entropy=0",
                EXP / f"eval/full_rsm_lr1e3_noent/{prefix}_rsm_lr1e3_noent_5k_seed0/models__*/*_results.json",
                EXP / f"train/lr1e3_noent/{prefix}_rsm_lr1e3_noent_5k_seed0/train_summary.json",
            ),
        )
        for condition, result_pattern, train_path in new_specs:
            rows.append(
                evaluated_row(family, condition, latest(result_pattern), train_path)
            )

        family_rows = {
            row["condition"]: row for row in rows if row["family"] == family
        }
        base = family_rows["Frozen Base"]["macro_average"]
        full_reg = family_rows["Full AttnRes, lr=1e-3, entropy=0.02"]["macro_average"]
        rsm_reg = family_rows["RSM, lr=1e-3, entropy=0.02"]["macro_average"]
        full_noent = family_rows["Full AttnRes, lr=1e-3, entropy=0"]["macro_average"]
        rsm_noent = family_rows["RSM, lr=1e-3, entropy=0"]["macro_average"]
        regularized_pass = rsm_reg > base and rsm_reg - full_reg >= 0.15
        noent_pass = (
            rsm_noent > base
            and rsm_noent - full_noent >= 0.15
            and rsm_noent > rsm_reg
        )
        selected = (
            "RSM, lr=1e-3, entropy=0"
            if noent_pass
            else "RSM, lr=1e-3, entropy=0.02"
            if regularized_pass
            else None
        )
        decisions.append(
            {
                "family": family,
                "base_macro": base,
                "regularized_rsm_macro": rsm_reg,
                "regularized_rsm_minus_full_pp": rsm_reg - full_reg,
                "regularized_rsm_minus_base_pp": rsm_reg - base,
                "regularized_passes_amendment_006": regularized_pass,
                "noent_rsm_macro": rsm_noent,
                "noent_rsm_minus_full_pp": rsm_noent - full_noent,
                "noent_rsm_minus_base_pp": rsm_noent - base,
                "noent_rsm_minus_regularized_rsm_pp": rsm_noent - rsm_reg,
                "noent_passes_amendment_007": noent_pass,
                "selected_for_new_seed_confirmation": selected,
            }
        )

    payload = {
        "status": "completed",
        "rows": rows,
        "decisions": decisions,
        "protocols": [
            str(EXP / "protocol/PROTOCOL_AMENDMENT_006.json"),
            str(EXP / "protocol/PROTOCOL_AMENDMENT_007.json"),
        ],
        "material_passport": {
            "models": ["SmolVLM2-2.2B-Instruct", "InternVL3_5-2B-HF"],
            "training_data": "/data/Minko/datasets, v3 stream",
            "evaluation": "official six-task local lmms-eval",
            "hardware": "node42 NVIDIA H100",
            "software": str(EXP / "code"),
        },
    }
    reports = EXP / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "GENERIC_VLM_IMPROVEMENT_RESULTS.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    fields = [
        "family",
        "condition",
        *DISPLAY.values(),
        "macro_average",
        "route_entropy_ema",
        "gate_mean",
        "gate_min",
        "gate_max",
        "identity_max_abs_logit_delta",
        "skip_training_forwards",
        "result_json",
        "train_summary",
    ]
    with (reports / "GENERIC_VLM_IMPROVEMENT_ROWS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Generic VLM AttnRes improvement experiment",
        "",
        "| Family | Condition | Macro | Route entropy | Gate mean [min, max] |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        entropy = "--" if row["route_entropy_ema"] is None else f"{row['route_entropy_ema']:.4f}"
        gate = (
            "--"
            if row["gate_mean"] is None
            else f"{row['gate_mean']:.3f} [{row['gate_min']:.3f}, {row['gate_max']:.3f}]"
        )
        lines.append(
            f"| {row['family']} | {row['condition']} | {row['macro_average']:.3f} | {entropy} | {gate} |"
        )
    lines.extend(["", "## Frozen decisions", ""])
    for item in decisions:
        lines.append(
            f"- {item['family']}: selected `{item['selected_for_new_seed_confirmation']}`; "
            f"regularized RSM−Full={item['regularized_rsm_minus_full_pp']:+.3f} pp, "
            f"no-entropy RSM−Full={item['noent_rsm_minus_full_pp']:+.3f} pp."
        )
    lines.extend(
        [
            "",
            "## Material Passport",
            "",
            "- Origin Skill: academic-research-suite / experiment-agent",
            "- Origin Mode: run + validate",
            "- Origin Date: 2026-09-01",
            "- Verification Status: VERIFIED after full artifact completion",
            "- Models: SmolVLM2-2.2B-Instruct; InternVL3_5-2B-HF",
            "- Data: local v3 adaptation stream and six formal lmms-eval tasks",
            "- Hardware: node42 NVIDIA H100",
        ]
    )
    (reports / "GENERIC_VLM_IMPROVEMENT_RESULTS.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(json.dumps({"decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
