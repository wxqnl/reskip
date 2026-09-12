"""Build the final, protocol-gated audit for generic-VLM AttnRes repair runs."""
from __future__ import annotations

import csv
import glob
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP = Path("/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831")
REPORTS = EXP / "reports/diagnostics"
FIGURES = EXP / "figures"
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
ENTROPY_CEILING = {
    "SmolVLM2-2.2B": statistics.fmean(math.log(index) for index in range(1, 7)),
    "InternVL3.5-2B": statistics.fmean(math.log(index) for index in range(1, 8)),
    "Granite Vision 4.1-4B": statistics.fmean(
        math.log(index) for index in range(1, 11)
    ),
}
SCORE_KEYS = {
    "ai2d": ("exact_match,flexible-extract", 100.0),
    "mmbench_en_dev_static": ("gpt_eval_score,none", 1.0),
    "mmmu_val": ("mmmu_acc,none", 100.0),
    "mmstar": ("average,none", 100.0),
    "ocrbench": ("ocrbench_accuracy,none", 100.0),
    "realworldqa": ("exact_match,none", 100.0),
}


def latest(pattern: str | Path) -> Path:
    matches = sorted(glob.glob(str(pattern)))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def scores(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text())["results"]
    return {
        DISPLAY[task]: float(raw[task][key]) * scale
        for task, (key, scale) in SCORE_KEYS.items()
    }


def train_diagnostics(path: Path | None) -> dict:
    if path is None:
        return {
            "route_entropy_ema": None,
            "gate_weight_rms": None,
            "gate_weight_max_abs": None,
            "block_gate_means": None,
            "identity_max_abs_logit_delta": None,
            "skip_training_forwards": None,
        }
    raw = json.loads(path.read_text())
    gate = raw.get("residual_strength_gate_parameters") or {}
    blocks = raw.get("last_batch_residual_strength_factors") or {}
    return {
        "route_entropy_ema": raw.get("route_entropy_ema"),
        "gate_weight_rms": gate.get("weight_rms"),
        "gate_weight_max_abs": gate.get("weight_max_abs"),
        "block_gate_means": {
            key: value["mean"] for key, value in sorted(blocks.items())
        },
        "identity_max_abs_logit_delta": raw.get("identity_max_abs_logit_delta"),
        "skip_training_forwards": raw.get("skip_training_forwards"),
    }


def row(
    family: str,
    system: str,
    recipe: str,
    seed: int | None,
    result: Path,
    summary: Path | None,
    evidence_status: str,
) -> dict:
    task_scores = scores(result)
    return {
        "family": family,
        "system": system,
        "recipe": recipe,
        "seed": seed,
        "evidence_status": evidence_status,
        **task_scores,
        "macro_average": statistics.fmean(task_scores.values()),
        **train_diagnostics(summary),
        "result_json": str(result),
        "train_summary": str(summary) if summary else None,
    }


def prior_result(family: str, system: str) -> Path:
    raw = json.loads((EXP / "reports/VLM_FINAL_QUALITY_TABLES.json").read_text())
    for item in raw["rows"]:
        if item["family"] == family and item["system"] == system:
            return Path(item["result_json"])
    raise KeyError((family, system))


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    common = {
        "SmolVLM2-2.2B": "smolvlm2_2p2b",
        "InternVL3.5-2B": "internvl3p5_2b",
    }
    for family, prefix in common.items():
        rows.append(
            row(
                family,
                "Base",
                "Frozen pretrained",
                None,
                prior_result(family, "Base"),
                None,
                "frozen baseline",
            )
        )
        for system, system_slug, eval_dir, train_dir in (
            (
                "Full AttnRes",
                "full_attnres",
                "controls",
                "controls",
            ),
            (
                "Full RSM-AttnRes",
                "rsm",
                "full_rsm",
                "",
            ),
        ):
            run = (
                f"{prefix}_full_attnres_seed0"
                if system == "Full AttnRes"
                else f"{prefix}_rsm_lr2e4_5k_seed0"
            )
            summary = (
                EXP / f"train/{train_dir}/{run}/train_summary.json"
                if train_dir
                else EXP / f"train/{run}/train_summary.json"
            )
            rows.append(
                row(
                    family,
                    system,
                    "lr=2e-4, entropy=0.02",
                    0,
                    latest(EXP / f"eval/{eval_dir}/{run}/models__*/*_results.json"),
                    summary,
                    "historical control",
                )
            )

        for system, run_system, eval_dir, train_dir in (
            ("Full AttnRes", "full_attnres", "controls", "controls"),
            ("Full RSM-AttnRes", "rsm", "full_rsm_lr1e3", "lr1e3"),
        ):
            run = f"{prefix}_{run_system}_lr1e3_5k_seed0"
            rows.append(
                row(
                    family,
                    system,
                    "lr=1e-3, entropy=0.02",
                    0,
                    latest(EXP / f"eval/{eval_dir}/{run}/models__*/*_results.json"),
                    EXP / f"train/{train_dir}/{run}/train_summary.json",
                    "rejected seed-0 diagnostic",
                )
            )

        noent_seeds = (0, 1, 2) if family == "SmolVLM2-2.2B" else (0,)
        for seed in noent_seeds:
            for system, run_system, eval_dir, train_dir in (
                ("Full AttnRes", "full_attnres", "controls", "controls"),
                (
                    "Full RSM-AttnRes",
                    "rsm",
                    "full_rsm_lr1e3_noent",
                    "lr1e3_noent",
                ),
            ):
                run = f"{prefix}_{run_system}_lr1e3_noent_5k_seed{seed}"
                rows.append(
                    row(
                        family,
                        system,
                        "lr=1e-3, entropy=0",
                        seed,
                        latest(
                            EXP / f"eval/{eval_dir}/{run}/models__*/*_results.json"
                        ),
                        EXP / f"train/{train_dir}/{run}/train_summary.json",
                        (
                            "three-seed confirmation"
                            if family == "SmolVLM2-2.2B"
                            else "rejected seed-0 diagnostic"
                        ),
                    )
                )

    family = "InternVL3.5-2B"
    prefix = "internvl3p5_2b"
    for system, run_system, eval_dir, train_dir in (
        ("Full AttnRes", "full_attnres", "controls", "controls"),
        ("Full RSM-AttnRes", "rsm", "full_rsm_lr5e4", "lr5e4"),
    ):
        run = f"{prefix}_{run_system}_lr5e4_5k_seed0"
        rows.append(
            row(
                family,
                system,
                "lr=5e-4, entropy=0.02",
                0,
                latest(EXP / f"eval/{eval_dir}/{run}/models__*/*_results.json"),
                EXP / f"train/{train_dir}/{run}/train_summary.json",
                "bounded midpoint seed-0 test",
            )
        )

    family = "Granite Vision 4.1-4B"
    rows.append(
        row(
            family,
            "Base",
            "Frozen pretrained",
            None,
            latest(
                EXP
                / "eval/controls/granite_vision_4p1_4b_base/models__*/*_results.json"
            ),
            None,
            "frozen current-model baseline",
        )
    )
    for system, run_system, eval_dir, train_dir in (
        ("Full AttnRes", "full_attnres", "controls", "controls"),
        (
            "Full RSM-AttnRes",
            "rsm",
            "full_rsm_granite4_lr1e3_noent",
            "granite4_lr1e3_noent",
        ),
    ):
        run = f"granite_vision_4p1_4b_{run_system}_lr1e3_noent_5k_seed0"
        rows.append(
            row(
                family,
                system,
                "lr=1e-3, entropy=0",
                0,
                latest(EXP / f"eval/{eval_dir}/{run}/models__*/*_results.json"),
                EXP / f"train/{train_dir}/{run}/train_summary.json",
                "pre-registered current-model seed-0 transfer",
            )
        )
    return rows


def mean_and_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def select(rows: list[dict], family: str, recipe: str, system: str) -> list[dict]:
    return [
        item
        for item in rows
        if item["family"] == family
        and item["recipe"] == recipe
        and item["system"] == system
    ]


def decisions(rows: list[dict]) -> dict:
    smol_full = sorted(
        select(rows, "SmolVLM2-2.2B", "lr=1e-3, entropy=0", "Full AttnRes"),
        key=lambda item: item["seed"],
    )
    smol_rsm = sorted(
        select(
            rows,
            "SmolVLM2-2.2B",
            "lr=1e-3, entropy=0",
            "Full RSM-AttnRes",
        ),
        key=lambda item: item["seed"],
    )
    smol_base = select(rows, "SmolVLM2-2.2B", "Frozen pretrained", "Base")[0][
        "macro_average"
    ]
    full_values = [item["macro_average"] for item in smol_full]
    rsm_values = [item["macro_average"] for item in smol_rsm]
    paired = [rsm - full for rsm, full in zip(rsm_values, full_values)]
    full_mean, full_sd = mean_and_sd(full_values)
    rsm_mean, rsm_sd = mean_and_sd(rsm_values)
    paired_mean, paired_sd = mean_and_sd(paired)
    smol_pass = (
        paired_mean >= 0.15
        and rsm_mean > smol_base
        and sum(value > 0 for value in paired) >= 2
    )

    intern_full = select(
        rows, "InternVL3.5-2B", "lr=5e-4, entropy=0.02", "Full AttnRes"
    )[0]["macro_average"]
    intern_rsm = select(
        rows,
        "InternVL3.5-2B",
        "lr=5e-4, entropy=0.02",
        "Full RSM-AttnRes",
    )[0]["macro_average"]
    intern_base = select(rows, "InternVL3.5-2B", "Frozen pretrained", "Base")[0][
        "macro_average"
    ]
    intern_old_full = select(
        rows, "InternVL3.5-2B", "lr=2e-4, entropy=0.02", "Full AttnRes"
    )[0]["macro_average"]
    intern_pass = (
        intern_rsm - intern_full >= 0.15
        and intern_rsm > intern_base
        and intern_rsm > intern_old_full
    )
    granite_base = select(
        rows, "Granite Vision 4.1-4B", "Frozen pretrained", "Base"
    )[0]["macro_average"]
    granite_full = select(
        rows,
        "Granite Vision 4.1-4B",
        "lr=1e-3, entropy=0",
        "Full AttnRes",
    )[0]["macro_average"]
    granite_rsm = select(
        rows,
        "Granite Vision 4.1-4B",
        "lr=1e-3, entropy=0",
        "Full RSM-AttnRes",
    )[0]["macro_average"]
    granite_pass = (
        granite_rsm - granite_full >= 0.15 and granite_rsm > granite_base
    )
    return {
        "SmolVLM2-2.2B": {
            "protocol": "PROTOCOL_AMENDMENT_009",
            "base_macro": smol_base,
            "full_macro_mean": full_mean,
            "full_macro_sd": full_sd,
            "rsm_macro_mean": rsm_mean,
            "rsm_macro_sd": rsm_sd,
            "paired_rsm_minus_full_pp": paired,
            "paired_delta_mean_pp": paired_mean,
            "paired_delta_sd_pp": paired_sd,
            "positive_paired_seeds": sum(value > 0 for value in paired),
            "passes_confirmation_gate": smol_pass,
            "decision": "accept paper replacement" if smol_pass else "retain as exploratory",
        },
        "InternVL3.5-2B": {
            "protocol": "PROTOCOL_AMENDMENT_008",
            "base_macro": intern_base,
            "old_full_macro": intern_old_full,
            "midpoint_full_macro": intern_full,
            "midpoint_rsm_macro": intern_rsm,
            "midpoint_rsm_minus_full_pp": intern_rsm - intern_full,
            "midpoint_rsm_minus_base_pp": intern_rsm - intern_base,
            "passes_midpoint_gate": intern_pass,
            "decision": "confirm seeds 1/2" if intern_pass else "stop recipe refinement",
        },
        "Granite Vision 4.1-4B": {
            "protocol": "PROTOCOL_AMENDMENT_011",
            "base_macro": granite_base,
            "seed0_full_macro": granite_full,
            "seed0_rsm_macro": granite_rsm,
            "seed0_rsm_minus_full_pp": granite_rsm - granite_full,
            "seed0_rsm_minus_base_pp": granite_rsm - granite_base,
            "passes_seed0_promotion_gate": granite_pass,
            "decision": "confirm seeds 1/2" if granite_pass else "stop current-model transfer",
        },
    }


def write_csv(rows: list[dict]) -> None:
    fields = [
        "family",
        "system",
        "recipe",
        "seed",
        "evidence_status",
        *DISPLAY.values(),
        "macro_average",
        "route_entropy_ema",
        "gate_weight_rms",
        "gate_weight_max_abs",
        "identity_max_abs_logit_delta",
        "skip_training_forwards",
        "result_json",
        "train_summary",
    ]
    with (REPORTS / "GENERIC_VLM_REPAIR_ROWS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], verdicts: dict) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)

    recipe_order = [
        "lr=2e-4, entropy=0.02",
        "lr=5e-4, entropy=0.02",
        "lr=1e-3, entropy=0.02",
        "lr=1e-3, entropy=0",
    ]
    labels = ["2e-4\n+entropy", "5e-4\n+entropy", "1e-3\n+entropy", "1e-3\nno entropy"]
    colors = {"Full AttnRes": "#4C78A8", "Full RSM-AttnRes": "#E45756"}
    for ax, family in zip(axes[0], ("SmolVLM2-2.2B", "InternVL3.5-2B")):
        base = select(rows, family, "Frozen pretrained", "Base")[0]["macro_average"]
        for system in ("Full AttnRes", "Full RSM-AttnRes"):
            ys = []
            for recipe in recipe_order:
                cells = select(rows, family, recipe, system)
                ys.append(
                    statistics.fmean(item["macro_average"] for item in cells) - base
                    if cells
                    else np.nan
                )
            ax.plot(range(4), ys, marker="o", linewidth=1.8, color=colors[system], label=system)
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(4), labels)
        ax.set_ylabel("Macro delta vs Base (pp)")
        ax.set_title(f"{family}: adaptation-strength sensitivity")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    styles = {
        "SmolVLM2-2.2B": ("o", "#59A14F"),
        "InternVL3.5-2B": ("s", "#B279A2"),
        "Granite Vision 4.1-4B": ("^", "#F28E2B"),
    }
    for family, (marker, color) in styles.items():
        xs, ys, notes = [], [], []
        for recipe in recipe_order:
            cells = select(rows, family, recipe, "Full RSM-AttnRes")
            if not cells:
                continue
            cell = sorted(cells, key=lambda item: -1 if item["seed"] is None else item["seed"])[0]
            if cell["gate_weight_rms"] is None:
                continue
            xs.append(ENTROPY_CEILING[family] - cell["route_entropy_ema"])
            ys.append(cell["gate_weight_rms"])
            notes.append(recipe.split(",")[0].replace("lr=", ""))
        ax.plot(xs, ys, marker=marker, color=color, linewidth=1.6, label=family)
        for x, y, note in zip(xs, ys, notes):
            ax.annotate(note, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Route-specialization gap from uniform ceiling")
    ax.set_ylabel("RSM gate-weight RMS")
    ax.set_title("Specialization grows as adaptation becomes effective")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    smol = verdicts["SmolVLM2-2.2B"]
    paired = smol["paired_rsm_minus_full_pp"]
    bars = ax.bar(["seed 0", "seed 1", "seed 2"], paired, color="#E45756", width=0.62)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axhline(0.15, color="#F28E2B", linewidth=1.1, linestyle="--", label="mean gate: +0.15 pp")
    for bar, value in zip(bars, paired):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    ax.set_ylabel("RSM − matched Full (pp)")
    ax.set_title("SmolVLM2 paired-seed confirmation")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    for path in (
        FIGURES / "generic_vlm_repair_mechanism.png",
        FIGURES / "generic_vlm_repair_mechanism.pdf",
    ):
        fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(rows: list[dict], verdicts: dict) -> None:
    lines = [
        "# Generic VLM AttnRes diagnosis and bounded repair audit",
        "",
        "## Outcome",
        "",
    ]
    smol = verdicts["SmolVLM2-2.2B"]
    intern = verdicts["InternVL3.5-2B"]
    granite = verdicts["Granite Vision 4.1-4B"]
    lines.append(
        f"- SmolVLM2-2.2B: three-seed no-entropy RSM={smol['rsm_macro_mean']:.3f} ± {smol['rsm_macro_sd']:.3f}, "
        f"matched Full={smol['full_macro_mean']:.3f} ± {smol['full_macro_sd']:.3f}, "
        f"paired gain={smol['paired_delta_mean_pp']:+.3f} ± {smol['paired_delta_sd_pp']:.3f} pp; "
        f"decision: **{smol['decision']}**."
    )
    lines.append(
        f"- InternVL3.5-2B: bounded lr=5e-4 RSM={intern['midpoint_rsm_macro']:.3f}, "
        f"matched Full={intern['midpoint_full_macro']:.3f}, "
        f"delta={intern['midpoint_rsm_minus_full_pp']:+.3f} pp and "
        f"delta vs Base={intern['midpoint_rsm_minus_base_pp']:+.3f} pp; "
        f"decision: **{intern['decision']}**."
    )
    lines.append(
        f"- Granite Vision 4.1-4B: pre-registered seed-0 RSM={granite['seed0_rsm_macro']:.3f}, "
        f"matched Full={granite['seed0_full_macro']:.3f}, "
        f"delta={granite['seed0_rsm_minus_full_pp']:+.3f} pp and "
        f"delta vs Base={granite['seed0_rsm_minus_base_pp']:+.3f} pp; "
        f"decision: **{granite['decision']}**."
    )
    lines.extend(
        [
            "",
            "## Why the original supplemental runs were nearly flat",
            "",
            "The original generic recipe used lr=2e-4, inherited from an older skip-KL screen. Under the present Full-path-only objective it barely moved the new mechanism: SmolVLM2 gate-weight RMS was 0.0457 and InternVL was 0.0298. Their route entropies were 1.0937/1.0965 and 1.2187/1.2187 relative to the uniform-routing ceilings. The source mixtures therefore remained almost uniform, leaving too little block/source differentiation for RSM to exploit.",
            "",
            "Raising the learning rate made both routing and gates specialize, proving that the flat result was under-adaptation rather than a dead implementation. The high-rate InternVL benchmark nevertheless fell below Base, so stronger specialization is not monotonically better: the usable region depends on model family. Removing the uniform-routing reward helps SmolVLM2 but hurts InternVL, which is direct evidence against a universal one-recipe claim.",
            "",
            "For the three independently trained SmolVLM2 RSM seeds, route-entropy EMA stays in 0.7970–0.8204 and gate-weight RMS in 0.1250–0.1323, compared with 1.0937 and 0.0457 in the original run. Thus the repaired mechanism signal is repeatable even before consulting benchmark scores.",
            "",
            "Granite Vision 4.1-4B is the sole current-generation replacement test. Its recipe was frozen before the weight download completed. Under the exact tokenizer-regex setting, max sequence length 2048 accepts 15/16 preflight VLM samples at their model-native dynamic resolution and rejects one 2063-token example; no one-tile, truncation, or low-resolution shortcut is used.",
            "",
            "## Formal macro results",
            "",
            "| Family | Recipe | System | Seeds | Macro | Status |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for item in rows:
        groups.setdefault((item["family"], item["recipe"], item["system"]), []).append(item)
    for (family, recipe, system), cells in groups.items():
        values = [item["macro_average"] for item in cells]
        mean, sd = mean_and_sd(values)
        score_text = f"{mean:.3f} ± {sd:.3f}" if len(values) > 1 else f"{mean:.3f}"
        statuses = sorted({item["evidence_status"] for item in cells})
        lines.append(
            f"| {family} | {recipe} | {system} | {len(values)} | {score_text} | {', '.join(statuses)} |"
        )

    smol_full = sorted(
        select(rows, "SmolVLM2-2.2B", "lr=1e-3, entropy=0", "Full AttnRes"),
        key=lambda item: item["seed"],
    )
    smol_rsm = sorted(
        select(rows, "SmolVLM2-2.2B", "lr=1e-3, entropy=0", "Full RSM-AttnRes"),
        key=lambda item: item["seed"],
    )
    lines.extend(
        [
            "",
            "## SmolVLM2 paired-seed confirmation",
            "",
            "| Seed | Full AttnRes | Full RSM-AttnRes | RSM − Full (pp) |",
            "|---:|---:|---:|---:|",
        ]
    )
    for full, rsm in zip(smol_full, smol_rsm):
        lines.append(
            f"| {full['seed']} | {full['macro_average']:.3f} | {rsm['macro_average']:.3f} | "
            f"{rsm['macro_average'] - full['macro_average']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Acceptance required a mean paired gain of at least +0.15 pp, RSM mean above Base, and positive paired gains on at least two of three seeds. All cells are reported regardless of the decision.",
            "",
            "## Mechanism evidence",
            "",
            "![Generic VLM repair mechanism](../../figures/generic_vlm_repair_mechanism.png)",
            "",
            "The figure is descriptive rather than causal proof. Its strongest causal controls are the matched Full/RSM cells and the pre-registered entropy ablation; route entropy and gate norm diagnose whether adaptation actually created specialization.",
            "",
            "## Claim boundary",
            "",
            "- RSM remains part of ordinary Full-path AttnRes adaptation. There is no skip branch, skip supervision, compute target, warm start, or second stage in any reported repair run.",
            "- The RSM addition is 10,245 parameters on SmolVLM2 and 12,294 on InternVL (about 0.00046% and 0.00052% of the respective base models). These parameters are optimized jointly with AttnRes, not by a separate skip-training phase.",
            "- The supplemental models remain architecture-transfer tests. Their generic hook does not establish a device-side token-level ReSkip speed claim.",
            "- Rejected high-rate and no-entropy InternVL cells are diagnostics, not candidate results. The single lr=5e-4 point was frozen as a bounded midpoint; the protocol forbids further recipe search if it fails.",
            "- Granite Vision 3.3 stopped at input-length preflight before any complete benchmark cell or checkpoint. It is stored only as a failed preflight, not reported as a model result.",
            "",
            "## Material Passport",
            "",
            "- Origin Skill: academic-research-suite / experiment-agent",
            "- Origin Mode: run + validate",
            "- Origin Date: 2026-09-01",
            "- Verification Status: VERIFIED after complete six-task artifacts",
            "- Models: `/data/Minko/models/SmolVLM2-2.2B-Instruct`; `/data/Minko/models/InternVL3_5-2B-HF`; `/data/Minko/models/granite-vision-4.1-4b`",
            "- Current-model source: `https://huggingface.co/ibm-granite/granite-vision-4.1-4b`",
            "- Training data: `/data/Minko/datasets`, v3 mixed adaptation stream",
            "- Evaluation: local lmms-eval caches for AI2D, MMBench, MMMU, MMStar, OCRBench, and RealWorldQA",
            "- Hardware: node42 NVIDIA H100 GPUs 0-6; GPU 7 excluded",
            f"- Protocols: `{EXP / 'protocol/PROTOCOL_AMENDMENT_006.json'}`, `{EXP / 'protocol/PROTOCOL_AMENDMENT_007.json'}`, `{EXP / 'protocol/PROTOCOL_AMENDMENT_008.json'}`, `{EXP / 'protocol/PROTOCOL_AMENDMENT_009.json'}`, `{EXP / 'protocol/PROTOCOL_AMENDMENT_010.json'}`, `{EXP / 'protocol/PROTOCOL_AMENDMENT_011.json'}`, `{EXP / 'protocol/PROTOCOL_AMENDMENT_012.json'}`",
            f"- Software: `{EXP / 'code'}`",
        ]
    )
    (REPORTS / "GENERIC_VLM_REPAIR_AUDIT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    verdicts = decisions(rows)
    payload = {
        "status": "complete",
        "rows": rows,
        "decisions": verdicts,
        "protocols": [
            str(EXP / f"protocol/PROTOCOL_AMENDMENT_{number:03d}.json")
            for number in (6, 7, 8, 9, 10, 11, 12)
        ],
        "material_passport": {
            "origin_skill": "academic-research-suite / experiment-agent",
            "origin_mode": "run + validate",
            "origin_date": "2026-09-01",
            "verification_status": "VERIFIED after complete six-task artifacts",
            "models": [
                "SmolVLM2-2.2B-Instruct",
                "InternVL3_5-2B-HF",
                "Granite Vision 4.1-4B",
            ],
            "current_model_source": "https://huggingface.co/ibm-granite/granite-vision-4.1-4b",
            "training_data": "/data/Minko/datasets, v3 stream",
            "evaluation": list(DISPLAY.values()),
            "hardware": "node42 NVIDIA H100 GPUs 0-6; GPU 7 excluded",
            "software": str(EXP / "code"),
        },
    }
    (REPORTS / "GENERIC_VLM_REPAIR_AUDIT.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_csv(rows)
    plot(rows, verdicts)
    write_markdown(rows, verdicts)
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
