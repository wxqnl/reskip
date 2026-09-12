"""Create the canonical four-panel RSM evidence figure from saved results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data/Minko/experiments/attnres_residual_strength_gate_2b_20260831")


def _load(relative: str):
    with (ROOT / relative).open() as file:
        return json.load(file)


def main() -> None:
    full = _load("full6/FULL6_COMPARISON.json")
    heldout = _load("screen_evidence_seed259123_v2/SCREEN_EVIDENCE.json")
    increment = _load(
        "screen_evidence_seed259123_v2/GATE_SKIP_INCREMENT.json"
    )
    final = _load("full6/FULL6_FINAL_TABLE.json")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    blue = "#3568A8"
    orange = "#D97932"
    gray = "#8A8A8A"
    fig, axes = plt.subplots(
        2, 2, figsize=(11.2, 7.5), layout="constrained"
    )

    # A: full-compute task deltas against the paper's original AttnRes.
    original, candidate = full["systems"][1], full["systems"][2]
    tasks = ["AI2D", "MMBench", "MMMU", "MMStar", "OCRBench", "RealWorldQA"]
    delta = np.array([candidate[t] - original[t] for t in tasks])
    y = np.arange(len(tasks))
    axes[0, 0].barh(
        y,
        delta,
        color=[blue if value >= 0 else gray for value in delta],
        height=0.62,
    )
    axes[0, 0].axvline(0, color="black", linewidth=0.8)
    axes[0, 0].set_yticks(y, tasks)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Score change (percentage points)")
    axes[0, 0].set_title(
        "A  Full-path change "
        f"({full['candidate_minus_paper_attnres_pp']:+.3f} pp macro)",
        loc="left",
        weight="bold",
        fontsize=10.5,
    )
    axes[0, 0].set_xlim(-1.35, 1.90)
    for index, value in enumerate(delta):
        axes[0, 0].text(
            value + (0.05 if value >= 0 else -0.05),
            index,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )
    # B: learned token/block variation. Error bars denote one standard deviation.
    stats = heldout["gate_factor_by_block"]
    blocks = np.array(sorted(int(block) for block in stats))
    means = np.array([stats[str(block)]["mean"] for block in blocks])
    stds = np.array([stats[str(block)]["std"] for block in blocks])
    axes[0, 1].errorbar(
        blocks,
        means,
        yerr=stds,
        fmt="o",
        color=orange,
        ecolor=orange,
        elinewidth=1.8,
        capsize=3,
    )
    axes[0, 1].axhline(1.0, color=gray, linestyle="--", linewidth=1, label="identity")
    axes[0, 1].axhline(0.5, color=gray, linestyle=":", linewidth=1, label="lower bound")
    axes[0, 1].set_xticks(blocks)
    axes[0, 1].set_ylim(0.42, 1.12)
    axes[0, 1].set_xlabel("AttnRes block")
    axes[0, 1].set_ylabel("Residual-strength factor")
    axes[0, 1].set_title(
        "B  Learned block/token differentiation",
        loc="left",
        weight="bold",
        fontsize=10.5,
    )
    axes[0, 1].legend(frameon=False, fontsize=8, loc="upper left")

    # C: cross-fitted diagnostic only; no selector is deployed from this fit.
    thresholds = [0.01, 0.05, 0.10]
    native_auc = [increment["thresholds"][str(t)]["macro_native_auc"] for t in thresholds]
    augmented_auc = [
        increment["thresholds"][str(t)]["macro_augmented_auc"] for t in thresholds
    ]
    axes[1, 0].plot(thresholds, native_auc, "o-", color=gray, label="native features")
    axes[1, 0].plot(thresholds, augmented_auc, "o-", color=orange, label="+ RSM factor")
    axes[1, 0].set_xticks(thresholds)
    axes[1, 0].set_xlabel("Unsafe-skip KL threshold")
    axes[1, 0].set_ylabel("Cross-fitted macro AUC")
    axes[1, 0].set_title(
        "C  Incremental skip-risk information",
        loc="left",
        weight="bold",
        fontsize=10.5,
    )
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_ylim(0.555, 0.58)

    # D: complete six-task Pareto comparison.
    final_rows = {row["system"]: row for row in final["systems"]}
    series = [
        (
            "Original AttnRes",
            gray,
            [0.0, final["paper_reskip"]["block_equivalents_per_decode_token"]],
            [
                final_rows["Full AttnRes (paper)"]["macro_average"],
                final_rows["ReSkip (paper)"]["macro_average"],
            ],
        ),
        (
            "RSM-AttnRes",
            orange,
            [0.0, final["rsm_reskip"]["block_equivalents_per_decode_token"]],
            [
                final_rows["Full RSM-AttnRes"]["macro_average"],
                final_rows["RSM-ReSkip"]["macro_average"],
            ],
        ),
    ]
    for label, color, x_values, y_values in series:
        axes[1, 1].plot(x_values, y_values, "o-", color=color, label=label)
        axes[1, 1].annotate(
            f"{y_values[1]:.3f}",
            (x_values[1], y_values[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    axes[1, 1].set_xlim(-0.05, 1.42)
    axes[1, 1].set_ylim(66.95, 68.08)
    axes[1, 1].set_xlabel("Removed block-equivalents / decode token")
    axes[1, 1].set_ylabel("Six-task macro score")
    axes[1, 1].set_title(
        "D  Formal token-level ReSkip Pareto",
        loc="left",
        weight="bold",
        fontsize=10.5,
    )
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower left")

    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "rsm_method_evidence.png", bbox_inches="tight")
    fig.savefig(output / "rsm_method_evidence.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
