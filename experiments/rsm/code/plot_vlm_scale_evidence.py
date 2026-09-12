"""Build the paper-facing scale, ablation, and ReSkip evidence figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data/Minko")
EXP = ROOT / "experiments/attnres_rsm_vlm_scale_generalization_20260831"
RSM2 = ROOT / "experiments/attnres_residual_strength_gate_2b_20260831"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def rows_by_key(table: dict) -> dict[tuple[str, str, int | None], dict]:
    return {
        (row["family"], row["system"], row["seed"]): row
        for row in table["rows"]
    }


def gate_kl(record: dict) -> list[float]:
    interventions = record["residual_strength_gate_interventions"]
    return [
        interventions["learned_gate_subset"]["teacher_kl"],
        interventions["identity_gate"]["teacher_kl"],
        interventions["cyclic_gate_permutation"]["teacher_kl"],
    ]


def main() -> None:
    table = load(EXP / "reports/VLM_FINAL_QUALITY_TABLES.json")
    indexed = rows_by_key(table)
    heldout2 = load(
        RSM2
        / "S1_rsg_h1_screen_seed259123_n128/HELDOUT_FULLPATH_RSG_SUMMARY.json"
    )
    heldout4 = load(
        EXP
        / "reskip/qwen3vl_4b_seed0_heldout_seed259123/HELDOUT_FULLPATH_RSG_SUMMARY.json"
    )
    increment2 = load(
        RSM2 / "screen_evidence_seed259123_v2/GATE_SKIP_INCREMENT.json"
    )
    increment4 = load(
        EXP
        / "reskip/qwen3vl_4b_seed0_heldout_seed259123/GATE_SKIP_INCREMENT.json"
    )
    runtime2 = load(RSM2 / "FINAL_RUNTIME_SUMMARY.json")
    fixed4 = load(
        EXP
        / "runtime/qwen3vl_4b_seed2_capped/fixed64_12_per_task_r3/BENCHMARK_NATURAL_RUNTIME.json"
    )
    natural4 = load(
        EXP
        / "runtime/qwen3vl_4b_seed2_capped/natural12_per_task_r3/BENCHMARK_NATURAL_RUNTIME.json"
    )
    skip4 = load(EXP / "eval/reskip/qwen3vl_4b_seed2_capped/skip_stats.json")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    blue, orange, gray, green = "#3568A8", "#D97932", "#8A8A8A", "#3A8D71"
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.5), layout="constrained")

    # A: paired RSM gain over the corresponding Full AttnRes seed.
    families = ("Qwen3-VL-2B", "Qwen3-VL-4B")
    x_positions, values, colors, labels = [], [], [], []
    cursor = 0
    for family, color in zip(families, (blue, orange)):
        for seed in range(3):
            full = indexed[(family, "Full AttnRes", seed)]["macro_average"]
            rsm = indexed[(family, "Full RSM-AttnRes", seed)]["macro_average"]
            x_positions.append(cursor)
            values.append(rsm - full)
            colors.append(color)
            labels.append(f"{family[-2:]} s{seed}")
            cursor += 1
        cursor += 0.7
    axes[0, 0].bar(x_positions, values, color=colors, width=0.72)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_xticks(x_positions, labels, rotation=35, ha="right")
    axes[0, 0].set_ylabel("RSM − Full AttnRes (pp)")
    axes[0, 0].set_title("A  Matched-seed full-path gain", loc="left", weight="bold")
    axes[0, 0].set_ylim(0.0, max(values) + 0.11)
    for x_value, value in zip(x_positions, values):
        axes[0, 0].text(
            x_value,
            value + (0.04 if value >= 0 else -0.04),
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7.5,
        )

    # B: direct intervention on the learned residual-strength factor.
    ablation_labels = ("learned", "identity", "cyclic")
    x = np.arange(len(ablation_labels))
    width = 0.35
    axes[0, 1].bar(x - width / 2, gate_kl(heldout2), width, color=blue, label="2B")
    axes[0, 1].bar(x + width / 2, gate_kl(heldout4), width, color=orange, label="4B")
    axes[0, 1].set_xticks(x, ablation_labels)
    axes[0, 1].set_ylabel("Teacher KL on paired subset")
    axes[0, 1].set_title("B  RSM causal intervention", loc="left", weight="bold")
    axes[0, 1].legend(frameon=False)

    # C: token-level and block-level factor variation.
    for record, label, color, offset in (
        (heldout2, "2B", blue, -0.08),
        (heldout4, "4B", orange, 0.08),
    ):
        stats = record["learned_residual_strength_factors"]
        blocks = np.array(sorted(int(block) for block in stats))
        means = np.array([stats[str(block)]["mean"] for block in blocks])
        stds = np.array([stats[str(block)]["std"] for block in blocks])
        axes[0, 2].errorbar(
            blocks + offset,
            means,
            yerr=stds,
            fmt="o",
            capsize=2.5,
            color=color,
            label=label,
        )
    axes[0, 2].axhline(1.0, color=gray, linestyle="--", linewidth=1)
    axes[0, 2].set_xlabel("AttnRes block")
    axes[0, 2].set_ylabel("Residual-strength factor")
    axes[0, 2].set_title("C  Block/token differentiation", loc="left", weight="bold")
    axes[0, 2].legend(frameon=False)

    # D: diagnostic increment to skip-risk discrimination; this is not a controller.
    thresholds = (0.01, 0.05, 0.10)
    for record, label, color in (
        (increment2, "2B", blue),
        (increment4, "4B", orange),
    ):
        deltas = [
            record["thresholds"][str(value)]["macro_augmented_auc"]
            - record["thresholds"][str(value)]["macro_native_auc"]
            for value in thresholds
        ]
        axes[1, 0].plot(thresholds, deltas, "o-", color=color, label=label)
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_xticks(thresholds)
    axes[1, 0].set_xlabel("Unsafe-skip KL threshold")
    axes[1, 0].set_ylabel("Cross-fitted AUC increment")
    axes[1, 0].set_title("D  Skip-risk diagnostic", loc="left", weight="bold")
    axes[1, 0].legend(frameon=False)

    # E: formal quality/compute Pareto on the two supported Qwen scales.
    block2 = runtime2["quality"]["formal_rollout_block_equivalents_per_decode_token"]
    block4 = skip4["avg_partial_decoder_layer_skips_per_decode_forward"] / 4.0
    for family, seed, block_equiv, color, label in (
        ("Qwen3-VL-2B", 0, block2, blue, "2B"),
        ("Qwen3-VL-4B", 2, block4, orange, "4B"),
    ):
        full = indexed[(family, "Full RSM-AttnRes", seed)]["macro_average"]
        reskip = indexed[(family, "RSM-ReSkip", seed)]["macro_average"]
        axes[1, 1].plot([0, block_equiv], [full, reskip], "o-", color=color, label=label)
        axes[1, 1].annotate(
            f"{block_equiv:.2f} blocks",
            (block_equiv, reskip),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
        )
    axes[1, 1].set_xlabel("Removed block-equivalents / decode token")
    axes[1, 1].set_ylabel("Six-task macro score")
    axes[1, 1].set_title("E  Formal token-level Pareto", loc="left", weight="bold")
    axes[1, 1].legend(frameon=False)

    # F: matched benchmark speedup against Base.
    speed_fixed = [
        runtime2["fixed_64_token_runtime"]["matched_length_speedups"]
        ["rsm_reskip_vs_base"]["geometric_mean"],
        fixed4["macro"]["rsm_reskip_vs_base"]
        ["matched_length_pooled_sample_geometric_mean"],
    ]
    speed_natural = [
        runtime2["natural_generation_runtime"]
        ["rsm_reskip_vs_base_matched_length"]["pooled_sample_geometric_mean"],
        natural4["macro"]["rsm_reskip_vs_base"]
        ["matched_length_pooled_sample_geometric_mean"],
    ]
    x = np.arange(2)
    axes[1, 2].bar(x - width / 2, speed_fixed, width, color=green, label="fixed-64")
    axes[1, 2].bar(x + width / 2, speed_natural, width, color=gray, label="natural")
    axes[1, 2].axhline(1.0, color="black", linewidth=0.8)
    axes[1, 2].set_xticks(x, ("2B", "4B"))
    axes[1, 2].set_ylabel("Speedup vs Base")
    axes[1, 2].set_title("F  Benchmark runtime", loc="left", weight="bold")
    axes[1, 2].legend(frameon=False)
    for x_value, pair in enumerate(zip(speed_fixed, speed_natural)):
        for shift, value in zip((-width / 2, width / 2), pair):
            axes[1, 2].text(x_value + shift, value + 0.006, f"{value:.3f}×", ha="center", fontsize=7.2)

    output = EXP / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "vlm_scale_method_evidence.png", bbox_inches="tight")
    fig.savefig(output / "vlm_scale_method_evidence.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
