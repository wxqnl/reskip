from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsons(path: Path):
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def select_recommended_skip(summary_rows: list[dict]) -> dict | None:
    skip_rows = [row for row in summary_rows if row["scenario"].startswith("attnres_skip_")]
    full_row = next((row for row in summary_rows if row["scenario"] == "attnres_full"), None)
    if not skip_rows or full_row is None or full_row["aggregate_mean"] is None:
        return None

    full_score = full_row["aggregate_mean"]
    viable = [
        row for row in skip_rows
        if row["aggregate_mean"] is not None
        and row["aggregate_mean"] >= full_score * 0.97
        and row["avg_blocks_executed"] is not None
    ]
    if viable:
        return min(viable, key=lambda row: row["avg_blocks_executed"])

    scored = [row for row in skip_rows if row["aggregate_mean"] is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: row["aggregate_mean"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark comparisons for ReSkip 350M pipeline.")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    evals = load_jsons(Path(args.eval_dir))
    profiles = {item["model_path"] + f"::{item['enable_skipping']}::{item['skip_threshold']}": item for item in load_jsons(Path(args.profile_dir))}

    labels = [item["scenario"] for item in evals]
    aggregate_scores = [item["aggregate_mean"] or 0.0 for item in evals]
    avg_blocks = []
    for item in evals:
        key = item["model_path"] + f"::{item['enable_skipping']}::{item['skip_threshold']}"
        avg_blocks.append((profiles.get(key) or {}).get("avg_blocks_executed", 0.0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(labels, aggregate_scores, color=["#4C78A8" if "baseline" in label else "#F58518" for label in labels])
    axes[0].set_ylabel("Mean lm_eval Score")
    axes[0].set_title("350M Benchmark Comparison")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].scatter(avg_blocks, aggregate_scores, s=80, c="#54A24B")
    for label, x, y in zip(labels, avg_blocks, aggregate_scores):
        axes[1].annotate(label, (x, y), fontsize=9)
    axes[1].set_xlabel("Average Blocks Executed")
    axes[1].set_ylabel("Mean lm_eval Score")
    axes[1].set_title("Score vs Effective Depth")

    fig.tight_layout()
    plot_path = output_dir / "reskip_350m_benchmark.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    summary = [
        {
            "scenario": label,
            "aggregate_mean": score,
            "avg_blocks_executed": block,
        }
        for label, score, block in zip(labels, aggregate_scores, avg_blocks)
    ]
    recommended = select_recommended_skip(summary)
    with (output_dir / "summary_table.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    with (output_dir / "selection_summary.json").open("w") as f:
        json.dump(
            {
                "recommended_skip": recommended,
                "selection_rule": "Prefer the lowest compute skip setting within 3% of attnres_full aggregate lm_eval score; otherwise choose the highest-scoring skip setting.",
            },
            f,
            indent=2,
        )
        f.write("\n")
    with (output_dir / "selection_summary.md").open("w") as f:
        f.write("# ReSkip 350M 结果摘要\n\n")
        if recommended is None:
            f.write("未能从 skip sweep 中选出推荐阈值。\n")
        else:
            f.write("## 推荐 skip 设置\n\n")
            f.write(f"- 场景: `{recommended['scenario']}`\n")
            f.write(f"- 平均 lm_eval 分数: `{recommended['aggregate_mean']:.4f}`\n")
            f.write(f"- 平均执行 block 数: `{recommended['avg_blocks_executed']:.2f}`\n")
            f.write("- 规则: 在不超过 `attnres_full` 3% 分数损失的前提下，选择执行 block 最少的 skip 设置；若没有满足条件的设置，则退化为选择分数最高的 skip 设置。\n")

    print(f"Saved plots and summary to {output_dir}")


if __name__ == "__main__":
    main()
