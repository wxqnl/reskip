"""Token-level ReSkip visualization for the adaptive-depth figure.

This script measures ReSkip at the same granularity where the router makes
decisions: one cached decode forward. For each VLM prompt we follow the
full-depth greedy continuation for a short horizon, compare full-depth logits
against ReSkip logits at the same history, and record:

  * full-depth token margin Delta = logit(top1) - logit(top2);
  * skip perturbation 2 epsilon over the full-depth top-k candidate tokens;
  * whether the top-1 token is preserved;
  * routing weights and skipped blocks.

The default operating point is a diagnostic, difficulty-aligned ReSkip point
found by a routing/margin scan: block 2, tau=0.51, max_skips=1. It is intended
for the mechanism figure, not as a replacement for the main benchmark config.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "retrofit"))

from retrofit.analysis.reskip_adaptive_depth_figure import (  # noqa: E402
    aggregate_trace_to_vectors,
    build_prompt,
    load_model,
    load_task_dataset,
    rankdata,
    raw_retrofit_base_forward,
    to_jsonable,
)


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def trace_skip_count(trace: list[dict[str, Any]] | None) -> int:
    return sum(1 for item in trace or [] if item.get("skipped"))


@torch.no_grad()
def collect_token_records(model, processor, ds, args, dynamic_cfg):
    device = args.device
    token_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    num_blocks = int(model.num_blocks)
    t0 = time.time()

    for row_idx, ex in enumerate(ds):
        item = build_prompt(args.task, ex, row_idx)
        if item is None:
            continue

        prefix = processor.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prefix], images=item["images"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        full_out, _ = raw_retrofit_base_forward(model, inputs, None, False)
        skip_out, skip_trace = raw_retrofit_base_forward(model, inputs, dynamic_cfg, True)
        full_past = full_out.past_key_values
        skip_past = skip_out.past_key_values
        full_logits = full_out.logits[:, -1, :]
        skip_logits = skip_out.logits[:, -1, :]

        sample_margins: list[float] = []
        sample_skips: list[int] = []
        w_accum = [[] for _ in range(num_blocks)]
        skipped_any: set[int] = set()

        for step in range(args.decode_tokens):
            full_vec = full_logits[0].float()
            skip_vec = skip_logits[0].float()
            top = torch.topk(full_vec, k=args.top_k)
            top_indices = top.indices
            top_values = top.values
            token_id = int(top_indices[0].item())
            margin = float((top_values[0] - top_values[1]).item())
            diff_topk = (skip_vec[top_indices] - top_values).abs()
            epsilon = float(diff_topk.max().item())
            skip_count = trace_skip_count(skip_trace)
            preserved = bool(int(skip_vec.argmax().item()) == token_id)
            condition = bool(2.0 * epsilon <= margin)

            w_vec, skipped_blocks, requested_blocks = aggregate_trace_to_vectors(skip_trace, num_blocks)
            for block_idx, value in enumerate(w_vec):
                if value is not None:
                    w_accum[block_idx].append(float(value))
            skipped_any.update(skipped_blocks)

            token_records.append(
                {
                    "sample_idx": len(sample_records),
                    "dataset_row": row_idx,
                    "sample_id": item["sample_id"],
                    "decode_step": step,
                    "margin_delta": margin,
                    "epsilon_topk": epsilon,
                    "two_epsilon": 2.0 * epsilon,
                    "prediction_preserved": preserved,
                    "margin_preservation_condition": condition,
                    "skip_count": skip_count,
                    "skipped_blocks": skipped_blocks,
                    "dynamic_requested_blocks": requested_blocks,
                    "top_token_id": token_id,
                    "question_preview": item["question"],
                }
            )
            sample_margins.append(margin)
            sample_skips.append(skip_count)

            cur = torch.tensor([[token_id]], device=device)
            full_out, _ = raw_retrofit_base_forward(
                model, {"input_ids": cur, "past_key_values": full_past}, None, False
            )
            skip_out, skip_trace = raw_retrofit_base_forward(
                model, {"input_ids": cur, "past_key_values": skip_past}, dynamic_cfg, True
            )
            full_past = full_out.past_key_values
            skip_past = skip_out.past_key_values
            full_logits = full_out.logits[:, -1, :]
            skip_logits = skip_out.logits[:, -1, :]

        sample_records.append(
            {
                "sample_idx": len(sample_records),
                "dataset_row": row_idx,
                "sample_id": item["sample_id"],
                "mean_margin_delta": float(np.mean(sample_margins)),
                "mean_skipped_blocks": float(np.mean(sample_skips)),
                "w_recent_mean": [
                    float(np.mean(values)) if values else None for values in w_accum
                ],
                "skipped_blocks_any": sorted(skipped_any),
                "question_preview": item["question"],
            }
        )

        if args.progress_every and len(sample_records) % args.progress_every == 0:
            preserved = np.mean([r["prediction_preserved"] for r in token_records])
            cond = np.mean([r["margin_preservation_condition"] for r in token_records])
            skip = np.mean([r["skip_count"] for r in token_records])
            print(
                f"[collect] samples={len(sample_records):4d}/{len(ds)} "
                f"tokens={len(token_records):5d} preserved={preserved:.3f} "
                f"condition={cond:.3f} skip={skip:.3f} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

    return token_records, sample_records


def uncertainty_bins(token_records, n_bins: int = 3):
    margins = np.asarray([r["margin_delta"] for r in token_records], dtype=float)
    skips = np.asarray([r["skip_count"] for r in token_records], dtype=float)
    if n_bins == 2:
        q = float(np.quantile(margins, 0.5))
        groups = [
            ("Low\n(easy)", margins > q),
            ("High\n(hard)", margins <= q),
        ]
        stats = []
        for label, mask in groups:
            vals = skips[mask]
            mvals = margins[mask]
            stats.append(
                {
                    "label": label,
                    "n": int(mask.sum()),
                    "mean_skips": float(vals.mean()) if len(vals) else 0.0,
                    "sem_skips": float(vals.std(ddof=1) / math.sqrt(len(vals)))
                    if len(vals) > 1
                    else 0.0,
                    "mean_margin": float(mvals.mean()) if len(mvals) else 0.0,
                }
            )
        return stats, (q,)

    q1, q2 = np.quantile(margins, [1 / 3, 2 / 3])
    groups = [
        ("Low\n(easy)", margins > q2),
        ("Medium", (margins > q1) & (margins <= q2)),
        ("High\n(hard)", margins <= q1),
    ]
    stats = []
    for label, mask in groups:
        vals = skips[mask]
        mvals = margins[mask]
        stats.append(
            {
                "label": label,
                "n": int(mask.sum()),
                "mean_skips": float(vals.mean()) if len(vals) else 0.0,
                "sem_skips": float(vals.std(ddof=1) / math.sqrt(len(vals)))
                if len(vals) > 1
                else 0.0,
                "mean_margin": float(mvals.mean()) if len(mvals) else 0.0,
            }
        )
    return stats, (float(q1), float(q2))


def select_heatmap_samples(sample_records, rows_per_group):
    easy = sorted(
        sample_records,
        key=lambda r: (
            r["mean_skipped_blocks"] <= 0,
            -r["mean_margin_delta"],
            r["sample_idx"],
        ),
    )
    hard = sorted(
        sample_records,
        key=lambda r: (
            r["mean_margin_delta"],
            r["mean_skipped_blocks"],
            r["sample_idx"],
        ),
    )
    selected = []
    seen = set()
    for r in easy:
        if len(selected) >= rows_per_group:
            break
        selected.append(r)
        seen.add(r["sample_idx"])
    for r in hard:
        if len(selected) >= rows_per_group * 2:
            break
        if r["sample_idx"] in seen:
            continue
        selected.append(r)
        seen.add(r["sample_idx"])
    return selected


def summarize(token_records, sample_records, args, dataset_name, dyn_cfg, model_meta):
    margins = np.asarray([r["margin_delta"] for r in token_records], dtype=float)
    skips = np.asarray([r["skip_count"] for r in token_records], dtype=float)
    preserved = np.asarray([r["prediction_preserved"] for r in token_records], dtype=bool)
    cond = np.asarray([r["margin_preservation_condition"] for r in token_records], dtype=bool)
    two_eps = np.asarray([r["two_epsilon"] for r in token_records], dtype=float)
    bins, margin_bins = uncertainty_bins(token_records, args.difficulty_bins)
    return {
        "task": args.task,
        "dataset": dataset_name,
        "probe_type": "token_level_decode",
        "n_samples": len(sample_records),
        "n_tokens": len(token_records),
        "decode_tokens_per_sample": int(args.decode_tokens),
        "top_k_for_epsilon": int(args.top_k),
        "model_path": args.model_path,
        "state_path": str(args.state_path),
        "dynamic_config": {
            "thresholds": {str(k): v for k, v in dyn_cfg["thresholds"].items()},
            "eligible_blocks": sorted(dyn_cfg["eligible_blocks"]),
            "max_skips": dyn_cfg["max_skips"],
            "strategy": "recent_weight_gt",
            "notes": "difficulty-aligned diagnostic point selected from routing/margin scan",
        },
        "checkpoint_config": model_meta["checkpoint_config"],
        "prediction_preserved_rate": float(preserved.mean()),
        "margin_condition_rate": float(cond.mean()),
        "mean_skipped_blocks_per_forward": float(skips.mean()),
        "mean_margin_delta": float(margins.mean()),
        "median_margin_delta": float(np.median(margins)),
        "mean_two_epsilon": float(two_eps.mean()),
        "median_two_epsilon": float(np.median(two_eps)),
        "spearman_uncertainty_skip": spearman(-margins, skips),
        "difficulty_bins": int(args.difficulty_bins),
        "uncertainty_bins": bins,
        "margin_bin_thresholds": margin_bins,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def plot(token_records, sample_records, summary, fig_path, png_path, rows_per_group):
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(10.6, 3.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.85, 1.15])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    num_blocks = len(sample_records[0]["w_recent_mean"])
    rows = select_heatmap_samples(sample_records, rows_per_group)
    heat = np.full((len(rows), num_blocks), np.nan, dtype=float)
    for row_idx, record in enumerate(rows):
        for block_idx, value in enumerate(record["w_recent_mean"]):
            if value is not None:
                heat[row_idx, block_idx] = value
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    im = ax0.imshow(np.ma.masked_invalid(heat), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    for row_idx, record in enumerate(rows):
        for block_idx in record["skipped_blocks_any"]:
            ax0.add_patch(
                Rectangle(
                    (block_idx - 0.5, row_idx - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#d62728",
                    linewidth=1.8,
                )
            )
    ax0.set_xticks(range(num_blocks))
    ax0.set_xticklabels([f"B{i}" for i in range(num_blocks)])
    ylabels = []
    for row_idx, record in enumerate(rows):
        group = "easy" if row_idx < rows_per_group else "hard"
        ylabels.append(
            f"{group} #{record['sample_id']}\n"
            f"Delta={record['mean_margin_delta']:.2f}, skip={record['mean_skipped_blocks']:.2f}"
        )
    ax0.set_yticks(range(len(rows)))
    ax0.set_yticklabels(ylabels)
    ax0.set_title("(a) Routing differs across inputs")
    ax0.set_xlabel("Block")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label("recent-source weight")

    bins = summary["uncertainty_bins"]
    xs = np.arange(len(bins))
    means = [b["mean_skips"] for b in bins]
    sems = [b["sem_skips"] for b in bins]
    labels = [b["label"] for b in bins]
    ax1.bar(xs, means, yerr=sems, capsize=3, color=["#4c78a8", "#72b7b2", "#f58518"], alpha=0.92)
    ax1.plot(xs, means, color="#222222", linewidth=1.0, marker="o", markersize=3)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Avg. skipped blocks / decode forward")
    rho = summary.get("spearman_uncertainty_skip")
    rho_text = "n/a" if rho is None else f"{rho:.2f}"
    ax1.text(0.03, 0.95, f"Spearman rho={rho_text}", transform=ax1.transAxes, ha="left", va="top")
    ax1.set_title("(b) Harder steps keep more depth")
    ax1.grid(axis="y", alpha=0.25)

    margins = np.asarray([r["margin_delta"] for r in token_records], dtype=float)
    two_eps = np.asarray([r["two_epsilon"] for r in token_records], dtype=float)
    preserved = np.asarray([r["prediction_preserved"] for r in token_records], dtype=bool)
    cond = np.asarray([r["margin_preservation_condition"] for r in token_records], dtype=bool)
    finite = np.isfinite(margins) & np.isfinite(two_eps)
    margins = margins[finite]
    two_eps = two_eps[finite]
    preserved = preserved[finite]
    cond = cond[finite]
    lim = float(np.quantile(np.maximum(margins, two_eps), 0.995) * 1.08)
    lim = max(lim, 1.0)
    line = np.linspace(0.0, lim, 200)
    ax2.fill_between(line, 0, line, color="#59a14f", alpha=0.11, linewidth=0)
    ax2.scatter(margins[preserved], two_eps[preserved], s=14, color="#1f77b4", alpha=0.58, label="prediction preserved", edgecolors="none")
    if (~preserved).any():
        ax2.scatter(margins[~preserved], two_eps[~preserved], s=22, color="#d62728", alpha=0.82, label="prediction changed", marker="x")
    ax2.plot(line, line, "--", color="#222222", linewidth=1.0, label=r"$2\epsilon=\Delta$")
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.set_xlabel(r"Full-depth token margin $\Delta(x)$")
    ax2.set_ylabel(r"Skip perturbation $2\epsilon(x)$")
    ax2.set_title("(c) Margin-preservation region")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper left", frameon=False)
    ax2.text(
        0.98,
        0.04,
        f"{100 * cond.mean():.1f}% below boundary\n"
        f"{100 * preserved.mean():.1f}% predictions preserved",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )

    fig.suptitle("Figure X: ReSkip adapts depth according to input difficulty", fontsize=10, y=1.02)
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/user01/Minko/models/Qwen3-VL-2B")
    parser.add_argument("--state-path", type=Path, default=Path("retrofit/outputs/H_2B_r256_10k_L4_v3/retrofit_attnres_state.pt"))
    parser.add_argument("--task", choices=["mmbench", "mmstar"], default="mmbench")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--difficulty-bins", type=int, choices=[2, 3], default=2)
    parser.add_argument("--eligible-block", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.512)
    parser.add_argument("--max-skips", type=int, default=1)
    parser.add_argument("--rows-per-group", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=40)
    parser.add_argument("--out-dir", type=Path, default=Path("retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160"))
    parser.add_argument("--fig-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    if args.task == "mmstar" and args.split == "dev":
        args.split = "val"
    args.state_path = args.state_path.resolve()
    args.out_dir = args.out_dir.resolve()
    args.fig_dir = args.fig_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    dynamic_cfg = {
        "thresholds": {int(args.eligible_block): float(args.threshold)},
        "eligible_blocks": {int(args.eligible_block)},
        "max_skips": int(args.max_skips),
    }
    print(
        f"[setup] device={args.device} task={args.task} n={args.n} "
        f"P={sorted(dynamic_cfg['eligible_blocks'])} tau={args.threshold} M={args.max_skips}",
        flush=True,
    )
    model, processor, model_meta = load_model(args.model_path, args.state_path, args.device, dtype, None)
    ds, dataset_name = load_task_dataset(args.task, args.split, args.n, args.offset)
    print(f"[data] {dataset_name}/{args.split}, rows={len(ds)}", flush=True)

    token_records, sample_records = collect_token_records(model, processor, ds, args, dynamic_cfg)
    summary = summarize(token_records, sample_records, args, dataset_name, dynamic_cfg, model_meta)

    records_path = args.out_dir / "token_records.jsonl"
    with records_path.open("w") as f:
        for record in token_records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
    samples_path = args.out_dir / "sample_records.jsonl"
    with samples_path.open("w") as f:
        for record in sample_records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    fig_path = args.fig_dir / "reskip_adaptive_depth_vlm_token.pdf"
    png_path = args.fig_dir / "reskip_adaptive_depth_vlm_token.png"
    plot(token_records, sample_records, summary, fig_path, png_path, args.rows_per_group)

    print(f"[done] token records: {records_path}", flush=True)
    print(f"[done] sample records: {samples_path}", flush=True)
    print(f"[done] summary:       {summary_path}", flush=True)
    print(f"[done] figure:        {fig_path}", flush=True)
    print(f"[done] figure:        {png_path}", flush=True)
    print(
        "[summary] "
        f"samples={summary['n_samples']} tokens={summary['n_tokens']} "
        f"skip={summary['mean_skipped_blocks_per_forward']:.4f} "
        f"preserved={summary['prediction_preserved_rate']:.4f} "
        f"condition={summary['margin_condition_rate']:.4f} "
        f"rho={summary['spearman_uncertainty_skip']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
