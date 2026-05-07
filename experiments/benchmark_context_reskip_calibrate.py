"""Calibrate dynamic ReSkip thresholds on lm-eval benchmark contexts.

The pretrain-corpus calibration can choose thresholds that never trigger on
lm-eval prompts, or trigger early blocks that are not benchmark-safe. This
script collects dynamic-skip probe metrics from the exact lm-eval request
contexts while running full-depth forwards, then exports candidate checkpoint
configs for a small threshold sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "experiments"))

from flame_analyze_reskip import dynamic_disable_threshold, dynamic_metric_from_trace  # noqa: E402
from flame_reskip_common import load_model_and_tokenizer, parse_csv_floats, save_json  # noqa: E402


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def q_tag(q: float) -> str:
    return f"q{int(round(q * 100)):03d}"


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile for an empty value list.")
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, q).item())


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": len(values),
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "max": float(tensor.max().item()),
    }


def install_metric_collection_hook(
    model,
    *,
    strategy: str,
    probe_mode: str,
    num_positions: int,
    trace_examples: int,
):
    stats: dict[str, Any] = {
        "strategy": strategy,
        "probe_mode": probe_mode,
        "total_forwards": 0,
        "forwards_with_trace": 0,
        "total_trace_entries": 0,
        "per_position_values": [[] for _ in range(num_positions)],
        "trace_examples": [],
    }
    orig_forward = model.forward
    disabled_thresholds = [dynamic_disable_threshold(strategy)] * num_positions

    def patched_forward(*args, **kwargs):
        kwargs["return_routing_info"] = True
        kwargs["dynamic_skip_strategy"] = strategy
        kwargs["dynamic_skip_probe_mode"] = probe_mode
        kwargs["dynamic_skip_position_thresholds"] = disabled_thresholds
        kwargs["dynamic_skip_max_skips"] = 0
        out = orig_forward(*args, **kwargs)

        stats["total_forwards"] += 1
        routing_info = getattr(out, "routing_info", None)
        trace = routing_info.get("execution_trace", []) if routing_info else []
        if trace:
            stats["forwards_with_trace"] += 1
            stats["total_trace_entries"] += len(trace)
            for entry in trace:
                position = int(entry["position"])
                value = dynamic_metric_from_trace(entry, strategy)
                if value is not None:
                    stats["per_position_values"][position].append(float(value))
            if len(stats["trace_examples"]) < trace_examples:
                stats["trace_examples"].append(
                    [
                        {
                            "position": int(entry["position"]),
                            "status": entry.get("status"),
                            "avg_phase1_recent_weight": entry.get("avg_phase1_recent_weight"),
                            "avg_phase1_embed_weight": entry.get("avg_phase1_embed_weight"),
                            "avg_phase1_entropy": entry.get("avg_phase1_entropy"),
                            "metric_value": dynamic_metric_from_trace(entry, strategy),
                        }
                        for entry in trace
                    ]
                )
        return out

    model.forward = patched_forward
    return stats, orig_forward


def build_policy(
    *,
    strategy: str,
    probe_mode: str,
    num_positions: int,
    positions: list[int],
    per_position_values: list[list[float]],
    threshold_quantile: float,
    max_skips: int,
) -> dict[str, Any]:
    thresholds = [dynamic_disable_threshold(strategy)] * num_positions
    for position in positions:
        if position <= 0 or position >= num_positions - 1:
            raise ValueError(f"Position {position} is not skippable for {num_positions} block positions.")
        values = per_position_values[position]
        if not values:
            raise ValueError(f"No calibration values collected for position {position}.")
        thresholds[position] = quantile(values, threshold_quantile)

    return {
        "strategy": strategy,
        "granularity": "block",
        "probe_mode": probe_mode,
        "positions": positions,
        "threshold_quantile": threshold_quantile,
        "position_thresholds": thresholds,
        "max_skips": max_skips,
    }


def export_policy_model(model, tokenizer, export_dir: Path, policy: dict[str, Any], analysis_payload: dict[str, Any]) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    decoder = model.model
    decoder.clear_skip_keep_mask()
    decoder.set_dynamic_skip_policy(
        strategy=policy["strategy"],
        granularity=policy["granularity"],
        probe_mode=policy["probe_mode"],
        position_thresholds=policy["position_thresholds"],
        max_skips=policy["max_skips"],
    )
    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    save_json(export_dir / "policy.json", policy)
    save_json(export_dir / "benchmark_context_analysis.json", analysis_payload)
    decoder.clear_dynamic_skip_policy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--batch_size", default="8")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--export_base_dir", required=True)
    parser.add_argument("--strategy", default="recent_weight_gt")
    parser.add_argument("--probe_mode", default="attn_only")
    parser.add_argument("--positions", default="5", help="Zero-indexed block positions, e.g. '5' or '4,5,6'.")
    parser.add_argument("--threshold_quantiles", default="0.60,0.70,0.80,0.85,0.90")
    parser.add_argument("--max_skips", type=int, default=1)
    parser.add_argument("--trace_examples", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    export_base_dir = Path(args.export_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_base_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, dtype=args.dtype)
    model.eval()
    num_positions = int(model.model.num_block_positions)
    positions = parse_csv_ints(args.positions)
    threshold_quantiles = parse_csv_floats(args.threshold_quantiles)

    stats, orig_forward = install_metric_collection_hook(
        model,
        strategy=args.strategy,
        probe_mode=args.probe_mode,
        num_positions=num_positions,
        trace_examples=args.trace_examples,
    )

    from lm_eval.models.huggingface import HFLM
    import lm_eval

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
    )
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    eval_kwargs: dict[str, Any] = {
        "model": lm,
        "tasks": tasks,
        "log_samples": False,
        "verbosity": "INFO",
    }
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit

    results = lm_eval.simple_evaluate(**eval_kwargs)
    model.forward = orig_forward

    per_position_values = stats["per_position_values"]
    position_summaries = [
        summarize_values(values)
        for values in per_position_values
    ]
    policies = [
        build_policy(
            strategy=args.strategy,
            probe_mode=args.probe_mode,
            num_positions=num_positions,
            positions=positions,
            per_position_values=per_position_values,
            threshold_quantile=q,
            max_skips=args.max_skips,
        )
        for q in threshold_quantiles
    ]

    analysis_payload: dict[str, Any] = {
        "model_path": args.model_path,
        "tasks": tasks,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "strategy": args.strategy,
        "probe_mode": args.probe_mode,
        "positions": positions,
        "max_skips": args.max_skips,
        "position_summaries": position_summaries,
        "collection_stats": {
            key: value for key, value in stats.items()
            if key != "per_position_values"
        },
        "calibration_results": results.get("results", {}) if isinstance(results, dict) else None,
        "policies": policies,
    }
    save_json(output_dir / "benchmark_context_analysis.json", analysis_payload)

    for policy in policies:
        pos_tag = "pos" + "-".join(str(position) for position in positions)
        export_dir = export_base_dir / f"{pos_tag}_{q_tag(policy['threshold_quantile'])}_M{policy['max_skips']}"
        export_policy_model(model, tokenizer, export_dir, policy, analysis_payload)
        print(
            "[export] "
            f"{export_dir} thresholds={json.dumps(policy['position_thresholds'])}"
        )

    print(f"[analysis] wrote {output_dir / 'benchmark_context_analysis.json'}")
    for position in positions:
        summary = position_summaries[position]
        print(f"[position {position}] {summary}")


if __name__ == "__main__":
    main()
