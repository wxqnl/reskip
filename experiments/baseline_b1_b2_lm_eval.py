"""Run lm-eval on a 340M reskip checkpoint, with optional runtime random-skip override.

Modes:
  --runtime_mode embedded     Use whatever skip policy is in the checkpoint config.
  --runtime_mode random       Override per-call: each forward picks a random keep_mask
                              from --random_choices (semicolon-separated bool lists).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

PROJECT = "/home/user01/Minko/reskip2/reskip"
sys.path.insert(0, PROJECT + "/experiments")

from flame_reskip_common import load_model_and_tokenizer  # noqa: E402


def _parse_keep_mask(spec: str) -> list[bool]:
    bits = [int(x) for x in spec.split(",")]
    return [bool(b) for b in bits]


def install_random_skip_hook(model, choices: list[list[bool]], seed: int = 0):
    """Wrap the inner decoder's forward so that on every call, a random keep_mask
    from `choices` is selected and pushed via set_skip_keep_mask.

    Per-call randomization: lm-eval calls model(..) once per request batch; setting
    keep_mask immediately before each forward gives per-batch random skip pattern,
    which approximates per-token random over many calls.
    """
    rng = random.Random(seed)
    decoder = model.model
    orig_forward = decoder.forward

    def patched_forward(*args, **kwargs):
        mask = list(rng.choice(choices))
        decoder.set_skip_keep_mask(mask)
        return orig_forward(*args, **kwargs)

    decoder.forward = patched_forward
    print(f"[random-hook] installed; {len(choices)} possible masks; seed={seed}")


def install_skip_stats_hook(model, *, trace_examples: int = 3):
    """Collect actual skip decisions made by lm-eval forwards.

    The hook asks the model to return routing_info and aggregates the execution
    trace from the same calls used for scoring, so a zero skip-rate cannot be
    mistaken for a valid dynamic-skip benchmark.
    """
    stats = {
        "total_forwards": 0,
        "forwards_with_trace": 0,
        "total_positions": 0,
        "skip_events": 0,
        "executed_blocks_sum": 0.0,
        "per_position_skips": [],
        "per_position_total": [],
        "trace_examples": [],
    }
    orig_forward = model.forward

    def ensure_position(position: int) -> None:
        while len(stats["per_position_skips"]) <= position:
            stats["per_position_skips"].append(0)
            stats["per_position_total"].append(0)

    def patched_forward(*args, **kwargs):
        kwargs["return_routing_info"] = True
        out = orig_forward(*args, **kwargs)
        stats["total_forwards"] += 1
        routing_info = getattr(out, "routing_info", None)
        trace = routing_info.get("execution_trace", []) if routing_info else []
        if trace:
            stats["forwards_with_trace"] += 1
            stats["total_positions"] += len(trace)
            stats["executed_blocks_sum"] += float(routing_info.get("num_blocks_executed", 0.0))
            for entry in trace:
                pos = int(entry["position"])
                ensure_position(pos)
                stats["per_position_total"][pos] += 1
                if entry.get("status") == "skipped":
                    stats["per_position_skips"][pos] += 1
                    stats["skip_events"] += 1
            if len(stats["trace_examples"]) < trace_examples:
                stats["trace_examples"].append(
                    [
                        {
                            "position": int(entry["position"]),
                            "status": entry.get("status"),
                            "avg_phase1_recent_weight": entry.get("avg_phase1_recent_weight"),
                            "avg_phase1_embed_weight": entry.get("avg_phase1_embed_weight"),
                            "avg_phase1_entropy": entry.get("avg_phase1_entropy"),
                        }
                        for entry in trace
                    ]
                )
        return out

    model.forward = patched_forward
    print("[skip-stats] installed; lm-eval forwards will return routing_info")
    return stats


def finalize_skip_stats(stats: dict) -> dict:
    total_positions = max(int(stats.get("total_positions", 0)), 1)
    forwards_with_trace = max(int(stats.get("forwards_with_trace", 0)), 1)
    per_position_total = stats.get("per_position_total", [])
    per_position_skips = stats.get("per_position_skips", [])
    return {
        **stats,
        "skip_rate_block_level": float(stats.get("skip_events", 0)) / total_positions,
        "avg_blocks_executed": float(stats.get("executed_blocks_sum", 0.0)) / forwards_with_trace,
        "per_position_skip_rate": [
            (float(skips) / total if total else 0.0)
            for skips, total in zip(per_position_skips, per_position_total)
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tasks", required=True)
    p.add_argument("--batch_size", default="8")
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--runtime_mode", choices=("embedded", "random"), default="embedded")
    p.add_argument("--random_choices", default="",
                   help="For runtime_mode=random; semicolon-separated keep_masks "
                        "(e.g. '1,1,1,0,1,1,1,1;1,1,1,1,1,0,1,1' for skip-P3-or-skip-P5).")
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--collect_skip_stats", action="store_true")
    p.add_argument("--skip_stats_trace_examples", type=int, default=3)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"[lm_eval] loading {args.model_path} on {args.device}")
    model, tok = load_model_and_tokenizer(args.model_path, args.device, dtype=args.dtype)
    model.eval()

    if args.runtime_mode == "random":
        if not args.random_choices:
            raise ValueError("--random_choices required for runtime_mode=random")
        choices = [_parse_keep_mask(s) for s in args.random_choices.split(";") if s.strip()]
        # Make sure dynamic skip is OFF; we use static keep_mask flips
        decoder = model.model
        decoder.clear_dynamic_skip_policy()
        decoder.clear_skip_keep_mask()
        install_random_skip_hook(model, choices, seed=args.random_seed)

    skip_stats = None
    if args.collect_skip_stats:
        skip_stats = install_skip_stats_hook(model, trace_examples=args.skip_stats_trace_examples)

    from lm_eval.models.huggingface import HFLM
    import lm_eval

    lm = HFLM(
        pretrained=model,
        tokenizer=tok,
        backend="causal",
        batch_size=args.batch_size,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[lm_eval] tasks: {tasks}")

    eval_kwargs = dict(model=lm, tasks=tasks, log_samples=False, verbosity="INFO")
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit

    results = lm_eval.simple_evaluate(**eval_kwargs)

    summary = {
        "model_path": args.model_path,
        "label": args.label,
        "tasks": tasks,
        "runtime_mode": args.runtime_mode,
        "random_choices": args.random_choices if args.runtime_mode == "random" else None,
        "limit": args.limit,
        "policy": {
            "enable_skip_inference": getattr(model.config, "enable_skip_inference", None),
            "skip_keep_mask": getattr(model.config, "skip_keep_mask", None),
            "dynamic_skip_strategy": getattr(model.config, "dynamic_skip_strategy", None),
            "dynamic_skip_probe_mode": getattr(model.config, "dynamic_skip_probe_mode", None),
            "dynamic_skip_position_thresholds": getattr(model.config, "dynamic_skip_position_thresholds", None),
            "dynamic_skip_max_skips": getattr(model.config, "dynamic_skip_max_skips", None),
        },
        "skip_stats": finalize_skip_stats(skip_stats) if skip_stats is not None else None,
        "results": results.get("results", {}) if isinstance(results, dict) else None,
    }
    out_path = Path(args.output) / "summary.json"
    out_path.write_text(json.dumps(summary, default=str, indent=2))
    print(f"[lm_eval] summary -> {out_path}")

    print("\n=== TASK RESULTS ===")
    for task_name, task_res in (results.get("results", {}) or {}).items():
        if isinstance(task_res, dict):
            metrics = {k: v for k, v in task_res.items() if k != "alias" and not k.endswith("_stderr")}
            print(f"  {task_name}: {metrics}")


if __name__ == "__main__":
    main()
