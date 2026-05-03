"""Q3 probe: q=0.85 fire rate on PIQA / MMLU / OpenBookQA / ARC-E / ARC-C / HellaSwag / LAMBADA.

Verifies the 'q=0.85 fires 0% on multiple-choice lm-eval distribution'
hypothesis that explains why ReSkip(q=0.85) is bit-identical to AttnRes-full
in `llm_340M_extended_benchmarks.md`.

Cells: Reskip_q085 (340M).
Output: per-task block-level skip rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT = "/home/user01/Minko/reskip2/reskip"
sys.path.insert(0, PROJECT + "/experiments")
from flame_reskip_common import load_model_and_tokenizer  # noqa: E402

CELL = PROJECT + "/outputs/reskip_340M_combined_35_skip2_q085"

# (task_id, hf_dataset, split, text_field) — pick the field that gives a
# representative input the model would see during scoring.
TASKS = [
    ("lambada",       "EleutherAI/lambada_openai", "test",        "text"),
    ("hellaswag",     "Rowan/hellaswag",           "validation",  "ctx"),
    ("piqa",          "ybisk/piqa",                "validation",  "goal"),
    ("openbookqa",    "allenai/openbookqa",        "validation",  "question_stem"),
    ("arc_easy",      "allenai/ai2_arc",           "validation",  "question"),
    ("arc_challenge", "allenai/ai2_arc",           "validation",  "question"),
    ("mmlu",          "cais/mmlu",                 "validation",  "question"),
]


@torch.no_grad()
def measure(model, tok, device, *, dataset_id, split, text_field, n=64, seq_len=512, config=None):
    from datasets import load_dataset
    if config is not None:
        ds = load_dataset(dataset_id, config, split=split)
    else:
        ds = load_dataset(dataset_id, split=split)
    decoder = model.model
    num_pos = decoder.num_block_positions
    skip_events = 0
    total_calls = 0
    total_pos = 0
    for i, ex in enumerate(ds):
        if i >= n:
            break
        text = ex.get(text_field, "")
        if not text or not isinstance(text, str):
            continue
        ids = tok.encode(text.strip(), add_special_tokens=False)[:seq_len]
        if not ids:
            continue
        x = torch.tensor([ids], device=device)
        out = model(input_ids=x, return_routing_info=True, use_cache=False, return_dict=True)
        ri = getattr(out, "routing_info", None)
        if not ri:
            continue
        trace = ri.get("execution_trace", [])
        for entry in trace:
            if entry.get("status") == "skipped":
                skip_events += 1
        total_calls += 1
        total_pos += num_pos
    rate = skip_events / total_pos if total_pos else 0.0
    return rate, skip_events, total_calls, total_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=CELL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--out", default=PROJECT + "/retrofit/outputs/probe_q085_per_task.json")
    args = ap.parse_args()

    print(f"[probe-q085] cell={args.cell}")
    model, tok = load_model_and_tokenizer(args.cell, args.device)
    model.eval()

    results = {}
    for spec in TASKS:
        # mmlu / arc_easy / arc_challenge use a config name distinct from id
        if spec[0] == "mmlu":
            tid, did, split, field = spec; cfg = "all"
        elif spec[0] == "arc_easy":
            tid, did, split, field = spec; cfg = "ARC-Easy"
        elif spec[0] == "arc_challenge":
            tid, did, split, field = spec; cfg = "ARC-Challenge"
        elif spec[0] == "openbookqa":
            tid, did, split, field = spec; cfg = "main"
        else:
            tid, did, split, field = spec; cfg = None
        print(f"\n=== {tid} ===  {did}/{cfg or '-'}  field={field}")
        try:
            rate, ev, calls, pos = measure(
                model, tok, args.device,
                dataset_id=did, split=split, text_field=field,
                n=args.n, seq_len=args.seq_len, config=cfg,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results[tid] = {"error": str(e)}
            continue
        print(f"  block-level skip rate = {rate*100:.3f}% "
              f"(events={ev}, calls={calls}, eligible_positions_total={pos})")
        results[tid] = {
            "skip_rate_block_level": rate,
            "skip_events": ev,
            "calls": calls,
            "eligible_positions_total": pos,
            "n_examples_used": args.n,
            "seq_len_max": args.seq_len,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
