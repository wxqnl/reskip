"""Quick probe: load each B1/B2 cell, run a few LAMBADA prompts, count actual skip events.

Used to verify that q=0.5 cells fire enough skips to be meaningful (vs. the
existing q=0.85 reskip_35 cell which was observed to fire 0% skips on lm-eval).

Output: per-cell skip-rate summary line.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT = "/home/user01/Minko/reskip2/reskip"
sys.path.insert(0, PROJECT + "/experiments")

from flame_reskip_common import load_model_and_tokenizer  # noqa: E402


CELLS = {
    # B0 baseline (no skip): kept for reference
    "B0_no_skip":               PROJECT + "/flame/saves/reskip_transformer-340M",
    # Original q=0.85 cell (the one we suspect doesn't fire on lm-eval data)
    "Reskip_q085":              PROJECT + "/outputs/reskip_340M_combined_35_skip2_q085",
    # New q=0.5 cells
    "B1b_recent_weight_gt":     PROJECT + "/outputs/reskip_340M_b1b2_recent_weight_gt_q050_M1",
    "B2b_entropy_lt":           PROJECT + "/outputs/reskip_340M_b1b2_entropy_lt_q050_M1",
    "B2c_recent_minus_embed_gt": PROJECT + "/outputs/reskip_340M_b1b2_recent_minus_embed_gt_q050_M1",
    # Static (always-fire)
    "B1c_static_p3_every":      PROJECT + "/outputs/reskip_340M_b1_static_p3_every",
    "B1d_static_p5_every":      PROJECT + "/outputs/reskip_340M_b1_static_p5_every",
}


@torch.no_grad()
def measure_skip_rate(model, tokenizer, device, *, n_seqs=8, seq_len=512):
    """Returns (avg blocks executed / total positions, per-position skip prob)."""
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test").shuffle(seed=0)
    decoder = model.model
    num_pos = decoder.num_block_positions
    skip_count = [0] * num_pos
    total_calls = 0
    total_tokens = 0
    skip_events = 0

    for i, ex in enumerate(ds):
        if i >= n_seqs:
            break
        ids = tokenizer.encode(ex["text"].strip(), add_special_tokens=False)[:seq_len]
        if not ids:
            continue
        x = torch.tensor([ids], device=device)
        out = model(input_ids=x, return_routing_info=True, use_cache=False, return_dict=True)
        ri = getattr(out, "routing_info", None)
        if not ri:
            continue
        trace = ri.get("execution_trace", [])
        # trace is one entry per position; status="skipped" if skipped
        for entry in trace:
            pos = int(entry["position"])
            if entry.get("status") == "skipped":
                skip_count[pos] += 1
                skip_events += 1
        total_calls += 1
        total_tokens += x.numel()

    print(f"  N seqs={total_calls}, total skip events={skip_events}, "
          f"per-position skip events={skip_count}")
    if total_calls > 0:
        rate = skip_events / (total_calls * num_pos)
        print(f"  block-level skip rate ≈ {rate*100:.2f}%")
        return rate, skip_count
    return 0.0, skip_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", default=",".join(CELLS),
                   help="comma-sep names from CELLS dict")
    p.add_argument("--n_seqs", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    names = [s.strip() for s in args.cells.split(",") if s.strip() in CELLS]
    print(f"[probe] cells={names}, n_seqs={args.n_seqs}, seq_len={args.seq_len}, device={args.device}")
    for name in names:
        path = CELLS[name]
        print(f"\n=== {name} ===")
        print(f"  path={path}")
        model, tok = load_model_and_tokenizer(path, args.device)
        model.eval()
        rate, per_pos = measure_skip_rate(model, tok, args.device,
                                          n_seqs=args.n_seqs, seq_len=args.seq_len)
        # Cleanup before next load
        del model, tok
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
