"""Q4 probe: per-example difficulty (token PPL) × per-example skip count correlation.

Tests whether ReSkip's dynamic skipping is correlated with example difficulty
(adaptive computation evidence) vs. uniform random firing (capacity-headroom
only). Run on the q=0.5 'B2c recent_minus_embed_gt' cell because q=0.85 fires
0% on lm-eval (see probe_q085_per_task.py).

Output:
  - per-example (text_idx, token_ppl, skip_count, fired_block_positions) JSON
  - Spearman ρ between PPL and skip_count
  - Per-quartile-of-difficulty mean skip count
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = "/home/user01/Minko/reskip2/reskip"
sys.path.insert(0, PROJECT + "/experiments")
from flame_reskip_common import load_model_and_tokenizer  # noqa: E402


@torch.no_grad()
def per_example(model, tok, device, *, dataset_id, split, text_field, n, seq_len, config):
    from datasets import load_dataset
    if config is not None:
        ds = load_dataset(dataset_id, config, split=split)
    else:
        ds = load_dataset(dataset_id, split=split)
    decoder = model.model
    num_pos = decoder.num_block_positions
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= n:
            break
        text = ex.get(text_field, "")
        if not text or not isinstance(text, str):
            continue
        ids = tok.encode(text.strip(), add_special_tokens=False)[:seq_len]
        if len(ids) < 8:
            continue
        x = torch.tensor([ids], device=device)
        out = model(input_ids=x, labels=x, return_routing_info=True,
                    use_cache=False, return_dict=True)
        ri = getattr(out, "routing_info", None)
        trace = ri.get("execution_trace", []) if ri else []
        # next-token CE → ppl
        logits = out.logits[:, :-1, :]
        targets = x[:, 1:]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction="mean",
        )
        ppl = float(math.exp(min(ce.item(), 30.0)))
        skip_count = sum(1 for e in trace if e.get("status") == "skipped")
        fired_positions = [int(e["position"]) for e in trace if e.get("status") == "skipped"]
        rows.append({
            "idx": i,
            "tokens": len(ids),
            "ce": float(ce.item()),
            "ppl": ppl,
            "skip_count": skip_count,
            "skip_positions": fired_positions,
            "n_eligible_block_positions": num_pos,
        })
    return rows


def spearman(x, y):
    """Tiny Spearman ρ — n^2 ranking, fine for n<5k."""
    if len(x) < 3:
        return 0.0
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx = sum(rx) / len(rx); my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=PROJECT + "/outputs/reskip_340M_b1b2_recent_minus_embed_gt_q050_M1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--out", default=PROJECT + "/retrofit/outputs/probe_difficulty_skip_correlation.json")
    args = ap.parse_args()

    print(f"[probe-q050-correlation] cell={args.cell}")
    model, tok = load_model_and_tokenizer(args.cell, args.device)
    model.eval()

    summary = {"cell": args.cell, "n_target": args.n, "seq_len": args.seq_len, "tasks": {}}
    for tid, did, split, field, cfg in [
        ("lambada", "EleutherAI/lambada_openai", "test", "text", None),
        ("hellaswag", "Rowan/hellaswag", "validation", "ctx", None),
    ]:
        print(f"\n=== {tid} ===")
        rows = per_example(model, tok, args.device,
                           dataset_id=did, split=split, text_field=field,
                           n=args.n, seq_len=args.seq_len, config=cfg)
        if not rows:
            continue
        ppls = [r["ppl"] for r in rows]
        skips = [r["skip_count"] for r in rows]
        ces = [r["ce"] for r in rows]
        rho_ppl = spearman(ppls, skips)
        rho_ce = spearman(ces, skips)

        # quartile-of-PPL → mean skip
        srt = sorted(rows, key=lambda r: r["ppl"])
        q = len(srt) // 4
        q_means = []
        for k in range(4):
            chunk = srt[k * q: (k + 1) * q] if k < 3 else srt[k * q:]
            if chunk:
                q_means.append(sum(r["skip_count"] for r in chunk) / len(chunk))
            else:
                q_means.append(0.0)
        print(f"  N={len(rows)}  mean ppl={sum(ppls)/len(ppls):.3f}  "
              f"mean skip={sum(skips)/len(skips):.3f}")
        print(f"  Spearman ρ(ppl, skip)={rho_ppl:+.3f}    ρ(ce, skip)={rho_ce:+.3f}")
        print(f"  PPL-quartile mean skip = {q_means} (low→high difficulty)")
        summary["tasks"][tid] = {
            "n": len(rows),
            "mean_ppl": sum(ppls) / len(ppls),
            "mean_skip": sum(skips) / len(skips),
            "spearman_rho_ppl_skip": rho_ppl,
            "spearman_rho_ce_skip": rho_ce,
            "ppl_quartile_mean_skip_low_to_high": q_means,
            "rows": rows,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
