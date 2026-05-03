"""Q1: Qwen3-VL 2B/4B latency speedup sweep under cache.

Measures prefill latency at several seq_lens for:
  (1) TRUE base (stock Qwen3-VL, no retrofit wrapper, no AttnRes)
  (2) Retrofit full path (γ=1, no skip — AttnRes overhead only)
  (3) Retrofit + forced-skip M ∈ {1, 2, 3, 4, 6, 8} blocks

Both cache=False (prefill) and cache=True (decode-style steady-state).

Skip-rate ↔ q mapping (from probe_b1_b2_skip_rate / B2c at q=0.5 ≈ 12.5%
block-level fire rate, q=0.85 calibrated 0% on lm-eval): we report
forced-static M curves so the user can map their preferred q → expected
average M from the probe data.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT = "/home/user01/Minko/reskip2/reskip"
sys.path.insert(0, PROJECT + "/retrofit")
from transformers import AutoModelForImageTextToText
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


PRESETS = {
    "2B": {
        "model": "/home/user01/Minko/models/Qwen3-VL-2B",
        "state": PROJECT + "/retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt",
        "default_M": [1, 2, 3, 4, 6, 8],
    },
    "4B": {
        "model": "/home/user01/Minko/models/Qwen3-VL-4B",
        "state": PROJECT + "/retrofit/outputs/H_4B_r256_5k/retrofit_attnres_state.pt",
        "default_M": [1, 2, 3, 4, 6, 8],
    },
}


@torch.no_grad()
def bench(call, warmup, timed):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000.0  # ms


def pick_skip_blocks(skippable, M):
    # Skip evenly spaced blocks excluding 0 (block 0 should run for input grounding).
    skippable = sorted(set(skippable) - {0})
    if M >= len(skippable):
        return list(skippable)
    # Linspace over indices into skippable.
    if M == 0:
        return []
    step = len(skippable) / M
    chosen = sorted({skippable[int(i * step)] for i in range(M)})
    while len(chosen) < M:
        # fill with next missing indices in case of duplicates
        for b in skippable:
            if b not in chosen:
                chosen.append(b); break
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["2B", "4B"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seq_lens", default="512,1024,2048")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--timed", type=int, default=15)
    ap.add_argument("--M_values", default=None,
                    help="comma-sep skip budgets; defaults from preset")
    ap.add_argument("--out", default=None,
                    help="JSON path; defaults to retrofit/outputs/bench_qwen3vl_skip_<scale>.json")
    args = ap.parse_args()

    pre = PRESETS[args.scale]
    out_path = Path(args.out or f"{PROJECT}/retrofit/outputs/bench_qwen3vl_skip_{args.scale}.json")
    M_values = [int(x) for x in (args.M_values.split(",") if args.M_values else pre["default_M"])]

    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16
    print(f"[bench] scale={args.scale} model={pre['model']} state={pre['state']}")
    print(f"[bench] M values = {M_values}")

    true_base = AutoModelForImageTextToText.from_pretrained(pre["model"], dtype=dtype).to(device).eval()

    rfit_base = AutoModelForImageTextToText.from_pretrained(pre["model"], dtype=dtype).to(device).eval()
    ck = torch.load(pre["state"], map_location="cpu")
    cfg = ck.get("config", {})
    kw = dict(num_blocks=cfg.get("num_blocks", 14))
    if "adapter_rank" in cfg:
        kw["adapter_rank"] = cfg["adapter_rank"]
    rfit = Qwen3VLAttnResRetrofit(rfit_base, **kw).to(device=device, dtype=dtype).eval()
    rfit.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    rfit.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    rfit.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
    skippable = list(rfit.skippable_blocks)

    results = {"scale": args.scale, "skippable": skippable, "by_seq": {}}

    for seq in [int(s) for s in args.seq_lens.split(",")]:
        ids = torch.randint(0, 100000, (1, seq), device=device)
        print(f"\n=== seq_len = {seq} ===")
        seq_res = {}

        b_nc = bench(lambda: true_base(input_ids=ids, use_cache=False), args.warmup, args.timed)
        b_c  = bench(lambda: true_base(input_ids=ids, use_cache=True),  args.warmup, args.timed)
        seq_res["true_base"] = {"cache_F_ms": b_nc, "cache_T_ms": b_c}
        print(f"  TRUE base                  cache=F {b_nc:7.2f}   cache=T {b_c:7.2f}")

        f_nc = bench(lambda: rfit(input_ids=ids, use_cache=False), args.warmup, args.timed)
        f_c  = bench(lambda: rfit(input_ids=ids, use_cache=True),  args.warmup, args.timed)
        seq_res["retrofit_full"] = {
            "cache_F_ms": f_nc, "cache_T_ms": f_c,
            "ratio_cache_F": f_nc / b_nc, "ratio_cache_T": f_c / b_c,
        }
        print(f"  retrofit full (γ=1)         cache=F {f_nc:7.2f} ({f_nc/b_nc:.3f}x)   "
              f"cache=T {f_c:7.2f} ({f_c/b_c:.3f}x)")

        for M in M_values:
            skip_blocks = pick_skip_blocks(skippable, M)
            s_nc = bench(lambda blk=skip_blocks: rfit(input_ids=ids, use_cache=False,
                                                      skip_block_indices=blk),
                         args.warmup, args.timed)
            s_c  = bench(lambda blk=skip_blocks: rfit(input_ids=ids, use_cache=True,
                                                      skip_block_indices=blk),
                         args.warmup, args.timed)
            seq_res[f"skip_M{M}"] = {
                "skip_blocks": skip_blocks,
                "cache_F_ms": s_nc, "cache_T_ms": s_c,
                "ratio_cache_F_vs_base": s_nc / b_nc,
                "ratio_cache_T_vs_base": s_c / b_c,
                "ratio_cache_T_vs_retrofit": s_c / f_c,
                "speedup_vs_base_cacheT": b_c / s_c,
            }
            print(f"  retrofit skip M={M} blks={skip_blocks}  "
                  f"cache=F {s_nc:7.2f} ({s_nc/b_nc:.3f}x)   "
                  f"cache=T {s_c:7.2f} ({s_c/b_c:.3f}x)   "
                  f"speedup vs base = {b_c/s_c:.3f}x")

        results["by_seq"][seq] = seq_res

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
