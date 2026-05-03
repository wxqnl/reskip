"""Compile-mode decode latency: T=1 decode loop with torch.compile.

Question this answers (Q1 in advisor questions): under torch.compile, does
retrofit decode latency match base — closing the +30-40% eager gap?

Why this script and not the existing bench_retrofit_compile.py: that one
does single-shot prefill (`model(input_ids=full_prompt, use_cache=True)`),
not the deployment-relevant T=1 decode loop. Decode is where the per-block
router fires N times per token (vs once for the whole prompt at prefill),
so it's the harder regime for compile to amortize.

How: torch.compile(mode="default", dynamic=True) lets the growing KV cache
work without StaticCache. No CUDA graphs (those need fixed shapes) but
Inductor still traces and fuses the per-block router gemms into the
captured forward, removing the small-kernel-launch overhead that dominates
eager-mode retrofit decode.

Reports decode/tok (ms) for base and retrofit (no skip), eager + compiled,
at seq=1024, 2048. Ratio retrofit_compiled / base_compiled is the
writeable iso-cost number. Skip is intentionally not measured under
compile; the paper's appendix already documents that compile + skip
cannot compose in a single forward (the skip rule's .item() forces a CPU
sync that breaks the captured graph).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


@torch.no_grad()
def run_decode(m, ids, n_decode):
    out = m(input_ids=ids, use_cache=True)
    pkv = out.past_key_values
    cur = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_decode):
        out = m(input_ids=cur, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench(m, ids, n_decode, warmup, timed):
    for _ in range(warmup):
        run_decode(m, ids, n_decode)
    ts = []
    for _ in range(timed):
        ts.append(run_decode(m, ids, n_decode))
    return statistics.median(ts) * 1000  # ms (total decode loop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--state-path", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--seq-lens", default="1024,2048")
    ap.add_argument("--n-decode", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timed", type=int, default=10)
    ap.add_argument("--compile-mode", default="default",
                    help="default = trace+fuse, dynamic shapes OK; "
                         "reduce-overhead needs StaticCache (not used here)")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    print(f"[compile-decode] mode={args.compile_mode}", flush=True)
    print(f"[compile-decode] model={args.model_path}", flush=True)
    print(f"[compile-decode] state={args.state_path}", flush=True)
    print(f"[compile-decode] n_decode={args.n_decode}, warmup={args.warmup}, timed={args.timed}", flush=True)

    base = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=dtype).to(device).eval()
    retro_base = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=dtype).to(device).eval()
    ck = torch.load(args.state_path, map_location="cpu")
    cfg = ck.get("config", {})
    kw = dict(num_blocks=cfg.get("num_blocks", 7))
    if "adapter_rank" in cfg:
        kw["adapter_rank"] = cfg["adapter_rank"]
    retro = Qwen3VLAttnResRetrofit(retro_base, **kw).to(device=device, dtype=dtype).eval()
    retro.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    retro.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    retro.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))

    # No skip path: full retrofit, no dyn-skip rule
    retro._active_skip_blocks = set()
    retro._dynamic_skip_config = None

    print(f"[compile-decode] num_blocks={kw['num_blocks']}, γ={retro.gamma.detach().cpu().tolist()}", flush=True)

    base_compiled = torch.compile(base, mode=args.compile_mode, dynamic=True)
    retro_compiled = torch.compile(retro.base_model, mode=args.compile_mode, dynamic=True)

    for seq in [int(x) for x in args.seq_lens.split(",")]:
        ids = torch.randint(0, 100000, (1, seq), device=device)
        print(f"\n=== prefill seq_len = {seq}, decode = {args.n_decode} tok ===", flush=True)

        b_e = bench(base, ids, args.n_decode, args.warmup, args.timed) / args.n_decode
        r_e = bench(retro.base_model, ids, args.n_decode, args.warmup, args.timed) / args.n_decode
        b_c = bench(base_compiled, ids, args.n_decode, args.warmup, args.timed) / args.n_decode
        r_c = bench(retro_compiled, ids, args.n_decode, args.warmup, args.timed) / args.n_decode

        print(f"  {'config':40s} {'eager (ms/tok)':>16s} {'compiled (ms/tok)':>20s} {'eager->cmp':>12s}", flush=True)
        print(f"  {'TRUE base':40s} {b_e:>16.3f} {b_c:>20.3f} {b_e/b_c:>12.3f}x", flush=True)
        print(f"  {'VLM retrofit (no skip)':40s} {r_e:>16.3f} {r_c:>20.3f} {r_e/r_c:>12.3f}x", flush=True)
        print(f"\n  decode ratios vs base:", flush=True)
        print(f"    retrofit_eager    / base_eager    = {r_e/b_e:.3f}x", flush=True)
        print(f"    retrofit_compiled / base_compiled = {r_c/b_c:.3f}x  <-- writeable iso-cost", flush=True)
        print(f"    retrofit_compiled / base_eager    = {r_c/b_e:.3f}x  (compiled vs uncompiled base)", flush=True)


if __name__ == "__main__":
    main()
