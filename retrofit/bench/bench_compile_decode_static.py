"""Compile-mode T=1 decode latency using StaticCache.

Why this script: torch.compile + decode + DynamicCache doesn't compose because
the cache shape grows per step and Inductor retraces. The standard fix is
StaticCache: preallocate a max-length tensor and write into fixed positions
each step. With fixed shapes, mode="reduce-overhead" can capture the decode
forward as a single CUDA graph that replays at every step.

This is what closes the +30-40% eager retrofit gap for Q1's deployment-relevant
T=1 decode regime — the writeable iso-cost number for the decode path.

Reports decode/tok (ms) for base and retrofit (no skip), eager + compiled,
at prefill_seq ∈ {1024, 2048}. The ratio retrofit_compiled / base_compiled
is the writeable Q1 decode iso-cost number.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText
from transformers.cache_utils import StaticCache
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


def make_static_cache(model, batch, max_len, device, dtype):
    text_cfg = model.config.get_text_config()
    return StaticCache(
        config=text_cfg,
        max_batch_size=batch,
        max_cache_len=max_len,
        device=device,
        dtype=dtype,
    )


def reset_static_cache(cache):
    """Zero-out a StaticCache so it can be reused across timed iterations."""
    if hasattr(cache, "reset"):
        cache.reset()
        return
    for layer in cache.layers:
        if hasattr(layer, "keys") and layer.keys is not None:
            layer.keys.zero_()
        if hasattr(layer, "values") and layer.values is not None:
            layer.values.zero_()


@torch.no_grad()
def run_decode(m, prefill_ids, n_decode, cache):
    reset_static_cache(cache)
    device = prefill_ids.device
    prefill_len = prefill_ids.shape[1]

    # CUDA-graph step begin: lets compiled CUDA graphs reuse cached tensors
    # without tripping the "overwritten by subsequent run" guard. Required
    # whenever a compiled module mutates a persistent buffer (here: StaticCache).
    torch.compiler.cudagraph_mark_step_begin()
    cache_pos_pre = torch.arange(prefill_len, device=device)
    out = m(input_ids=prefill_ids, past_key_values=cache,
            cache_position=cache_pos_pre, use_cache=True)
    cur = out.logits[:, -1:].argmax(-1).clone()
    cur_pos = prefill_len

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_decode):
        torch.compiler.cudagraph_mark_step_begin()
        cp = torch.tensor([cur_pos], device=device)
        out = m(input_ids=cur, past_key_values=cache,
                cache_position=cp, use_cache=True)
        cur = out.logits[:, -1:].argmax(-1).clone()
        cur_pos += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench(m, prefill_ids, n_decode, cache, warmup, timed):
    for _ in range(warmup):
        run_decode(m, prefill_ids, n_decode, cache)
    ts = []
    for _ in range(timed):
        ts.append(run_decode(m, prefill_ids, n_decode, cache))
    return statistics.median(ts) * 1000  # total decode loop in ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--state-path", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--seq-lens", default="1024,2048")
    ap.add_argument("--n-decode", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timed", type=int, default=10)
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    help="reduce-overhead enables CUDA graphs (needs fixed shape - "
                         "StaticCache provides that). default also OK but slower.")
    ap.add_argument("--max-cache-len", type=int, default=4096)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    print(f"[compile-decode-static] mode={args.compile_mode}", flush=True)
    print(f"[compile-decode-static] model={args.model_path}", flush=True)
    print(f"[compile-decode-static] state={args.state_path}", flush=True)
    print(f"[compile-decode-static] n_decode={args.n_decode}, warmup={args.warmup}, timed={args.timed}, max_cache_len={args.max_cache_len}", flush=True)

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

    retro._active_skip_blocks = set()
    retro._dynamic_skip_config = None

    print(f"[compile-decode-static] num_blocks={kw['num_blocks']}, γ={retro.gamma.detach().cpu().tolist()}", flush=True)

    base_compiled = torch.compile(base, mode=args.compile_mode, dynamic=False)
    retro_compiled = torch.compile(retro.base_model, mode=args.compile_mode, dynamic=False)

    base_cache_e = make_static_cache(base, 1, args.max_cache_len, device, dtype)
    base_cache_c = make_static_cache(base, 1, args.max_cache_len, device, dtype)
    retro_cache_e = make_static_cache(retro.base_model, 1, args.max_cache_len, device, dtype)
    retro_cache_c = make_static_cache(retro.base_model, 1, args.max_cache_len, device, dtype)

    for seq in [int(x) for x in args.seq_lens.split(",")]:
        ids = torch.randint(0, 100000, (1, seq), device=device)
        print(f"\n=== prefill seq_len = {seq}, decode = {args.n_decode} tok (StaticCache) ===", flush=True)

        b_e = bench(base, ids, args.n_decode, base_cache_e, args.warmup, args.timed) / args.n_decode
        r_e = bench(retro.base_model, ids, args.n_decode, retro_cache_e, args.warmup, args.timed) / args.n_decode
        b_c = bench(base_compiled, ids, args.n_decode, base_cache_c, args.warmup, args.timed) / args.n_decode
        r_c = bench(retro_compiled, ids, args.n_decode, retro_cache_c, args.warmup, args.timed) / args.n_decode

        print(f"  {'config':40s} {'eager (ms/tok)':>16s} {'compiled (ms/tok)':>20s} {'eager->cmp':>12s}", flush=True)
        print(f"  {'TRUE base':40s} {b_e:>16.3f} {b_c:>20.3f} {b_e/b_c:>12.3f}x", flush=True)
        print(f"  {'VLM retrofit (no skip)':40s} {r_e:>16.3f} {r_c:>20.3f} {r_e/r_c:>12.3f}x", flush=True)
        print(f"\n  decode ratios vs base:", flush=True)
        print(f"    retrofit_eager    / base_eager    = {r_e/b_e:.3f}x", flush=True)
        print(f"    retrofit_compiled / base_compiled = {r_c/b_c:.3f}x  <-- writeable iso-cost (decode)", flush=True)


if __name__ == "__main__":
    main()
