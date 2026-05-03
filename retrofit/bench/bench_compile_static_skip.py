"""Compile + STATIC skip smoke: probe whether a compile-friendly skip-dispatch
path can beat base. Uses `_active_skip_blocks` (Python set, traced at Python
time) so dynamo never hits the `.item()` graph-break that kills
bench_compile_decode_skip.py.

This is a sanity check, NOT the final algorithm: it skips a fixed set of
blocks every step instead of the per-token alpha gate. The number it produces
is the lower bound for what a properly-fixed dynamic skip can achieve once we
move the alpha threshold off the .item() path.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch
import torch._dynamo

torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.accumulated_recompile_limit = 1024

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

    cache_pos_pre = torch.arange(prefill_len, device=device)
    out = m(input_ids=prefill_ids, past_key_values=cache,
            cache_position=cache_pos_pre, use_cache=True)
    cur = out.logits[:, -1:].argmax(-1).clone()
    cur_pos = prefill_len

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_decode):
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
    return statistics.median(ts) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--state-path", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--seq-lens", default="1024,2048")
    ap.add_argument("--n-decode", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timed", type=int, default=10)
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--max-cache-len", type=int, default=4096)
    ap.add_argument("--skip-blocks", default="1,4",
                    help="comma-separated block indices to skip (static)")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    print(f"[static-skip] mode={args.compile_mode}, skip_blocks={args.skip_blocks}", flush=True)

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

    skip_set = set(int(x) for x in args.skip_blocks.split(",")) if args.skip_blocks else set()
    print(f"[static-skip] num_blocks={kw['num_blocks']}, static skip_set={skip_set}", flush=True)

    base_compiled = torch.compile(base, mode=args.compile_mode, dynamic=False)

    # No-skip retrofit (baseline reference)
    retro._active_skip_blocks = set()
    retro._dynamic_skip_config = None
    retro_noskip_compiled = torch.compile(retro.base_model, mode=args.compile_mode, dynamic=False)

    base_cache = make_static_cache(base, 1, args.max_cache_len, device, dtype)
    retro_noskip_cache = make_static_cache(retro.base_model, 1, args.max_cache_len, device, dtype)
    retro_skip_cache = make_static_cache(retro.base_model, 1, args.max_cache_len, device, dtype)

    for seq in [int(x) for x in args.seq_lens.split(",")]:
        ids = torch.randint(0, 100000, (1, seq), device=device)
        print(f"\n=== prefill seq_len = {seq}, decode = {args.n_decode} tok ===", flush=True)

        # 1) base
        b_c = bench(base_compiled, ids, args.n_decode, base_cache, args.warmup, args.timed) / args.n_decode

        # 2) retrofit no-skip
        retro._active_skip_blocks = set()
        retro._dynamic_skip_config = None
        rf_c = bench(retro_noskip_compiled, ids, args.n_decode, retro_noskip_cache, args.warmup, args.timed) / args.n_decode

        # 3) retrofit + static skip
        retro._active_skip_blocks = set(skip_set)
        retro._dynamic_skip_config = None
        # Compile a fresh module so dynamo doesn't carry over noskip traces.
        retro_skip_compiled = torch.compile(retro.base_model, mode=args.compile_mode, dynamic=False)
        rs_c = bench(retro_skip_compiled, ids, args.n_decode, retro_skip_cache, args.warmup, args.timed) / args.n_decode

        print(f"  {'config':40s} {'compiled (ms/tok)':>20s} {'vs base':>12s}", flush=True)
        print(f"  {'TRUE base':40s} {b_c:>20.3f} {'1.000x':>12s}", flush=True)
        print(f"  {'VLM retrofit (no skip)':40s} {rf_c:>20.3f} {rf_c/b_c:>11.3f}x", flush=True)
        print(f"  {'VLM retrofit + STATIC skip':40s} {rs_c:>20.3f} {rs_c/b_c:>11.3f}x", flush=True)


if __name__ == "__main__":
    main()
