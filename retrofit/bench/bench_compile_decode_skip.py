"""Compile-mode T=1 decode latency WITH dynamic skip enabled.

Goal: produce the q-sweep latency table the paper needs — compile + dyn-skip,
varying q ∈ {0.85, 0.99}, in the deployment-relevant T=1 decode regime, on the
canonical L=4 partition.

Implementation: same StaticCache + mode=default approach as
bench_compile_decode_static.py, but the retrofit's `_dynamic_skip_config` is
left enabled. The skip rule's `.item()` will graph-break dynamo per block, but
each side of the break is still compiled — we lose CUDA graphs but keep
Inductor fusion. Threshold calibration is done on LAMBADA-text (same as
bench_cache_regime.py), per (model, q).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict

import torch
import torch._dynamo

# Per-block_idx + per-layer specialization is needed for the L=4 partition
# (up to 9 blocks at 4B + 28-36 attention layers + per-step shape variants).
# Default cap is 8 — far too low; the fallback-to-eager is what made compile
# look like 4× of base in our smoke test. Bump well above the maximum needed.
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.accumulated_recompile_limit = 1024

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer
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
def calibrate_thresholds(model, tok, device, n=32, seq_len=512, q=0.85):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test").select(range(n, n + n))
    per_block = defaultdict(list)
    for ex in ds:
        ids = tok.encode(ex["text"].strip(), add_special_tokens=False)[:seq_len]
        out = model(input_ids=torch.tensor([ids], device=device), return_alpha=True)
        for i, tr in enumerate(out.skip_trace or []):
            if tr.get("w_recent") is not None:
                per_block[i].append(tr["w_recent"])
    thr = {}
    for b, vals in per_block.items():
        if vals:
            vs = sorted(vals)
            thr[b] = vs[int(q * (len(vs) - 1))]
    return thr


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
    ap.add_argument("--qs", default="0.85,0.99",
                    help="comma list of q values to sweep")
    ap.add_argument("--eligible", default="1,4",
                    help="eligible block indices (P), comma-separated")
    ap.add_argument("--max-skips", type=int, default=2)
    ap.add_argument("--skip-calibration", action="store_true",
                    help="Use hardcoded thr=0.5 instead of LAMBADA calibration (debug)")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    print(f"[compile-decode-skip] mode={args.compile_mode}", flush=True)
    print(f"[compile-decode-skip] model={args.model_path}", flush=True)
    print(f"[compile-decode-skip] state={args.state_path}", flush=True)
    print(f"[compile-decode-skip] qs={args.qs}, eligible={args.eligible}, max_skips={args.max_skips}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
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

    print(f"[compile-decode-skip] num_blocks={kw['num_blocks']}, γ={retro.gamma.detach().cpu().tolist()}", flush=True)

    eligible = set(int(x) for x in args.eligible.split(","))
    qs = [float(x) for x in args.qs.split(",")]

    # Calibrate thresholds once per q (uses retrofit unwrapped — no compile needed)
    thr_per_q = {}
    if args.skip_calibration:
        # Hardcoded fallback for smoke / debug — uniform per-block.
        for q in qs:
            thr_per_q[q] = {b: 0.5 for b in range(kw["num_blocks"])}
        print(f"[compile-decode-skip] SKIPPING calibration; thr = uniform 0.5", flush=True)
    else:
        for q in qs:
            retro._dynamic_skip_config = None
            retro._active_skip_blocks = set()
            thr = calibrate_thresholds(retro, tok, device, q=q)
            thr_per_q[q] = thr
            print(f"[compile-decode-skip] q={q}: per-block thr = {thr}", flush=True)

    base_compiled = torch.compile(base, mode=args.compile_mode, dynamic=False)
    retro_compiled = torch.compile(retro.base_model, mode=args.compile_mode, dynamic=False)

    base_cache = make_static_cache(base, 1, args.max_cache_len, device, dtype)
    retro_cache = make_static_cache(retro.base_model, 1, args.max_cache_len, device, dtype)
    retro_skip_cache = make_static_cache(retro.base_model, 1, args.max_cache_len, device, dtype)

    for seq in [int(x) for x in args.seq_lens.split(",")]:
        ids = torch.randint(0, 100000, (1, seq), device=device)
        print(f"\n=== prefill seq_len = {seq}, decode = {args.n_decode} tok (StaticCache, compile={args.compile_mode}) ===", flush=True)

        # Base compiled
        retro._dynamic_skip_config = None
        retro._active_skip_blocks = set()
        b_c = bench(base_compiled, ids, args.n_decode, base_cache, args.warmup, args.timed) / args.n_decode

        # Retrofit no-skip compiled
        rf_c = bench(retro_compiled, ids, args.n_decode, retro_cache, args.warmup, args.timed) / args.n_decode

        rs_results = {}
        for q in qs:
            retro._dynamic_skip_config = dict(thresholds=thr_per_q[q],
                                              eligible_blocks=eligible,
                                              max_skips=args.max_skips)
            rs_c = bench(retro_compiled, ids, args.n_decode, retro_skip_cache, args.warmup, args.timed) / args.n_decode
            rs_results[q] = rs_c

        print(f"  {'config':40s} {'compiled (ms/tok)':>20s} {'vs base':>12s}", flush=True)
        print(f"  {'TRUE base':40s} {b_c:>20.3f} {'1.000x':>12s}", flush=True)
        print(f"  {'VLM retrofit (no skip)':40s} {rf_c:>20.3f} {rf_c/b_c:>11.3f}x", flush=True)
        for q in qs:
            label = f"VLM retrofit + dyn-skip (q={q})"
            rs_c = rs_results[q]
            print(f"  {label:40s} {rs_c:>20.3f} {rs_c/b_c:>11.3f}x", flush=True)
        print(f"\n  q-sweep summary (decode/tok ms, vs base_compiled):", flush=True)
        print(f"    base_compiled        = {b_c:.3f} ms", flush=True)
        print(f"    retrofit (no skip)   = {rf_c:.3f} ms ({rf_c/b_c:.3f}x)", flush=True)
        for q in qs:
            rs_c = rs_results[q]
            print(f"    retrofit + skip q={q}  = {rs_c:.3f} ms ({rs_c/b_c:.3f}x)  <-- iso-cost target", flush=True)


if __name__ == "__main__":
    main()
