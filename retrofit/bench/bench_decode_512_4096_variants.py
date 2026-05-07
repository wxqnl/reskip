"""Decode-speed benchmark for Qwen3-VL-2B skip baselines.

Runs a strict text-side decode loop:
  - prefill prompt length: configurable, default 512 tokens
  - decode length: configurable, default 4096 one-token steps
  - KV cache: StaticCache, preallocated to prefill + decode + 8

This avoids `generate()` EOS early-stop so every variant performs the same
number of decode steps. It writes one JSON per variant for appendix tables.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.cache_utils import StaticCache
from peft import PeftModel

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


def _parse_ints(value: str):
    if not value:
        return []
    sep = "|" if "|" in str(value) else ","
    return [int(x) for x in str(value).split(sep) if str(x).strip()]


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


def prompt_ids(tok, length, device):
    text = (
        "This is a deterministic decode-speed benchmark prompt for measuring "
        "Qwen3-VL text-side cache latency. "
    )
    ids = tok.encode(text * max(1, math.ceil(length / 16)), add_special_tokens=False)
    if len(ids) < length:
        ids = (ids * math.ceil(length / max(1, len(ids))))[:length]
    else:
        ids = ids[:length]
    return torch.tensor([ids], device=device, dtype=torch.long)


def load_retrofit(model_path, state_path, device, dtype):
    base = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype).to(device).eval()
    ck = torch.load(state_path, map_location="cpu")
    cfg = ck.get("config", {})
    kwargs = {"num_blocks": int(cfg.get("num_blocks", 7))}
    if "adapter_rank" in cfg:
        kwargs["adapter_rank"] = int(cfg["adapter_rank"])
    if cfg.get("no_adapter", False):
        kwargs["no_adapter"] = True
        kwargs.pop("adapter_rank", None)

    retro = Qwen3VLAttnResRetrofit(base, **kwargs).to(device=device, dtype=dtype).eval()
    retro.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    if not kwargs.get("no_adapter", False):
        retro.adapters.load_state_dict(
            {k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()}
        )
    retro.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
    return retro


def install_identity_layers(model, layers_to_skip):
    import types

    layers = model.model.language_model.layers
    for i in sorted(layers_to_skip):
        if i < 0 or i >= len(layers):
            raise ValueError(f"skip_layer {i} out of range [0, {len(layers)})")

        def _identity(self, hidden_states, *args, **kwargs):
            return hidden_states

        layers[i].forward = types.MethodType(_identity, layers[i])


def install_mod_proxy(model, layers_to_patch, keep_ratio, router_state_path=None):
    import types

    layers = model.model.language_model.layers
    stats = {
        "layers": sorted(layers_to_patch),
        "keep_ratio": float(keep_ratio),
        "calls": 0,
        "tokens": 0,
        "tokens_bypassed": 0,
        "per_layer_tokens_bypassed": {str(i): 0 for i in sorted(layers_to_patch)},
    }
    routers = nn.ModuleDict()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    hidden_size = model.config.text_config.hidden_size
    for i in sorted(layers_to_patch):
        if i < 0 or i >= len(layers):
            raise ValueError(f"mod_layer {i} out of range [0, {len(layers)})")
        router = nn.Linear(hidden_size, 1, bias=False).to(device=device, dtype=dtype)
        nn.init.normal_(router.weight, std=0.02)
        routers[str(i)] = router
        original_forward = layers[i].forward

        def _mod_forward(self, hidden_states, *args, _idx=i, _orig=original_forward, **kwargs):
            out = _orig(hidden_states, *args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            if h.ndim != 3:
                return out
            bsz, seqlen, _ = h.shape
            keep = max(1, min(seqlen, int(math.ceil(seqlen * float(keep_ratio)))))
            if router_state_path:
                scores = routers[str(_idx)](hidden_states).squeeze(-1)
            else:
                scores = hidden_states.float().pow(2).mean(dim=-1)
            top_idx = scores.topk(k=keep, dim=1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, top_idx, True)
            mixed = torch.where(mask.unsqueeze(-1), h, hidden_states)
            bypassed = int((~mask).sum().item())
            stats["calls"] += 1
            stats["tokens"] += int(bsz * seqlen)
            stats["tokens_bypassed"] += bypassed
            stats["per_layer_tokens_bypassed"][str(_idx)] += bypassed
            if isinstance(out, tuple):
                return (mixed,) + out[1:]
            return mixed

        layers[i].forward = types.MethodType(_mod_forward, layers[i])
    if router_state_path:
        state = torch.load(router_state_path, map_location=device)
        routers.load_state_dict(state)
    model.mod_routers = routers
    return stats


def build_model(args, device, dtype):
    if args.variant in {
        "retrofit_full",
        "reskip",
        "retrofit_static_b4",
        "retrofit_random_b4",
    }:
        if not args.state_path:
            raise ValueError("--state-path is required for retrofit variants")
        retro = load_retrofit(args.model_path, args.state_path, device, dtype)
        retro.reset_skip_stats()
        if args.variant == "reskip":
            cfg = json.load(open(args.dynamic_skip_config_path))
            retro._dynamic_skip_config = {
                "strategy": cfg.get("strategy", "recent_weight_gt"),
                "thresholds": {int(k): float(v) for k, v in cfg["thresholds"].items()},
                "eligible_blocks": set(int(x) for x in cfg.get("eligible_blocks", [])),
                "max_skips": int(cfg.get("max_skips", 1)),
            }
        elif args.variant == "retrofit_static_b4":
            retro._active_skip_blocks = {4}
        elif args.variant == "retrofit_random_b4":
            retro._random_skip_config = {
                "eligible_blocks": {4},
                "p": float(args.random_skip_p),
                "max_skips": 1,
                "seed": int(args.seed),
                "rng": random.Random(int(args.seed)),
            }
        return retro.base_model, retro, None

    model = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=dtype).to(device).eval()
    if args.lora_adapter_path:
        model = PeftModel.from_pretrained(model, args.lora_adapter_path).to(device).eval()
    aux_stats = None
    if args.variant == "base":
        return model, None, aux_stats
    if args.variant == "gromov_drop4":
        install_identity_layers(model, {23, 24, 25, 26})
        return model, None, aux_stats
    if args.variant == "gromov_drop8":
        install_identity_layers(model, {12, 13, 14, 22, 23, 24, 25, 26})
        return model, None, aux_stats
    if args.variant == "layerskip_exit24":
        install_identity_layers(model, set(range(24, 28)))
        return model, None, aux_stats
    if args.variant == "mod_proxy":
        aux_stats = install_mod_proxy(
            model,
            {12, 13, 14, 15},
            args.mod_keep_ratio,
            router_state_path=args.mod_router_path or None,
        )
        return model, None, aux_stats
    raise ValueError(f"unknown variant: {args.variant}")


@torch.no_grad()
def decode_loop(model, ids, decode_tokens, cache):
    reset_static_cache(cache)
    device = ids.device
    prefill_len = ids.shape[1]
    cache_pos_pre = torch.arange(prefill_len, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, past_key_values=cache, cache_position=cache_pos_pre, use_cache=True)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    cur = out.logits[:, -1:].argmax(-1).clone()
    cur_pos = prefill_len

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(decode_tokens):
        cp = torch.tensor([cur_pos], device=device)
        out = model(input_ids=cur, past_key_values=cache, cache_position=cp, use_cache=True)
        cur = out.logits[:, -1:].argmax(-1).clone()
        cur_pos += 1
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t1
    return prefill_s, decode_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=[
        "base",
        "retrofit_full",
        "reskip",
        "retrofit_static_b4",
        "retrofit_random_b4",
        "mod_proxy",
        "official_layerskip",
        "official_calm",
        "official_mod",
        "gromov_drop4",
        "gromov_drop8",
        "layerskip_exit24",
    ])
    ap.add_argument("--model-path", default="/home/user01/Minko/models/Qwen3-VL-2B")
    ap.add_argument("--state-path", default="")
    ap.add_argument("--lora-adapter-path", default="")
    ap.add_argument("--mod-router-path", default="")
    ap.add_argument("--dynamic-skip-config-path", default="")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=4096)
    ap.add_argument("--warmup-decode-tokens", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--random-skip-p", type=float, default=0.553)
    ap.add_argument("--mod-keep-ratio", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    requested_variant = args.variant
    if args.variant == "official_layerskip":
        args.variant = "layerskip_exit24"
        if not args.lora_adapter_path:
            raise ValueError("--lora-adapter-path is required for official_layerskip")
    elif args.variant == "official_calm":
        # CALM trained adapter with the calibrated always-exit-at-24 operating
        # point. Dynamic confidence calibration is evaluated separately; this
        # path measures the skip-enabled decode speed.
        args.variant = "layerskip_exit24"
        if not args.lora_adapter_path:
            raise ValueError("--lora-adapter-path is required for official_calm")
    elif args.variant == "official_mod":
        args.variant = "mod_proxy"
        if not args.lora_adapter_path or not args.mod_router_path:
            raise ValueError("--lora-adapter-path and --mod-router-path are required for official_mod")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model_path)
    ids = prompt_ids(tok, args.prompt_len, device)
    model, retro, aux_stats = build_model(args, device, dtype)
    cache = make_static_cache(
        model,
        batch=1,
        max_len=args.prompt_len + args.decode_tokens + 8,
        device=device,
        dtype=dtype,
    )

    if args.warmup_decode_tokens > 0:
        decode_loop(model, ids, args.warmup_decode_tokens, cache)

    rows = []
    for rep in range(args.repeat):
        if retro is not None:
            retro.reset_skip_stats()
        torch.cuda.reset_peak_memory_stats(device)
        prefill_s, decode_s = decode_loop(model, ids, args.decode_tokens, cache)
        stats = retro.get_skip_stats() if retro is not None else None
        rows.append({
            "rep": rep,
            "prefill_ms": prefill_s * 1000.0,
            "decode_ms": decode_s * 1000.0,
            "decode_ms_per_token": decode_s * 1000.0 / args.decode_tokens,
            "decode_tokens_per_second": args.decode_tokens / decode_s,
            "skip_stats": stats,
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        })

    best = min(rows, key=lambda r: r["decode_ms_per_token"])
    payload = {
        "requested_variant": requested_variant,
        "variant": args.variant,
        "model_path": args.model_path,
        "state_path": args.state_path or None,
        "dynamic_skip_config_path": args.dynamic_skip_config_path or None,
        "prompt_len": args.prompt_len,
        "decode_tokens": args.decode_tokens,
        "dtype": str(dtype),
        "device": device,
        "repeat": args.repeat,
        "rows": rows,
        "best": best,
        "aux_stats": aux_stats,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
