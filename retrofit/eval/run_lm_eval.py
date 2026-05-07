"""Run lm-evaluation-harness benchmarks on Qwen3-VL base or retrofit.

Loads Qwen3VLForConditionalGeneration directly (since Qwen3-VL is not registered
under AutoModelForCausalLM), optionally wraps it with the AttnRes retrofit
(monkey-patching the language_model.forward), then constructs an HFLM and calls
``lm_eval.simple_evaluate``.

Usage::
    python run_lm_eval.py \
        --model-path /home/user01/Minko/models/Qwen3-VL-2B \
        --state-path retrofit/outputs/H_2B_r256_10k_L4_v3/retrofit_attnres_state.pt \
        --tasks piqa,mmlu,openbookqa,arc_easy,arc_challenge,lambada_openai,hellaswag \
        --gpu 0 \
        --output /home/user01/Minko/reskip2/reskip/retrofit/outputs/lm_eval_v3/2B_retrofit \
        --batch-size 8

Tasks are passed verbatim to lm-eval-harness. For ``--state-path``-less
invocations the base model is evaluated.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from peft import PeftModel

PROJECT = "/home/user01/Minko/reskip2/reskip"
sys.path.insert(0, PROJECT + "/retrofit")
sys.path.insert(0, PROJECT + "/retrofit/eval")


def load_retrofit(model, state_path, num_blocks_override=None, adapter_rank_override=None):
    from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit

    ck = torch.load(state_path, map_location="cpu")
    cfg = ck.get("config", {})
    kwargs = {"num_blocks": int(cfg.get("num_blocks", num_blocks_override or 14))}
    ar = cfg.get("adapter_rank", adapter_rank_override or 256)
    kwargs["adapter_rank"] = int(ar)
    no_adapter = cfg.get("no_adapter", not ck.get("adapters"))
    if no_adapter:
        kwargs["no_adapter"] = True
        kwargs.pop("adapter_rank", None)

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    wrapper = Qwen3VLAttnResRetrofit(model, **kwargs).to(device=device, dtype=dtype)
    wrapper.router.load_state_dict(
        {k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()}
    )
    if not no_adapter:
        wrapper.adapters.load_state_dict(
            {k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()}
        )
    wrapper.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
    wrapper.eval()
    gmax = float(wrapper.gamma.detach().abs().max())
    gmean = float(wrapper.gamma.detach().mean())
    print(
        f"[retrofit] state loaded: num_blocks={kwargs['num_blocks']}, "
        f"adapter_rank={kwargs.get('adapter_rank', 'Identity')}, "
        f"γ_max={gmax:.3f}, γ_mean={gmean:+.3f}",
        flush=True,
    )
    return wrapper


def _parse_ints(value: str):
    if not value:
        return []
    sep = "|" if "|" in value else ","
    return [int(x) for x in value.split(sep) if x.strip()]


@torch.no_grad()
def calibrate_retrofit_thresholds(wrapper, tok, device, q, positions, n_samples=32, seq_len=512):
    from collections import defaultdict
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    start = min(n_samples, max(len(ds) - n_samples, 0))
    ds = ds.select(range(start, min(start + n_samples, len(ds))))
    samples = defaultdict(list)
    for ex in ds:
        ids = tok.encode(ex["text"].strip(), add_special_tokens=False)[:seq_len]
        if not ids:
            continue
        inp = torch.tensor([ids], device=device)
        out = wrapper(input_ids=inp, return_alpha=True)
        for trace in out.skip_trace or []:
            block_idx = int(trace["block_idx"])
            if positions and block_idx not in positions:
                continue
            w = trace.get("w_recent")
            if w is not None:
                samples[block_idx].append(float(w))
    thresholds = {}
    for block_idx, vals in samples.items():
        vals = sorted(vals)
        if vals:
            thresholds[block_idx] = vals[int(float(q) * (len(vals) - 1))]
    return thresholds, {int(k): v for k, v in samples.items()}


def install_mod_proxy(model, layers_to_patch, keep_ratio, router_state_path=None):
    import math
    import types
    import torch.nn as nn

    layers = model.model.language_model.layers
    stats = {
        "layers": sorted(layers_to_patch),
        "keep_ratio": float(keep_ratio),
        "calls": 0,
        "tokens": 0,
        "tokens_bypassed": 0,
        "per_layer_tokens_bypassed": {i: 0 for i in sorted(layers_to_patch)},
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
        def _mod_forward(self, hidden_states, *a, _idx=i, _orig=original_forward, **kw):
            out = _orig(hidden_states, *a, **kw)
            h = out[0] if isinstance(out, tuple) else out
            if h.ndim != 3:
                return out
            bsz, seqlen, _ = h.shape
            n_tok = bsz * seqlen
            keep = max(1, min(seqlen, int(math.ceil(seqlen * float(keep_ratio)))))
            if router_state_path:
                scores = routers[str(_idx)](hidden_states).squeeze(-1)
            else:
                # Zero-training MoD proxy: high hidden-state norm tokens get the
                # transformed layer output; low-score tokens bypass this layer.
                scores = hidden_states.float().pow(2).mean(dim=-1)
            top_idx = scores.topk(k=keep, dim=1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, top_idx, True)
            mixed = torch.where(mask.unsqueeze(-1), h, hidden_states)
            bypassed = int((~mask).sum().item())
            stats["calls"] += 1
            stats["tokens"] += n_tok
            stats["tokens_bypassed"] += bypassed
            stats["per_layer_tokens_bypassed"][_idx] += bypassed
            if isinstance(out, tuple):
                return (mixed,) + out[1:]
            return mixed
        layers[i].forward = types.MethodType(_mod_forward, layers[i])
    if router_state_path:
        routers.load_state_dict(torch.load(router_state_path, map_location=device))
    model.mod_routers = routers
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-adapter-path", default="",
                   help="Optional PEFT adapter to load on top of --model-path "
                        "before applying baseline skip/MoD mechanisms.")
    p.add_argument("--state-path", default=None,
                   help="If provided, apply the AttnRes retrofit before evaluation.")
    p.add_argument("--num-blocks", type=int, default=None,
                   help="Fallback if config not in state file.")
    p.add_argument("--adapter-rank", type=int, default=None)
    p.add_argument("--skip-layers", default="",
                   help="Comma-separated decoder-layer indices to replace with "
                        "identity (Gromov / ShortGPT static-pruning baseline). "
                        "Mutually exclusive with --state-path.")
    p.add_argument("--early-exit-layer", type=int, default=None,
                   help="LayerSkip/CALM-style untrained early-exit proxy: "
                        "run layers [0, early_exit_layer) and identity-skip "
                        "all later decoder layers. Mutually exclusive with --state-path.")
    p.add_argument("--random-skip-layers", default="",
                   help="Pipe/comma-separated decoder-layer candidates for a random "
                        "static-pruning baseline on the base model.")
    p.add_argument("--random-skip-p", type=float, default=0.0,
                   help="Per-forward probability for each candidate layer in "
                        "--random-skip-layers.")
    p.add_argument("--random-seed", type=int, default=1234)
    p.add_argument("--mod-layers", default="",
                   help="Pipe/comma-separated decoder layers for a zero-training "
                        "MoD-style token-capacity proxy, or trained MoD if "
                        "--mod-router-path is set.")
    p.add_argument("--mod-keep-ratio", type=float, default=0.8,
                   help="Fraction of tokens that execute each --mod-layers layer.")
    p.add_argument("--mod-router-path", default="",
                   help="Optional trained MoD router state from "
                        "train_qwen3vl_depth_baseline.py.")
    p.add_argument("--retrofit-static-blocks", default="",
                   help="Pipe/comma-separated AttnRes block ids to force-skip on "
                        "a retrofit state.")
    p.add_argument("--retrofit-random-blocks", default="",
                   help="Pipe/comma-separated AttnRes block ids for random block skip "
                        "on a retrofit state.")
    p.add_argument("--retrofit-random-p", type=float, default=0.0)
    p.add_argument("--retrofit-random-max-skips", type=int, default=1)
    p.add_argument("--retrofit-dynamic-q", type=float, default=None,
                   help="If set with --state-path, calibrate recent_weight_gt "
                        "thresholds on LAMBADA prefixes and enable dynamic ReSkip.")
    p.add_argument("--retrofit-dynamic-positions", default="",
                   help="Pipe/comma-separated eligible retrofit block ids for dynamic ReSkip.")
    p.add_argument("--retrofit-dynamic-max-skips", type=int, default=1)
    p.add_argument("--retrofit-dynamic-calib-n", type=int, default=32)
    p.add_argument("--tasks", required=True,
                   help="Comma-separated lm-eval-harness task names.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=str, default="8")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = f"cuda:{args.gpu}"

    print(f"[run_lm_eval] loading {args.model_path} on {device}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=device
    )
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if args.lora_adapter_path:
        print(f"[run_lm_eval] loading LoRA adapter {args.lora_adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, args.lora_adapter_path).to(device)
    model.eval()

    if args.state_path and (args.skip_layers or args.early_exit_layer is not None or args.random_skip_layers or args.mod_layers):
        raise ValueError("--state-path is mutually exclusive with base skip/early-exit/random layer baselines")
    active_retrofit_modes = sum(
        bool(x) for x in [
            args.retrofit_static_blocks,
            args.retrofit_random_blocks,
            args.retrofit_dynamic_q is not None,
        ]
    )
    if active_retrofit_modes > 1:
        raise ValueError("Choose only one retrofit skip mode: static, random, or dynamic")

    retrofit_wrapper = None
    if args.state_path:
        print(f"[run_lm_eval] applying retrofit from {args.state_path}", flush=True)
        retrofit_wrapper = load_retrofit(
            model,
            args.state_path,
            num_blocks_override=args.num_blocks,
            adapter_rank_override=args.adapter_rank,
        )
        # The retrofit monkey-patches model.model.language_model.forward in
        # place, so subsequent .forward() / generate() goes through AttnRes.
        if args.retrofit_static_blocks:
            blocks = set(_parse_ints(args.retrofit_static_blocks))
            retrofit_wrapper._active_skip_blocks = blocks
            retrofit_wrapper.reset_skip_stats()
            print(f"[run_lm_eval] retrofit static block skip: {sorted(blocks)}", flush=True)
        if args.retrofit_random_blocks:
            blocks = set(_parse_ints(args.retrofit_random_blocks))
            retrofit_wrapper._random_skip_config = {
                "eligible_blocks": blocks,
                "p": float(args.retrofit_random_p),
                "max_skips": int(args.retrofit_random_max_skips),
                "seed": int(args.random_seed),
                "rng": random.Random(int(args.random_seed)),
            }
            retrofit_wrapper.reset_skip_stats()
            print(
                f"[run_lm_eval] retrofit random block skip: blocks={sorted(blocks)} "
                f"p={args.retrofit_random_p} M={args.retrofit_random_max_skips} "
                f"seed={args.random_seed}",
                flush=True,
            )
        if args.retrofit_dynamic_q is not None:
            positions = set(_parse_ints(args.retrofit_dynamic_positions))
            thresholds, samples = calibrate_retrofit_thresholds(
                retrofit_wrapper,
                tok,
                device,
                q=args.retrofit_dynamic_q,
                positions=positions,
                n_samples=args.retrofit_dynamic_calib_n,
            )
            retrofit_wrapper._dynamic_skip_config = {
                "strategy": "recent_weight_gt",
                "thresholds": thresholds,
                "eligible_blocks": positions if positions else None,
                "max_skips": args.retrofit_dynamic_max_skips,
            }
            retrofit_wrapper.reset_skip_stats()
            print(
                "[run_lm_eval] retrofit dynamic ReSkip: "
                f"q={args.retrofit_dynamic_q} P={sorted(positions) if positions else 'all'} "
                f"M={args.retrofit_dynamic_max_skips} thresholds={thresholds}",
                flush=True,
            )

    base_skip_layers = args.skip_layers
    if args.early_exit_layer is not None:
        n_layers = len(model.model.language_model.layers)
        if args.early_exit_layer < 0 or args.early_exit_layer > n_layers:
            raise ValueError(f"--early-exit-layer must be in [0, {n_layers}]")
        base_skip_layers = ",".join(str(i) for i in range(args.early_exit_layer, n_layers))
        print(
            f"[run_lm_eval] early-exit proxy: exit_layer={args.early_exit_layer}, "
            f"identity layers={base_skip_layers}",
            flush=True,
        )

    if base_skip_layers:
        skip_set = sorted(_parse_ints(base_skip_layers))
        layers = model.model.language_model.layers
        import types
        for i in skip_set:
            if i < 0 or i >= len(layers):
                raise ValueError(f"skip_layer {i} out of range [0, {len(layers)})")
            def _identity(self, hidden_states, *a, **kw):
                return hidden_states
            layers[i].forward = types.MethodType(_identity, layers[i])
        print(
            f"[run_lm_eval] static-pruning: dropped {len(skip_set)}/{len(layers)} "
            f"layers: {skip_set}",
            flush=True,
        )

    if args.random_skip_layers:
        rng = random.Random(args.random_seed)
        random_set = set(_parse_ints(args.random_skip_layers))
        layers = model.model.language_model.layers
        import types
        stats = {"calls": 0, "skips": 0, "per_layer": {i: 0 for i in sorted(random_set)}}
        for i in sorted(random_set):
            if i < 0 or i >= len(layers):
                raise ValueError(f"random_skip_layer {i} out of range [0, {len(layers)})")
            original_forward = layers[i].forward
            def _maybe_identity(self, hidden_states, *a, _idx=i, _orig=original_forward, **kw):
                stats["calls"] += 1
                if rng.random() < float(args.random_skip_p):
                    stats["skips"] += 1
                    stats["per_layer"][_idx] += 1
                    return hidden_states
                return _orig(hidden_states, *a, **kw)
            layers[i].forward = types.MethodType(_maybe_identity, layers[i])
        print(
            f"[run_lm_eval] random layer skip: layers={sorted(random_set)} "
            f"p={args.random_skip_p} seed={args.random_seed}",
            flush=True,
        )

    mod_stats = None
    if args.mod_layers:
        mod_set = set(_parse_ints(args.mod_layers))
        mod_stats = install_mod_proxy(
            model,
            mod_set,
            args.mod_keep_ratio,
            router_state_path=args.mod_router_path or None,
        )
        print(
            f"[run_lm_eval] MoD {'trained' if args.mod_router_path else 'proxy'}: "
            f"layers={sorted(mod_set)} keep_ratio={args.mod_keep_ratio}",
            flush=True,
        )

    # Build HFLM around the loaded HF model object.
    from lm_eval.models.huggingface import HFLM
    import lm_eval

    lm = HFLM(
        pretrained=model,
        tokenizer=tok,
        backend="causal",
        batch_size=args.batch_size,
        device=device,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[run_lm_eval] tasks: {tasks}", flush=True)

    eval_kwargs = dict(
        model=lm,
        tasks=tasks,
        log_samples=False,
        verbosity="INFO",
    )
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit

    results = lm_eval.simple_evaluate(**eval_kwargs)

    summary = {
        "model_path": args.model_path,
        "lora_adapter_path": args.lora_adapter_path or None,
        "mod_router_path": args.mod_router_path or None,
        "state_path": args.state_path,
        "label": args.label,
        "tasks": tasks,
        "limit": args.limit,
        "results": results.get("results", {}) if isinstance(results, dict) else None,
        "config": results.get("config", {}) if isinstance(results, dict) else None,
    }
    if retrofit_wrapper is not None:
        summary["skip_stats"] = retrofit_wrapper.get_skip_stats()
        dyn_cfg = getattr(retrofit_wrapper, "_dynamic_skip_config", None)
        if dyn_cfg is not None:
            summary["retrofit_dynamic_config"] = {
                "strategy": dyn_cfg.get("strategy"),
                "thresholds": {str(k): v for k, v in (dyn_cfg.get("thresholds") or {}).items()},
                "eligible_blocks": (
                    sorted(dyn_cfg.get("eligible_blocks"))
                    if dyn_cfg.get("eligible_blocks") is not None
                    else None
                ),
                "max_skips": dyn_cfg.get("max_skips"),
            }
        rnd_cfg = getattr(retrofit_wrapper, "_random_skip_config", None)
        if rnd_cfg is not None:
            summary["retrofit_random_config"] = {
                k: (sorted(v) if isinstance(v, set) else v)
                for k, v in rnd_cfg.items()
                if k != "rng"
            }
        summary["retrofit_static_blocks"] = sorted(getattr(retrofit_wrapper, "_active_skip_blocks", set()))
    if args.random_skip_layers:
        summary["random_skip_stats"] = stats
    if mod_stats is not None:
        summary["mod_proxy_stats"] = mod_stats
    out_path = Path(args.output) / "summary.json"
    out_path.write_text(json.dumps(summary, default=str, indent=2))
    print(f"[run_lm_eval] summary -> {out_path}", flush=True)

    # Pretty print the per-task headline metric to stdout.
    print("\n=== TASK RESULTS ===")
    for task_name, task_res in (results.get("results", {}) or {}).items():
        if isinstance(task_res, dict):
            metrics = {k: v for k, v in task_res.items() if k != "alias"}
            print(f"  {task_name}: {metrics}")


if __name__ == "__main__":
    main()
