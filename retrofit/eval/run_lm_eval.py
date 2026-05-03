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
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--state-path", default=None,
                   help="If provided, apply the AttnRes retrofit before evaluation.")
    p.add_argument("--num-blocks", type=int, default=None,
                   help="Fallback if config not in state file.")
    p.add_argument("--adapter-rank", type=int, default=None)
    p.add_argument("--skip-layers", default="",
                   help="Comma-separated decoder-layer indices to replace with "
                        "identity (Gromov / ShortGPT static-pruning baseline). "
                        "Mutually exclusive with --state-path.")
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
    model.eval()

    if args.state_path and args.skip_layers:
        raise ValueError("--state-path and --skip-layers are mutually exclusive")

    if args.state_path:
        print(f"[run_lm_eval] applying retrofit from {args.state_path}", flush=True)
        load_retrofit(
            model,
            args.state_path,
            num_blocks_override=args.num_blocks,
            adapter_rank_override=args.adapter_rank,
        )
        # The retrofit monkey-patches model.model.language_model.forward in
        # place, so subsequent .forward() / generate() goes through AttnRes.

    if args.skip_layers:
        skip_set = sorted(int(x) for x in args.skip_layers.split(",") if x.strip())
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
        "state_path": args.state_path,
        "label": args.label,
        "tasks": tasks,
        "limit": args.limit,
        "results": results.get("results", {}) if isinstance(results, dict) else None,
        "config": results.get("config", {}) if isinstance(results, dict) else None,
    }
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
