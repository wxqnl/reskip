"""Train official-style depth baselines on Qwen3-VL with the retrofit data mix.

This script is for method-specific trained baselines, not zero-training proxies:

- layerskip: LayerSkip-style early-exit supervision with stochastic layer dropout.
- calm: CALM-style early-exit supervision; downstream eval chooses the
  confidence or fixed-exit operating point after training.
- mod: Mixture-of-Depths-style token router with a fixed capacity top-k and a
  straight-through gate.

All modes use the same v3 data mixer as the canonical retrofit runs. The base
backbone is frozen and LoRA is trained as the adaptation carrier, matching the
paper's low-cost setting and keeping trainable parameters in the retrofit range.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit/train")
from train_qwen3vl_attnres_retrofit import compute_assistant_mask, move_inputs
import data_v2


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def _parse_ints(value: str):
    if not value:
        return []
    sep = "|" if "|" in str(value) else ","
    return [int(x) for x in str(value).split(sep) if str(x).strip()]


def _language_model(model):
    obj = model
    # PeftModel -> base HF model.
    if hasattr(obj, "base_model") and hasattr(obj.base_model, "model"):
        obj = obj.base_model.model
    return obj.model.language_model


def _decoder_layers(model):
    return _language_model(model).layers


def _decoder_norm(model):
    return _language_model(model).norm


def _lm_head(model):
    obj = model
    if hasattr(obj, "base_model") and hasattr(obj.base_model, "model"):
        obj = obj.base_model.model
    return obj.lm_head


def _text_config(model):
    obj = model
    if hasattr(obj, "base_model") and hasattr(obj.base_model, "model"):
        obj = obj.base_model.model
    return obj.config.text_config


def ce_from_logits(logits, labels):
    shifted = torch.cat([labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shifted.reshape(-1),
        ignore_index=-100,
    )


def aux_ce_from_hidden(model, hidden, labels):
    h = _decoder_norm(model)(hidden)
    logits = _lm_head(model)(h)
    return ce_from_logits(logits, labels)


def build_stream(processor, args):
    spec = data_v2.parse_mix_string(args.data_mix)
    print(
        f"[depth-baseline] data-mix={args.data_mix} "
        f"weights={dict(zip(spec.names(), [round(w, 3) for w in spec.normed()]))}",
        flush=True,
    )

    def _stream():
        raw = data_v2.build_mixed_stream(spec, seed=args.seed)
        n_fail = 0
        while True:
            try:
                name, sample = next(raw)
                enc = data_v2.encode_sample(
                    sample, processor, args.max_seq, compute_assistant_mask
                )
            except Exception as e:
                n_fail += 1
                if n_fail % 50 == 1:
                    print(
                        f"[depth-baseline] stream_fail#{n_fail}: "
                        f"{type(e).__name__}: {str(e)[:160]}",
                        flush=True,
                    )
                raw = data_v2.build_mixed_stream(
                    spec, seed=args.seed + 1_000_000 + n_fail
                )
                continue
            if enc is None:
                continue
            inputs, labels = enc
            yield inputs, labels, data_v2.MODALITY.get(name, "text"), name

    return _stream()


def install_layer_dropout(model, min_layer: int, max_layer: int, p_drop: float):
    """LayerSkip-style stochastic layer dropout during training."""
    import types

    layers = _decoder_layers(model)
    max_layer = min(max_layer, len(layers) - 1)
    for i in range(min_layer, max_layer + 1):
        original_forward = layers[i].forward

        def _drop_forward(self, hidden_states, *args, _orig=original_forward, **kwargs):
            if self.training and torch.rand((), device=hidden_states.device).item() < p_drop:
                return hidden_states
            return _orig(hidden_states, *args, **kwargs)

        layers[i].forward = types.MethodType(_drop_forward, layers[i])


def install_mod_routers(model, layer_ids: list[int], keep_ratio: float):
    """MoD-style per-token capacity router with straight-through hard top-k."""
    import types

    cfg = _text_config(model)
    layers = _decoder_layers(model)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    routers = nn.ModuleDict()
    stats = {
        "layers": sorted(layer_ids),
        "keep_ratio": float(keep_ratio),
        "calls": 0,
        "tokens": 0,
        "tokens_bypassed": 0,
        "per_layer_tokens_bypassed": {str(i): 0 for i in sorted(layer_ids)},
    }

    for i in sorted(layer_ids):
        if i < 0 or i >= len(layers):
            raise ValueError(f"mod_layer {i} out of range [0, {len(layers)})")
        router = nn.Linear(cfg.hidden_size, 1, bias=False).to(device=device, dtype=dtype)
        nn.init.normal_(router.weight, std=0.02)
        routers[str(i)] = router
        original_forward = layers[i].forward

        def _mod_forward(self, hidden_states, *args, _idx=i, _orig=original_forward, **kwargs):
            out = _orig(hidden_states, *args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            if h.ndim != 3:
                return out
            bsz, seqlen, _ = h.shape
            n_tok = bsz * seqlen
            keep = max(1, min(seqlen, int(math.ceil(seqlen * keep_ratio))))
            scores = routers[str(_idx)](hidden_states).squeeze(-1)
            top_idx = scores.topk(k=keep, dim=1).indices
            hard = torch.zeros_like(scores)
            hard.scatter_(1, top_idx, 1.0)
            probs = torch.sigmoid(scores)
            gate = hard.detach() - probs.detach() + probs
            mixed = gate.unsqueeze(-1) * h + (1.0 - gate).unsqueeze(-1) * hidden_states
            bypassed = int((hard == 0).sum().item())
            stats["calls"] += 1
            stats["tokens"] += int(n_tok)
            stats["tokens_bypassed"] += bypassed
            stats["per_layer_tokens_bypassed"][str(_idx)] += bypassed
            if isinstance(out, tuple):
                return (mixed,) + out[1:]
            return mixed

        layers[i].forward = types.MethodType(_mod_forward, layers[i])

    # Register routers on the model so optimizers and state_dict can see them.
    model.mod_routers = routers
    return routers, stats


def make_model(args, device, dtype):
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=dtype).to(device)
    for p in model.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.target_modules.split(","),
        task_type=None,
    )
    model = get_peft_model(model, lora_cfg)
    if args.mode == "layerskip":
        install_layer_dropout(
            model,
            min_layer=args.layerdrop_min_layer,
            max_layer=args.layerdrop_max_layer,
            p_drop=args.layerdrop_p,
        )
    routers = None
    mod_stats = None
    if args.mode == "mod":
        routers, mod_stats = install_mod_routers(
            model, _parse_ints(args.mod_layers), args.mod_keep_ratio
        )
        for p in routers.parameters():
            p.requires_grad = True
    return model, routers, mod_stats


def train(args):
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path)
    model, routers, mod_stats = make_model(args, device, dtype)
    model.train()

    exit_layers = _parse_ints(args.exit_layers)
    if args.mode in {"layerskip", "calm"} and not exit_layers:
        raise ValueError("--exit-layers is required for layerskip/calm")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[depth-baseline] mode={args.mode} trainable={n_train/1e6:.2f}M "
        f"total={n_total/1e9:.2f}B exit_layers={exit_layers}",
        flush=True,
    )

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    stream = build_stream(processor, args)
    log_f = open(out_dir / "train.log", "a")
    meta = vars(args).copy()
    meta.update({"trainable_params": n_train, "total_params": n_total})
    with open(out_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    def lr_scale(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (
            1
            + math.cos(
                math.pi
                * min(
                    (step - args.warmup_steps)
                    / max(args.steps - args.warmup_steps, 1),
                    1.0,
                )
            )
        )

    ema = {}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        inputs, labels, kind, source = next(stream)
        inputs = move_inputs(inputs, device)
        labels = labels.unsqueeze(0).to(device)

        need_hidden = args.mode in {"layerskip", "calm"}
        out = model(**inputs, use_cache=False, output_hidden_states=need_hidden)
        final_ce = ce_from_logits(out.logits, labels)
        total = final_ce
        aux_loss = torch.zeros((), device=device, dtype=final_ce.dtype)

        if need_hidden:
            hidden_states = out.hidden_states
            aux_terms = []
            for layer_n in exit_layers:
                if layer_n < 1 or layer_n >= len(hidden_states):
                    raise ValueError(
                        f"exit layer {layer_n} invalid for {len(hidden_states)-1} decoder layers"
                    )
                aux_terms.append(aux_ce_from_hidden(model, hidden_states[layer_n], labels))
            if aux_terms:
                aux_loss = torch.stack(aux_terms).mean()
                total = total + args.aux_weight * aux_loss

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scale = lr_scale(step)
        for g in opt.param_groups:
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            g["lr"] = g["base_lr"] * scale
        opt.step()

        vals = {
            "loss": float(total.detach()),
            "final_ce": float(final_ce.detach()),
            "aux_ce": float(aux_loss.detach()),
        }
        for k, v in vals.items():
            ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - t0
            line = (
                f"step {step}/{args.steps} "
                f"loss={ema['loss']:.3f} final_ce={ema['final_ce']:.3f} "
                f"aux_ce={ema['aux_ce']:.3f} kind={kind} src={source} "
                f"T_len={inputs['input_ids'].shape[1]} lr_scale={scale:.3f} "
                f"elapsed={elapsed:.0f}s"
            )
            if mod_stats is not None and mod_stats["tokens"] > 0:
                frac = mod_stats["tokens_bypassed"] / max(mod_stats["tokens"], 1)
                line += f" mod_bypass={frac:.3f}"
            print(f"[depth-baseline] {line}", flush=True)
            log_f.write(line + "\n")
            log_f.flush()

    model.save_pretrained(out_dir / "lora_adapter")
    if routers is not None:
        torch.save(routers.state_dict(), out_dir / "mod_routers.pt")
        with open(out_dir / "mod_stats.json", "w") as f:
            json.dump(mod_stats, f, indent=2, sort_keys=True)
    log_f.close()
    print(f"[depth-baseline] saved {out_dir}", flush=True)
    # Some dataset/media extension threads can trip CPython finalization after
    # successful save in this environment. Exit immediately once artifacts are
    # durable so long official baseline jobs return a clean status.
    import os
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["layerskip", "calm", "mod"])
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--data-mix", default="v3")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--target-modules", default="q_proj,v_proj")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output-dir", required=True)

    # Early-exit / LayerSkip / CALM.
    ap.add_argument("--exit-layers", default="16,20,24")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--layerdrop-min-layer", type=int, default=8)
    ap.add_argument("--layerdrop-max-layer", type=int, default=27)
    ap.add_argument("--layerdrop-p", type=float, default=0.15)

    # MoD.
    ap.add_argument("--mod-layers", default="12,13,14,15")
    ap.add_argument("--mod-keep-ratio", type=float, default=0.8)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
