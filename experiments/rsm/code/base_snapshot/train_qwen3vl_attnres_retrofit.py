"""Training for Qwen3VLAttnResRetrofit (rechange design).

Follows RECHANGE_DESIGN_CN.md:
  - γ zero-init → identity-at-init
  - Freeze ALL base params; train only router + adapters + γ
  - Loss = L_task(full) + λ_kl · L_kl(skip‖teacher) + λ_ent · L_entropy
  - Optional L_hidden = ||x_n - h_n^teacher||²  (local regression)

Data: post-training mixture (UltraChat for text + LLaVA-Instruct-VSFT for
multimodal) — not just one distribution. Sampling controlled by --p-multimodal.

Skip sampling: each step, pick one random block ∈ skippable for a skip-branch
forward. Skip-branch KL pulls the skip-output toward teacher full-path logits.
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
import torch.nn.functional as F
from datasets import load_dataset

RETROFIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RETROFIT_ROOT))
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"
ULTRACHAT_PATH = "/home/user01/Minko/datasets/ultrachat_200k"
LLAVA_PATH = "/home/user01/Minko/datasets/llava_instruct_vsft"

IM_START_ID = 151644
IM_END_ID = 151645
ASSISTANT_ID = 77091
NEWLINE_ID = 198


def compute_assistant_mask(ids: list[int]) -> list[int]:
    """Mark assistant tokens (for SFT loss). Matches Qwen chat template."""
    mask = [0] * len(ids)
    i, n = 0, len(ids)
    while i < n:
        if (i + 2 < n and ids[i] == IM_START_ID and ids[i + 1] == ASSISTANT_ID
                and ids[i + 2] == NEWLINE_ID):
            j = i + 3
            while j < n and ids[j] != IM_END_ID:
                mask[j] = 1; j += 1
            if j < n:
                mask[j] = 1; j += 1
            i = j
        else:
            i += 1
    return mask


def ultrachat_iter(processor, seed=0, split="train_sft"):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                      cache_dir=ULTRACHAT_PATH, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=500)
    for s in ds:
        msgs = s.get("messages")
        if not msgs:
            continue
        try:
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=False)
            inputs = processor(text=[text], return_tensors="pt")
        except Exception:
            continue
        ids = inputs["input_ids"][0].tolist()
        mask = compute_assistant_mask(ids)
        if sum(mask) == 0:
            continue
        labels = torch.tensor(
            [(t if m == 1 else -100) for t, m in zip(ids, mask)], dtype=torch.long,
        )
        yield inputs, labels, "text"


def llava_iter(processor, seed=0):
    ds = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft",
                      cache_dir=LLAVA_PATH, split="train")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for idx in indices:
        s = ds[idx]
        msgs = s["messages"]
        imgs = s.get("images") or []
        try:
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=False)
            inputs = processor(text=[text], images=imgs if imgs else None,
                               return_tensors="pt")
        except Exception:
            continue
        ids = inputs["input_ids"][0].tolist()
        mask = compute_assistant_mask(ids)
        if sum(mask) == 0:
            continue
        labels = torch.tensor(
            [(t if m == 1 else -100) for t, m in zip(ids, mask)], dtype=torch.long,
        )
        yield inputs, labels, "vlm"


def mixed_stream(processor, seed=0, p_mm=0.5):
    """Interleave text (UltraChat) and multimodal (LLaVA) samples."""
    rng = random.Random(seed + 2)
    text = ultrachat_iter(processor, seed)
    vlm = llava_iter(processor, seed + 1)
    while True:
        pick_mm = rng.random() < p_mm
        try:
            yield next(vlm if pick_mm else text)
        except StopIteration:
            if pick_mm:
                vlm = llava_iter(processor, seed + rng.randint(1, 1_000_000))
            else:
                text = ultrachat_iter(processor, seed + rng.randint(1, 1_000_000))


def move_inputs(inputs, device):
    return {k: v.to(device) for k, v in inputs.items()}


def masked_kl(student_logits, teacher_logits, mask_pos, temperature):
    """Token-mean KL on the same assistant positions used by the SFT loss."""
    if not mask_pos.any():
        return torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
    vocab = student_logits.shape[-1]
    student = student_logits.view(-1, vocab)[mask_pos]
    teacher = teacher_logits.view(-1, vocab)[mask_pos]
    student_logp = F.log_softmax(student / temperature, dim=-1)
    teacher_p = F.softmax(teacher / temperature, dim=-1)
    return F.kl_div(
        student_logp.float(), teacher_p.float(), reduction="batchmean"
    ) * (temperature * temperature)


def linear_gamma_target(args, step, max_steps):
    ramp_end = max(int(max_steps * args.gamma_ramp_frac), 1)
    progress = min(step / ramp_end, 1.0)
    target = args.gamma_start + (args.gamma_end - args.gamma_start) * progress
    return float(target), ramp_end


def advance_trust_region_gamma(
    current,
    target,
    ramp_end,
    gamma_start,
    gamma_end,
    previous_kl_ema,
    previous_ratio_ema,
    kl_cap,
    ratio_cap,
):
    """Advance each block by at most one scheduled increment when admissible.

    The decision uses diagnostics from the preceding training sample, avoiding
    a second candidate forward.  A block is held when either the global
    teacher KL or that block's relative correction RMS exceeds its fixed cap.
    """
    direction = 1.0 if gamma_end >= gamma_start else -1.0
    increment = abs(gamma_end - gamma_start) / max(ramp_end, 1)
    proposed = current + direction * increment
    target_tensor = torch.full_like(current, target)
    if direction > 0:
        proposed = torch.minimum(proposed, target_tensor)
    else:
        proposed = torch.maximum(proposed, target_tensor)

    if previous_kl_ema is None or previous_ratio_ema is None:
        admissible = torch.ones_like(current, dtype=torch.bool)
    else:
        admissible = previous_ratio_ema.to(current.device) <= ratio_cap
        if previous_kl_ema > kl_cap:
            admissible = torch.zeros_like(admissible)
    return torch.where(admissible, proposed, current), admissible


def train(args):
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_path = getattr(args, "model_path", None) or MODEL_PATH
    print(f"[attnres-retrofit] Loading student + teacher from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    teacher = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    base = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype).to(device)
    skippable = None  # all blocks
    if args.skippable_blocks:
        skippable = [int(x) for x in args.skippable_blocks.split(",")]
    model = Qwen3VLAttnResRetrofit(
        base,
        num_blocks=args.num_blocks,
        skippable_blocks=skippable,
        adapter_rank=args.adapter_rank,
        no_adapter=args.no_adapter,
        bridge_mode=args.bridge_mode,
        compat_peak=args.compat_peak,
    ).move_retrofit_modules(device=device, dtype=dtype)

    # Freeze EVERY base param. Only router/adapters/gamma train.
    model.freeze_base()
    if args.gamma_schedule and not args.legacy_scheduled_gamma_optimizer:
        model.gamma.requires_grad_(False)
    trainable = [p for p in model.retrofit_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.base_model.parameters())
    print(f"[attnres-retrofit] trainable: {n_train/1e6:.2f}M  frozen base: {n_frozen/1e9:.2f}B")
    print(f"[attnres-retrofit] skippable blocks: {model.skippable_blocks}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    max_steps = args.steps
    warmup = args.warmup_steps

    def lr_scale(step):
        if step < warmup:
            return step / max(warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(
            math.pi * min((step - warmup) / max(max_steps - warmup, 1), 1.0)))

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "a")
    run_config = {
        **vars(args),
        "resolved_model_path": model_path,
        "resolved_bridge_mode": model.bridge_mode,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "trainable_parameters": n_train,
        "frozen_base_parameters": n_frozen,
        "command": [sys.executable, *sys.argv],
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, sort_keys=True)
    log_f.write("RUN_CONFIG=" + json.dumps(run_config, sort_keys=True) + "\n")
    log_f.flush()

    # Data stream: v1 path keeps ultrachat+llava mix (backwards-compatible
    # with existing H_r256_5k recipe); v2 path uses the multi-source mixer
    # from data_v2.py with reasoning / long-context / VL-math data.
    if args.data_mix and args.data_mix != "v1":
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        import data_v2
        spec = data_v2.parse_mix_string(args.data_mix)
        print(f"[attnres-retrofit] data-mix={args.data_mix} "
              f"weights={dict(zip(spec.names(), [round(w,3) for w in spec.normed()]))}")
        log_f.write(f"DATA_MIX={args.data_mix} weights={spec.weights}\n")

        def _mixed_stream_v2():
            raw = data_v2.build_mixed_stream(spec, seed=args.seed)
            n_fail = 0
            while True:
                try:
                    name, sample = next(raw)
                except Exception as e:
                    # Adapter failed (bad row, corrupt parquet, etc.). Reseed
                    # and retry — do not bring down the whole training run.
                    n_fail += 1
                    if n_fail % 50 == 1:
                        log_f.write(f"v2_stream_fail#{n_fail}: {type(e).__name__}: "
                                    f"{str(e)[:200]}\n"); log_f.flush()
                    raw = data_v2.build_mixed_stream(
                        spec, seed=args.seed + 1_000_000 + n_fail)
                    continue
                try:
                    enc = data_v2.encode_sample(
                        sample, processor, args.max_seq, compute_assistant_mask)
                except Exception as e:
                    n_fail += 1
                    if n_fail % 50 == 1:
                        log_f.write(f"v2_encode_fail#{n_fail} src={name}: "
                                    f"{type(e).__name__}: {str(e)[:200]}\n")
                        log_f.flush()
                    continue
                if enc is None:
                    continue
                inputs, labels = enc
                kind = data_v2.MODALITY.get(name, "text")
                yield inputs, labels, kind
        stream = _mixed_stream_v2()
    else:
        stream = mixed_stream(processor, seed=args.seed, p_mm=args.p_multimodal)
    t0 = time.time()
    ce_ema = kl_ema = ent_ema = full_kl_ema = None
    ratio_ema = None
    trust_hold_steps = 0
    trust_block_holds = torch.zeros(args.num_blocks, dtype=torch.long)
    rng = random.Random(args.seed + 7)

    for step in range(1, max_steps + 1):
        # Fetch sample (reject too-long)
        while True:
            inputs, labels, kind = next(stream)
            if inputs["input_ids"].shape[1] <= args.max_seq:
                break
        inputs = move_inputs(inputs, device)
        labels = labels.unsqueeze(0).to(device)

        # 1) Full-path forward on student — drives gamma / router / adapter
        model.train()
        # γ curriculum: force γ to scheduled value (overrides optimizer drift)
        if args.gamma_schedule:
            gamma_target, ramp_end = linear_gamma_target(args, step, max_steps)
            if args.gamma_controller == "trust_region":
                with torch.no_grad():
                    gamma_next, admissible = advance_trust_region_gamma(
                        current=model.gamma.data,
                        target=gamma_target,
                        ramp_end=ramp_end,
                        gamma_start=args.gamma_start,
                        gamma_end=args.gamma_end,
                        previous_kl_ema=full_kl_ema,
                        previous_ratio_ema=ratio_ema,
                        kl_cap=args.trust_kl_cap,
                        ratio_cap=args.trust_ratio_cap,
                    )
                    model.gamma.data.copy_(gamma_next)
                held = (~admissible).detach().cpu()
                if bool(held.any()):
                    trust_hold_steps += 1
                    trust_block_holds += held.long()
            else:
                with torch.no_grad():
                    model.gamma.data.fill_(gamma_target)

        full_out = model(
            **inputs,
            labels=labels,
            return_alpha=(args.entropy_weight > 0),
            return_correction_ratios=True,
        )
        ce_loss = full_out.loss

        # 2) Teacher forward (KD target)
        need_teacher = args.kl_weight > 0 or args.gamma_controller == "trust_region"
        full_kl_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
        if need_teacher:
            with torch.no_grad():
                t_out = teacher(**inputs, use_cache=False)
                teacher_logits = t_out.logits

            shifted = torch.cat(
                [labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1
            )
            mask_pos = (shifted != -100).view(-1)
            full_kl_loss = masked_kl(
                full_out.logits, teacher_logits, mask_pos, args.kd_temperature
            )

            if args.kl_weight > 0:
                # Skip-path forward: randomly pick one skippable block to skip
                skip_block = rng.choice(model.skippable_blocks)
                skip_out = model(**inputs, labels=None, skip_block_indices=[skip_block])
                kl_loss = masked_kl(
                    skip_out.logits, teacher_logits, mask_pos, args.kd_temperature
                )
            else:
                kl_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
                skip_block = None
        else:
            kl_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            skip_block = None

        # 3) Entropy reg on alpha: negate to PREVENT collapse (maximize entropy).
        #    ent_penalty = -Σ p log p  (positive, max at uniform, min at one-hot).
        #    Loss = ... - entropy_weight · ent  → optimizer maximizes entropy.
        #    Original buggy code used `+ entropy_weight · ent` which minimized it
        #    and caused α to collapse to a single source (typically embed).
        ent_term = torch.zeros((), device=device, dtype=ce_loss.dtype)
        if args.entropy_weight > 0 and full_out.entropy_penalty is not None:
            ent_term = -args.entropy_weight * full_out.entropy_penalty

        # Optional: pull γ toward target value (soft force AttnRes contribution)
        gamma_l2 = torch.zeros((), device=device, dtype=ce_loss.dtype)
        if args.gamma_l2_weight > 0:
            gamma_l2 = args.gamma_l2_weight * ((model.gamma - args.gamma_l2_target) ** 2).sum()

        total = ce_loss + args.kl_weight * kl_loss + ent_term + gamma_l2

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scale = lr_scale(step)
        for g in opt.param_groups:
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            g["lr"] = g["base_lr"] * scale
        opt.step()

        cf = float(ce_loss.detach())
        kf = float(kl_loss.detach())
        fkf = float(full_kl_loss.detach())
        ef = float(ent_term.detach()) if torch.is_tensor(ent_term) else float(ent_term)
        ratio_values = torch.stack(full_out.correction_ratios).float().cpu()
        ce_ema = cf if ce_ema is None else 0.98 * ce_ema + 0.02 * cf
        kl_ema = kf if kl_ema is None else 0.98 * kl_ema + 0.02 * kf
        full_kl_ema = fkf if full_kl_ema is None else 0.98 * full_kl_ema + 0.02 * fkf
        ratio_ema = ratio_values if ratio_ema is None else 0.98 * ratio_ema + 0.02 * ratio_values
        ent_ema = ef if ent_ema is None else 0.98 * ent_ema + 0.02 * ef

        if step % args.log_every == 0 or step == 1 or step == max_steps:
            gamma_max = float(model.gamma.detach().abs().max())
            gamma_mean = float(model.gamma.detach().mean())
            ratio_max = float(ratio_ema.max())
            elapsed = time.time() - t0
            line = (f"step {step}/{max_steps} ce={ce_ema:.3f} kl={kl_ema:.3f} "
                    f"full_kl={full_kl_ema:.4f} corr_rms_max={ratio_max:.4f} "
                    f"ent={ent_ema:.3f} γ_max={gamma_max:.4f} γ_mean={gamma_mean:+.4f} "
                    f"skip_blk={skip_block} kind={kind} lr_scale={scale:.3f} "
                    f"T_len={inputs['input_ids'].shape[1]} elapsed={elapsed:.0f}s")
            print(f"[attnres-retrofit] {line}", flush=True)
            log_f.write(line + "\n"); log_f.flush()

    gammas = model.gamma.detach().cpu().tolist()
    msg = f"FINAL gammas={gammas}"
    print(f"[attnres-retrofit] {msg}"); log_f.write(msg + "\n")
    summary = {
        "final_gammas": gammas,
        "ce_ema": ce_ema,
        "skip_kl_ema": kl_ema,
        "full_teacher_kl_ema": full_kl_ema,
        "correction_ratio_ema": ratio_ema.tolist() if ratio_ema is not None else None,
        "trust_hold_steps": trust_hold_steps,
        "trust_block_holds": trust_block_holds.tolist(),
        "elapsed_seconds": time.time() - t0,
    }
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    log_f.write("TRAIN_SUMMARY=" + json.dumps(summary, sort_keys=True) + "\n")
    log_f.close()

    state = {
        "router": model.router.state_dict(),
        "adapters": model.adapters.state_dict(),
        "gamma": model.gamma.data.cpu(),
        "skippable_blocks": list(model.skippable_blocks),
        "config": {
            "num_blocks": args.num_blocks,
            "adapter_rank": args.adapter_rank,
            "no_adapter": model.no_adapter,
            "bridge_mode": model.bridge_mode,
            "compat_peak": args.compat_peak,
            "steps": max_steps,
            "kl_weight": args.kl_weight,
            "p_multimodal": args.p_multimodal,
            "data_mix": args.data_mix,
            "max_seq": args.max_seq,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "kd_temperature": args.kd_temperature,
            "entropy_weight": args.entropy_weight,
            "gamma_schedule": args.gamma_schedule,
            "gamma_start": args.gamma_start,
            "gamma_end": args.gamma_end,
            "gamma_ramp_frac": args.gamma_ramp_frac,
            "gamma_controller": args.gamma_controller,
            "trust_kl_cap": args.trust_kl_cap,
            "trust_ratio_cap": args.trust_ratio_cap,
            "legacy_scheduled_gamma_optimizer": args.legacy_scheduled_gamma_optimizer,
            "seed": args.seed,
        },
    }
    torch.save(state, out_dir / "retrofit_attnres_state.pt")
    print(f"[attnres-retrofit] saved {out_dir}/retrofit_attnres_state.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=None,
                   help="HF model dir. Defaults to MODEL_PATH constant (Qwen3-VL-2B).")
    p.add_argument("--num-blocks", type=int, default=14)
    p.add_argument("--adapter-rank", type=int, default=128)
    p.add_argument("--no-adapter", action="store_true",
                   help="Pure Route A: x_n=(1-γ)h+γr, no adapter bottleneck.")
    p.add_argument(
        "--bridge-mode",
        choices=("adapter", "pure", "exact_compat"),
        default=None,
        help="AttnRes injection path. Omitted keeps --no-adapter compatibility.",
    )
    p.add_argument(
        "--compat-peak",
        type=float,
        default=1.0,
        help="Peak compatibility-adapter coefficient for exact_compat.",
    )
    p.add_argument("--skippable-blocks", default=None,
                   help="comma-separated block indices (default: all)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--p-multimodal", type=float, default=0.5,
                   help="v1-mix only: fraction of batches from LLaVA.")
    p.add_argument("--data-mix", default=None,
                   help="v1 (ultrachat+llava), v2 (canonical retrofit-v2 mix), "
                        "equal (uniform over all known sources), or a custom "
                        "spec 'name1:w1,name2:w2,...'. See retrofit/train/data_v2.py.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--kd-temperature", type=float, default=1.0)
    p.add_argument("--entropy-weight", type=float, default=0.02)
    p.add_argument("--gamma-schedule", action="store_true",
                   help="force γ to a scheduled value (default: free-learning)")
    p.add_argument("--gamma-start", type=float, default=0.0,
                   help="γ value at step 0 (used with --gamma-schedule)")
    p.add_argument("--gamma-end", type=float, default=1.0,
                   help="γ value after ramp (used with --gamma-schedule)")
    p.add_argument("--gamma-ramp-frac", type=float, default=0.5,
                   help="fraction of training over which γ ramps start→end")
    p.add_argument(
        "--gamma-controller",
        choices=("linear", "trust_region"),
        default="linear",
        help="linear schedule or per-block trust-region advancement.",
    )
    p.add_argument(
        "--trust-kl-cap", type=float, default=0.10,
        help="Previous-sample teacher-KL EMA cap for gamma advancement.",
    )
    p.add_argument(
        "--trust-ratio-cap", type=float, default=0.50,
        help="Per-block correction-RMS EMA cap for gamma advancement.",
    )
    p.add_argument(
        "--legacy-scheduled-gamma-optimizer",
        action="store_true",
        help="Keep scheduled gamma in AdamW, reproducing the original recipe.",
    )
    p.add_argument("--gamma-l2-weight", type=float, default=0.0,
                   help="L2 pull on γ toward --gamma-l2-target (soft alternative to curriculum)")
    p.add_argument("--gamma-l2-target", type=float, default=1.0,
                   help="target γ value for L2 pull")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    if args.gamma_controller == "trust_region" and not args.gamma_schedule:
        p.error("--gamma-controller trust_region requires --gamma-schedule")
    if args.gamma_l2_weight > 0 and args.gamma_schedule and not args.legacy_scheduled_gamma_optimizer:
        p.error("gamma L2 has no effect when scheduled gamma is excluded from AdamW")
    train(args)


if __name__ == "__main__":
    main()
