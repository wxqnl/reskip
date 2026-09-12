#!/usr/bin/env python3
"""Train Full-path RSM-AttnRes on a standard Hugging Face VLM decoder.

The residual-strength component participates in every ordinary AttnRes
forward. Training contains no skipped execution, skip label, compute target,
selector, or second stage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from generic_vlm_attnres_retrofit import GenericVLMAttnResRetrofit


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _assistant_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content or "")


def _normalize_messages_for_template(messages: list[dict], processor) -> list[dict]:
    """Normalize string turns for templates that require typed content lists.

    SmolVLM2's chat template indexes every turn as a list of typed content
    items. Passing a plain string silently renders an empty answer followed by
    only ``<end_of_utterance>``. Granite Vision's template likewise indexes
    ``message['content'][0]['text']`` and otherwise raises on assistant turns.
    InternVL accepts both representations, so keep its native input untouched
    unless the active template explicitly requires typed items.
    """
    template = getattr(processor, "chat_template", None)
    if not template:
        template = getattr(getattr(processor, "tokenizer", None), "chat_template", "")
    template_text = str(template)
    typed_content_required = (
        "<end_of_utterance>" in template_text
        or "message['content'][0]['text']" in template_text
        or 'message["content"][0]["text"]' in template_text
    )
    if not typed_content_required:
        return messages

    normalized = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, str):
            copied["content"] = [{"type": "text", "text": content}]
        normalized.append(copied)
    return normalized


def _common_prefix_length(left: torch.Tensor, right: torch.Tensor) -> int:
    shared = 0
    upper = min(left.numel(), right.numel())
    while shared < upper and int(left[shared]) == int(right[shared]):
        shared += 1
    return shared


def encode_all_assistant_turns(
    sample: dict,
    processor,
    max_seq: int,
):
    """Encode a conversation and supervise every non-empty assistant turn.

    This mirrors the canonical Qwen training mask without hard-coding
    model-specific role-token IDs. For each assistant turn, its response span
    is recovered by aligning (1) the preceding dialogue with a generation
    prompt, (2) the dialogue through that response, and (3) the full dialogue.
    """
    messages = _normalize_messages_for_template(sample["messages"], processor)
    images = sample.get("images") or []
    assistant_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and _assistant_text(message.get("content")).strip()
    ]
    if not assistant_positions:
        return None

    try:
        full_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_inputs = processor(
            text=[full_text],
            images=images if images else None,
            return_tensors="pt",
        )
    except Exception:
        return None

    full_ids = full_inputs["input_ids"][0]
    if full_ids.numel() > max_seq or full_ids.numel() < 8:
        return None

    labels = torch.full_like(full_ids, -100)
    for assistant_index in assistant_positions:
        prefix_messages = messages[:assistant_index]
        if not prefix_messages:
            continue
        through_messages = messages[: assistant_index + 1]
        try:
            prefix_text = processor.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix_ids = processor(
                text=[prefix_text],
                images=images if images else None,
                return_tensors="pt",
            )["input_ids"][0]
            if len(through_messages) == len(messages):
                through_ids = full_ids
            else:
                through_text = processor.apply_chat_template(
                    through_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                through_ids = processor(
                    text=[through_text],
                    images=images if images else None,
                    return_tensors="pt",
                )["input_ids"][0]
        except Exception:
            return None

        start = _common_prefix_length(prefix_ids, through_ids)
        end = _common_prefix_length(through_ids, full_ids)
        # A completed dialogue prefix should be token-identical to the same
        # prefix inside the full conversation. Reject ambiguous alignments
        # instead of silently supervising the wrong span.
        if end != through_ids.numel() or start >= end:
            return None
        labels[start:end] = full_ids[start:end]

    if not bool((labels != -100).any()):
        return None
    return full_inputs, labels


def move_inputs(inputs, device: str, dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def build_stream(processor, args):
    project_train = Path(args.project) / "retrofit" / "train"
    sys.path.insert(0, str(project_train))
    import data_v2

    data_v2.DATA_ROOT = Path(args.data_root)
    spec = data_v2.parse_mix_string(args.data_mix)
    print(
        f"[generic-ar] data-mix={args.data_mix} "
        f"weights={dict(zip(spec.names(), [round(x, 3) for x in spec.normed()]))}",
        flush=True,
    )

    def iterator():
        failures = 0
        raw = data_v2.build_mixed_stream(spec, seed=args.seed)
        while True:
            try:
                source, sample = next(raw)
                encoded = encode_all_assistant_turns(sample, processor, args.max_seq)
            except Exception as error:
                failures += 1
                if failures % 50 == 1:
                    print(
                        f"[generic-ar] stream_failure#{failures}: "
                        f"{type(error).__name__}: {str(error)[:160]}",
                        flush=True,
                    )
                raw = data_v2.build_mixed_stream(
                    spec,
                    seed=args.seed + 1_000_000 + failures,
                )
                continue
            if encoded is None:
                continue
            inputs, labels = encoded
            yield inputs, labels, data_v2.MODALITY.get(source, "text"), source

    return iterator()


def masked_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    shifted = torch.cat(
        [labels[..., 1:], torch.full_like(labels[:, :1], -100)],
        dim=1,
    )
    mask = (shifted != -100).reshape(-1)
    if not mask.any():
        return student_logits.new_zeros(())
    vocab = student_logits.shape[-1]
    student = student_logits.reshape(-1, vocab)[mask]
    teacher = teacher_logits.reshape(-1, vocab)[mask]
    student_logp = F.log_softmax(student / temperature, dim=-1)
    teacher_prob = F.softmax(teacher / temperature, dim=-1)
    return F.kl_div(
        student_logp.float(),
        teacher_prob.float(),
        reduction="batchmean",
    ) * (temperature * temperature)


def train(args) -> None:
    seed_everything(args.seed)
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        fix_mistral_regex=args.fix_mistral_regex,
    )
    print(f"[generic-ar] loading teacher from {args.model_path}", flush=True)
    teacher = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    print(f"[generic-ar] loading student from {args.model_path}", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    base.eval()
    enable_rsm = not args.disable_residual_strength_gate
    model = GenericVLMAttnResRetrofit(
        base,
        num_blocks=args.num_blocks,
        adapter_rank=args.adapter_rank,
        residual_strength_gate=enable_rsm,
        gate_scale=args.gate_scale,
    )
    model.move_retrofit(device=device, dtype=dtype)
    model.freeze_base()
    model.gamma.requires_grad_(False)
    trainable = (
        list(model.router.parameters())
        + list(model.adapters.parameters())
        + list(model.residual_strength_gates.parameters())
    )
    for parameter in trainable:
        parameter.requires_grad = True

    optimized_count = sum(parameter.numel() for parameter in trainable)
    gate_count = sum(
        parameter.numel()
        for parameter in model.residual_strength_gates.parameters()
    )
    base_retrofit_count = optimized_count - gate_count
    added_count = optimized_count + model.gamma.numel()
    base_count = sum(parameter.numel() for parameter in base.parameters())
    print(
        f"[generic-ar] family={model.family} layers={model.num_layers} "
        f"blocks={model.num_blocks} layers_per_block={model.layers_per_block} "
        f"optimized={optimized_count / 1e6:.3f}M "
        f"rsm_added={gate_count} "
        f"added={added_count / 1e6:.3f}M "
        f"base={base_count / 1e9:.3f}B",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    def lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = min(
            (step - args.warmup_steps)
            / max(args.steps - args.warmup_steps, 1),
            1.0,
        )
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    config = vars(args).copy()
    config.update(
        {
            "family": model.family,
            "num_layers": model.num_layers,
            "layers_per_block": model.layers_per_block,
            "optimized_params": optimized_count,
            "added_params": added_count,
            "base_retrofit_trainable_params": base_retrofit_count,
            "residual_strength_gate_params": gate_count,
            "residual_strength_gate": enable_rsm,
            "gate_scale": args.gate_scale,
            "gamma_policy": "deterministic schedule; excluded from optimizer",
            "base_params": base_count,
            "label_policy": "all assistant turns via model chat-template alignment",
            "objective": "full assistant CE + full-path teacher KL - native route entropy",
            "fresh_pretrained_base": True,
            "skip_branch_in_training": False,
            "skip_training_forwards": 0,
            "skip_supervision": False,
            "coverage_or_compute_target": False,
            "second_stage_training": False,
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
    )

    stream = build_stream(processor, args)
    log_file = (output_dir / "train.log").open("a")
    averages: dict[str, float] = {}
    start_time = time.time()
    ramp_steps = max(int(args.steps * args.gamma_ramp_frac), 1)
    identity_max_abs_logit_delta = None

    for step in range(1, args.steps + 1):
        raw_inputs, raw_labels, modality, source = next(stream)
        inputs = move_inputs(raw_inputs, device, dtype)
        labels = raw_labels.unsqueeze(0).to(device)

        if step == 1:
            model.eval()
            with torch.no_grad():
                identity_logits = model(
                    **inputs,
                    labels=None,
                    use_cache=False,
                ).logits
                teacher_identity_logits = teacher(
                    **inputs,
                    use_cache=False,
                ).logits
            identity_max_abs_logit_delta = float(
                (identity_logits - teacher_identity_logits).abs().max()
            )
            if identity_max_abs_logit_delta > 1.0e-4:
                raise RuntimeError(
                    "identity-at-init failed: max absolute logit delta="
                    f"{identity_max_abs_logit_delta}"
                )
            print(
                "[generic-rsm] identity_at_init_max_abs_logit_delta="
                f"{identity_max_abs_logit_delta:.8g}",
                flush=True,
            )

        gamma_target = (
            args.gamma_start
            + (args.gamma_end - args.gamma_start) * (step / ramp_steps)
            if step <= ramp_steps
            else args.gamma_end
        )
        with torch.no_grad():
            model.gamma.fill_(gamma_target)

        model.train()
        full = model(
            **inputs,
            labels=labels,
            return_alpha=args.entropy_weight > 0,
            use_cache=False,
        )
        if full.loss is None:
            raise RuntimeError("Full-path loss was not computed")

        with torch.no_grad():
            teacher_logits = teacher(**inputs, use_cache=False).logits
        full_kl_loss = masked_kl(
            full.logits,
            teacher_logits,
            labels,
            args.kd_temperature,
        )
        entropy = (
            full.entropy_penalty
            if full.entropy_penalty is not None
            else full.loss.new_zeros(())
        )
        total = (
            full.loss
            + args.full_kl_weight * full_kl_loss
            - args.entropy_weight * entropy
        )

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        scale = lr_scale(step)
        for group in optimizer.param_groups:
            if "base_lr" not in group:
                group["base_lr"] = group["lr"]
            group["lr"] = group["base_lr"] * scale
        optimizer.step()

        values = {
            "loss": float(total.detach()),
            "ce": float(full.loss.detach()),
            "full_kl": float(full_kl_loss.detach()),
            "entropy": float(entropy.detach()),
        }
        for key, value in values.items():
            averages[key] = (
                value
                if key not in averages
                else 0.98 * averages[key] + 0.02 * value
            )

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.time() - start_time
            gate_records = model.residual_strength_statistics()
            gate_count_observed = sum(
                record["count"] for record in gate_records.values()
            )
            gate_mean = (
                sum(
                    record["count"] * record["mean"]
                    for record in gate_records.values()
                )
                / gate_count_observed
                if gate_count_observed
                else 1.0
            )
            gate_min = min(
                (record["min"] for record in gate_records.values()),
                default=1.0,
            )
            gate_max = max(
                (record["max"] for record in gate_records.values()),
                default=1.0,
            )
            line = (
                f"step {step}/{args.steps} loss={averages['loss']:.3f} "
                f"ce={averages['ce']:.3f} "
                f"full_kl={averages['full_kl']:.3f} "
                f"entropy={averages['entropy']:.3f} "
                f"gate_mean={gate_mean:.4f} "
                f"gate_range=[{gate_min:.4f},{gate_max:.4f}] "
                f"gamma={float(model.gamma.mean()):.4f} "
                f"modality={modality} source={source} "
                f"seq={inputs['input_ids'].shape[1]} lr_scale={scale:.3f} "
                f"elapsed={elapsed:.0f}s"
            )
            print(f"[generic-ar] {line}", flush=True)
            log_file.write(line + "\n")
            log_file.flush()

    gate_weights = [
        gate.weight.detach().float().reshape(-1)
        for gate in model.residual_strength_gates.values()
    ]
    gate_biases = [
        gate.bias.detach().float().reshape(-1)
        for gate in model.residual_strength_gates.values()
    ]
    train_summary = {
        "steps": args.steps,
        "elapsed_seconds": time.time() - start_time,
        "ce_ema": averages.get("ce"),
        "full_teacher_kl_ema": averages.get("full_kl"),
        "route_entropy_ema": averages.get("entropy"),
        "identity_max_abs_logit_delta": identity_max_abs_logit_delta,
        "final_gammas": model.gamma.detach().float().cpu().tolist(),
        "last_batch_residual_strength_factors": model.residual_strength_statistics(),
        "residual_strength_gate_parameters": (
            {
                "weight_rms": float(
                    torch.cat(gate_weights).square().mean().sqrt()
                ),
                "weight_max_abs": float(torch.cat(gate_weights).abs().max()),
                "bias_rms": float(
                    torch.cat(gate_biases).square().mean().sqrt()
                ),
                "bias_max_abs": float(torch.cat(gate_biases).abs().max()),
            }
            if gate_weights
            else None
        ),
        "skip_training_forwards": 0,
        "skip_specific_parameters": 0,
        "residual_strength_gate_parameters_count": gate_count,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(train_summary, indent=2, sort_keys=True)
    )
    log_file.write("TRAIN_SUMMARY=" + json.dumps(train_summary, sort_keys=True) + "\n")
    log_file.close()
    model.save_state(
        output_dir / "generic_attnres_state.pt",
        extra_config={
            "model_path": args.model_path,
            "steps": args.steps,
            "max_seq": args.max_seq,
            "data_mix": args.data_mix,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "gamma_start": args.gamma_start,
            "gamma_end": args.gamma_end,
            "gamma_ramp_frac": args.gamma_ramp_frac,
            "full_kl_weight": args.full_kl_weight,
            "entropy_weight": args.entropy_weight,
            "seed": args.seed,
            "optimized_params": optimized_count,
            "added_params": added_count,
            "base_retrofit_trainable_params": base_retrofit_count,
            "residual_strength_gate_params": gate_count,
            "residual_strength_gate": enable_rsm,
            "gate_scale": args.gate_scale,
            "gamma_policy": "deterministic schedule; excluded from optimizer",
            "training_objective": "full_only_residual_strength_gated_attnres",
            "skip_branch_in_training": False,
            "skip_training_forwards": 0,
            "skip_specific_parameters": 0,
            "second_stage_training": False,
        },
    )
    print(
        f"[generic-ar] saved {output_dir / 'generic_attnres_state.pt'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--fix-mistral-regex", action="store_true")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--data-mix", default="v3")
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--adapter-rank", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--max-seq", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gamma-start", type=float, default=0.0)
    parser.add_argument("--gamma-end", type=float, default=1.0)
    parser.add_argument("--gamma-ramp-frac", type=float, default=0.3)
    parser.add_argument("--full-kl-weight", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--gate-scale", type=float, default=0.5)
    parser.add_argument("--disable-residual-strength-gate", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output-dir", required=True)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
