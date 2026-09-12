"""Fresh adaptation of Full AttnRes with a residual-strength gate.

The gate belongs to the ordinary Full AttnRes forward path. It rescales the
existing AttnRes correction at each target block and is optimized only by the
same adaptation objective as the original model:

    task CE + full-path teacher KL - native route entropy.

There is no skip forward, skip label, compute target, selector, recovery
module, or second training stage in this script.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

CODE_ROOT = Path(__file__).resolve().parent
BASE_SNAPSHOT_ROOT = CODE_ROOT / "base_snapshot"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(BASE_SNAPSHOT_ROOT))

from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit
from residual_strength_gate import (
    install_residual_strength_gate,
    residual_strength_gate_statistics,
)
from role_structured_router import (
    capture_head_alpha_list,
    install_role_router,
    routing_role_statistics,
)
from train_qwen3vl_attnres_retrofit import (
    compute_assistant_mask,
    linear_gamma_target,
    masked_kl,
    move_inputs,
)


FORBIDDEN_TRAINABLE_FRAGMENTS = (
    "recovery",
    "risk",
    "surrogate",
    "selector",
    "joint_gate",
)


def _build_encoded_stream(args, processor, log_f):
    os.environ["RESKIP_DATA_ROOT"] = str(args.data_root)
    import data_v2

    data_v2.DATA_ROOT = Path(args.data_root)
    spec = data_v2.parse_mix_string(args.data_mix)
    weights = dict(zip(spec.names(), [round(value, 6) for value in spec.normed()]))
    print(f"[rsg-attnres] data_mix={args.data_mix} weights={weights}", flush=True)
    log_f.write(f"DATA_MIX={args.data_mix} weights={weights}\n")
    log_f.flush()
    raw = data_v2.build_mixed_stream(spec, seed=args.seed)
    failures = 0
    while True:
        try:
            source_name, sample = next(raw)
        except Exception as error:
            failures += 1
            if failures % 50 == 1:
                log_f.write(
                    f"stream_fail#{failures}: {type(error).__name__}: "
                    f"{str(error)[:240]}\n"
                )
                log_f.flush()
            raw = data_v2.build_mixed_stream(
                spec,
                seed=args.seed + 1_000_000 + failures,
            )
            continue
        try:
            encoded = data_v2.encode_sample(
                sample,
                processor,
                args.max_seq,
                compute_assistant_mask,
            )
        except Exception as error:
            failures += 1
            if failures % 50 == 1:
                log_f.write(
                    f"encode_fail#{failures} source={source_name}: "
                    f"{type(error).__name__}: {str(error)[:240]}\n"
                )
                log_f.flush()
            continue
        if encoded is None:
            continue
        inputs, labels = encoded
        yield inputs, labels, source_name, data_v2.MODALITY.get(source_name, "text")


def _parameter_audit(model, base_retrofit_parameters: int, expected_added: int) -> dict:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    forbidden = sorted(
        name
        for name, _ in trainable
        if any(fragment in name.lower() for fragment in FORBIDDEN_TRAINABLE_FRAGMENTS)
    )
    if forbidden:
        raise RuntimeError(f"forbidden trainable modules present: {forbidden}")
    gate_names = sorted(
        name for name, _ in trainable if name.startswith("residual_strength_gates.")
    )
    expected_names = sorted(
        f"residual_strength_gates.{block}.{suffix}"
        for block in model.residual_strength_gate_blocks
        for suffix in ("bias", "weight")
    )
    if gate_names != expected_names:
        raise RuntimeError(
            f"unexpected residual-strength gate parameters: {gate_names}"
        )
    added = sum(
        parameter.numel()
        for name, parameter in trainable
        if name.startswith("residual_strength_gates.")
    )
    if added != int(expected_added):
        raise RuntimeError(
            f"gate parameter mismatch: observed={added}, expected={expected_added}"
        )
    total = sum(parameter.numel() for _, parameter in trainable)
    if total != int(base_retrofit_parameters + added):
        raise RuntimeError(
            "trainable parameter mismatch: "
            f"observed={total}, expected={base_retrofit_parameters + added}"
        )
    return {
        "variant": "attnres",
        "routing_heads": 1,
        "base_retrofit_trainable_parameters": int(base_retrofit_parameters),
        "additional_trainable_parameters": int(added),
        "total_trainable_parameters": int(total),
        "trainable_names": [name for name, _ in trainable],
        "residual_strength_gate_trainable_names": gate_names,
        "forbidden_trainable_names": forbidden,
        "skip_specific_parameters": 0,
        "external_controller_parameters": 0,
    }


def _gate_parameter_statistics(model) -> dict:
    weights = torch.cat(
        [
            gate.weight.detach().float().reshape(-1)
            for gate in model.residual_strength_gates.values()
        ]
    )
    biases = torch.cat(
        [
            gate.bias.detach().float().reshape(-1)
            for gate in model.residual_strength_gates.values()
        ]
    )
    return {
        "weight_rms": float(weights.square().mean().sqrt()),
        "weight_max_abs": float(weights.abs().max()),
        "bias_rms": float(biases.square().mean().sqrt()),
        "bias_max_abs": float(biases.abs().max()),
    }


def _aggregate_factor_statistics(model) -> dict:
    by_block = residual_strength_gate_statistics(model)
    if not by_block:
        return {
            "count": 0,
            "mean": 1.0,
            "min": 1.0,
            "max": 1.0,
            "by_block": {},
        }
    count = sum(record["count"] for record in by_block.values())
    mean = sum(
        record["count"] * record["mean"] for record in by_block.values()
    ) / count
    return {
        "count": int(count),
        "mean": float(mean),
        "min": min(record["min"] for record in by_block.values()),
        "max": max(record["max"] for record in by_block.values()),
        "by_block": by_block,
    }


def train(args):
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    final_checkpoint = output_dir / "retrofit_attnres_state.pt"
    log_f = open(output_dir / "train.log", "w", encoding="utf-8")

    print(
        f"[rsg-attnres] loading independent teacher and student from {args.model_path}",
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and args.max_image_pixels > 0:
        image_size = dict(getattr(image_processor, "size", {}) or {})
        image_size["longest_edge"] = int(args.max_image_pixels)
        image_processor.size = image_size

    teacher = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
    ).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    base = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
    ).to(device)
    model = Qwen3VLAttnResRetrofit(
        base,
        num_blocks=args.num_blocks,
        adapter_rank=args.adapter_rank,
        bridge_mode="adapter",
        identity_anchored_source_calibration=False,
    ).move_retrofit_modules(device=device, dtype=dtype)
    model.freeze_base()
    model.gamma.requires_grad_(False)
    install_report = install_role_router(
        model,
        variant="attnres",
        num_heads=1,
        num_basis=4,
        role_scale=1.0,
        random_basis_seed=314159,
    )
    if install_report.additional_parameters != 0:
        raise RuntimeError("native H=1 router unexpectedly added parameters")
    base_retrofit_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    gate_report = install_residual_strength_gate(
        model,
        gate_scale=args.gate_scale,
        gated_blocks=tuple(range(1, args.num_blocks)),
    )
    audit = _parameter_audit(
        model,
        base_retrofit_parameters=base_retrofit_parameters,
        expected_added=gate_report.additional_parameters,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen_base = sum(parameter.numel() for parameter in model.base_model.parameters())
    (output_dir / "PARAMETER_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[rsg-attnres] method={args.method_id} trainable="
        f"{audit['total_trainable_parameters'] / 1e6:.4f}M "
        f"added={audit['additional_trainable_parameters']} "
        f"frozen_base={frozen_base / 1e9:.2f}B",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    def learning_rate_scale(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = min(
            (step - args.warmup_steps)
            / max(args.steps - args.warmup_steps, 1),
            1.0,
        )
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    run_config = {
        **vars(args),
        "bridge_mode": "adapter",
        "identity_anchored_source_calibration": False,
        "recent_source_calibrated": False,
        "residual_strength_gate": True,
        "residual_strength_gate_scale": float(args.gate_scale),
        "residual_strength_gate_blocks": list(gate_report.gated_blocks),
        "router_variant": "attnres",
        "routing_heads": 1,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameter_audit": audit,
        "training_objective": "full_task_ce_plus_full_teacher_kl_minus_native_route_entropy",
        "single_new_component": "token_conditioned_full_path_residual_strength_gate",
        "primary_purpose": "improve_full_attnres_residual_adaptation",
        "fresh_pretrained_base": True,
        "loaded_training_checkpoint": None,
        "warm_started_from_prior_experiment": False,
        "skip_branch_in_training": False,
        "skip_training_forwards": 0,
        "skip_supervision": False,
        "counterfactual_skip_labels": False,
        "coverage_or_compute_target": False,
        "second_stage_training": False,
        "command": [sys.executable, *sys.argv],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log_f.write("RUN_CONFIG=" + json.dumps(run_config, sort_keys=True) + "\n")
    log_f.flush()

    stream = _build_encoded_stream(args, processor, log_f)
    start_time = time.time()
    ce_ema = None
    full_kl_ema = None
    entropy_ema = None
    role_js_ema = None
    pairwise_js_ema = None
    effective_rank_ema = None
    ratio_ema = None
    identity_max_abs_logit_delta = None

    for step in range(1, args.steps + 1):
        inputs, labels, source_name, modality = next(stream)
        inputs = move_inputs(inputs, device)
        labels = labels.unsqueeze(0).to(device)

        if step == 1:
            model.eval()
            with torch.no_grad():
                identity_output = model(**inputs, labels=labels)
                identity_teacher = teacher(**inputs, use_cache=False)
            identity_max_abs_logit_delta = float(
                (identity_output.logits - identity_teacher.logits).abs().max()
            )
            if identity_max_abs_logit_delta > 1.0e-4:
                raise RuntimeError(
                    "identity-at-init failed: max absolute logit delta="
                    f"{identity_max_abs_logit_delta}"
                )
            print(
                "[rsg-attnres] identity_at_init_max_abs_logit_delta="
                f"{identity_max_abs_logit_delta:.8g}",
                flush=True,
            )

        model.train()
        gamma_target, _ = linear_gamma_target(args, step, args.steps)
        with torch.no_grad():
            model.gamma.data.fill_(gamma_target)
        full_output = model(
            **inputs,
            labels=labels,
            return_alpha=True,
            return_correction_ratios=True,
        )
        ce_loss = full_output.loss
        with torch.no_grad():
            teacher_output = teacher(**inputs, use_cache=False)
        shifted_labels = torch.cat(
            [labels[..., 1:], torch.full_like(labels[:, :1], -100)],
            dim=1,
        )
        prediction_mask = shifted_labels.ne(-100).reshape(-1)
        full_kl_loss = masked_kl(
            full_output.logits,
            teacher_output.logits,
            prediction_mask,
            args.kd_temperature,
        )
        head_alphas = capture_head_alpha_list(model, full_output.alpha_list)
        route_stats = routing_role_statistics(head_alphas, num_bins=4)
        total_loss = (
            ce_loss
            + args.full_kl_weight * full_kl_loss
            - args.entropy_weight * route_stats["route_entropy"]
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        lr_scale = learning_rate_scale(step)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale
        optimizer.step()

        ce_value = float(ce_loss.detach())
        kl_value = float(full_kl_loss.detach())
        entropy_value = float(route_stats["route_entropy"].detach())
        role_js_value = float(route_stats["role_js"].detach())
        pair_js_value = float(route_stats["pairwise_js"].detach())
        rank_value = float(route_stats["effective_rank"].detach())
        ratio_values = torch.stack(full_output.correction_ratios).float().cpu()
        ce_ema = ce_value if ce_ema is None else 0.98 * ce_ema + 0.02 * ce_value
        full_kl_ema = kl_value if full_kl_ema is None else 0.98 * full_kl_ema + 0.02 * kl_value
        entropy_ema = entropy_value if entropy_ema is None else 0.98 * entropy_ema + 0.02 * entropy_value
        role_js_ema = role_js_value if role_js_ema is None else 0.98 * role_js_ema + 0.02 * role_js_value
        pairwise_js_ema = pair_js_value if pairwise_js_ema is None else 0.98 * pairwise_js_ema + 0.02 * pair_js_value
        effective_rank_ema = rank_value if effective_rank_ema is None else 0.98 * effective_rank_ema + 0.02 * rank_value
        ratio_ema = ratio_values if ratio_ema is None else 0.98 * ratio_ema + 0.02 * ratio_values

        if step == 1 or step == args.steps or step % args.log_every == 0:
            parameters = _gate_parameter_statistics(model)
            factors = _aggregate_factor_statistics(model)
            elapsed = time.time() - start_time
            line = (
                f"step {step}/{args.steps} ce={ce_ema:.4f} "
                f"full_kl={full_kl_ema:.5f} route_H={entropy_ema:.4f} "
                f"role_js={role_js_ema:.6f} pair_js={pairwise_js_ema:.6f} "
                f"eff_rank={effective_rank_ema:.4f} "
                f"gate_w_rms={parameters['weight_rms']:.5f} "
                f"gate_mean={factors['mean']:.5f} "
                f"gate_range=[{factors['min']:.5f},{factors['max']:.5f}] "
                f"gamma={float(model.gamma.mean()):.4f} "
                f"corr_max={float(ratio_ema.max()):.4f} source={source_name} "
                f"kind={modality} seq={inputs['input_ids'].shape[1]} "
                f"lr_scale={lr_scale:.4f} elapsed={elapsed:.0f}s"
            )
            print(f"[rsg-attnres] {line}", flush=True)
            log_f.write(line + "\n")
            log_f.flush()

    gate_parameter_summary = _gate_parameter_statistics(model)
    gate_factor_summary = _aggregate_factor_statistics(model)
    summary = {
        "method_id": args.method_id,
        "steps": int(args.steps),
        "ce_ema": ce_ema,
        "full_teacher_kl_ema": full_kl_ema,
        "route_entropy_ema": entropy_ema,
        "role_js_ema": role_js_ema,
        "pairwise_js_ema": pairwise_js_ema,
        "effective_rank_ema": effective_rank_ema,
        "correction_ratio_ema": ratio_ema.tolist(),
        "identity_max_abs_logit_delta": identity_max_abs_logit_delta,
        "residual_strength_gate_parameters": gate_parameter_summary,
        "last_batch_residual_strength_factors": gate_factor_summary,
        "final_gammas": model.gamma.detach().float().cpu().tolist(),
        "elapsed_seconds": time.time() - start_time,
        "skip_training_forwards": 0,
        "parameter_audit": audit,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log_f.write("TRAIN_SUMMARY=" + json.dumps(summary, sort_keys=True) + "\n")
    log_f.close()

    checkpoint = {
        "router": model.router.state_dict(),
        "adapters": model.adapters.state_dict(),
        "residual_strength_gates": model.residual_strength_gates.state_dict(),
        "gamma": model.gamma.detach().cpu(),
        "skippable_blocks": list(model.skippable_blocks),
        "config": {
            "num_blocks": args.num_blocks,
            "adapter_rank": args.adapter_rank,
            "no_adapter": False,
            "bridge_mode": "adapter",
            "compat_peak": 1.0,
            "identity_anchored_source_calibration": False,
            "residual_strength_gate": True,
            "residual_strength_gate_scale": float(args.gate_scale),
            "residual_strength_gate_blocks": list(gate_report.gated_blocks),
            "steps": args.steps,
            "data_mix": args.data_mix,
            "max_seq": args.max_seq,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "full_kl_weight": args.full_kl_weight,
            "kd_temperature": args.kd_temperature,
            "entropy_weight": args.entropy_weight,
            "gamma_schedule": True,
            "gamma_start": args.gamma_start,
            "gamma_end": args.gamma_end,
            "gamma_ramp_frac": args.gamma_ramp_frac,
            "gamma_controller": "linear",
            "legacy_scheduled_gamma_optimizer": False,
            "router_variant": "attnres",
            "routing_heads": 1,
            "role_basis_count": 4,
            "role_logit_scale": 1.0,
            "random_basis_seed": 314159,
            "method_id": args.method_id,
            "training_objective": "full_only_residual_strength_gated_attnres",
            "additional_trainable_parameters": audit[
                "additional_trainable_parameters"
            ],
            "skip_specific_parameters": 0,
            "skip_supervision": False,
            "second_stage_training": False,
        },
    }
    torch.save(checkpoint, final_checkpoint)
    print(f"[rsg-attnres] saved {final_checkpoint}", flush=True)

    close_stream = getattr(stream, "close", None)
    if close_stream is not None:
        close_stream()
    model.router.role_head_alpha_by_position.clear()
    model.router.role_head_alpha_raw_by_position.clear()
    model.router.role_keys_by_position.clear()
    del checkpoint, optimizer, trainable, model, base, teacher, processor, stream
    gc.collect()
    torch.cuda.empty_cache()
    print("[rsg-attnres] cleanup complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-id", default="S1_rsg_h1")
    parser.add_argument("--data-root", default="/data/Minko/datasets")
    parser.add_argument("--data-mix", default="v3")
    parser.add_argument("--num-blocks", type=int, default=7)
    parser.add_argument("--adapter-rank", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--max-seq", type=int, default=1536)
    parser.add_argument("--max-image-pixels", type=int, default=802816)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--full-kl-weight", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--gate-scale", type=float, default=0.5)
    parser.add_argument("--gamma-start", type=float, default=0.0)
    parser.add_argument("--gamma-end", type=float, default=1.0)
    parser.add_argument("--gamma-ramp-frac", type=float, default=0.30)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.gpu not in {0, 1, 2, 3, 4, 5}:
        parser.error("GPU must be one of 0,1,2,3,4,5; cards 6 and 7 are forbidden")
    train(args)


if __name__ == "__main__":
    main()
    # PyArrow's background filesystem thread can crash CPython during process
    # finalization on this host after every artifact has already been saved.
    # A clean hard exit avoids that external shutdown race without masking any
    # exception raised by main().
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
