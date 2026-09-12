"""Paired held-out Full-path and skip-relevance analysis for RSG-AttnRes."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "base_snapshot"))

from checkpoint_utils import load_role_retrofit
from residual_strength_gate import set_residual_strength_gate_mode
from role_structured_router import _triangular_depth_basis, capture_head_alpha_list
from train_qwen3vl_attnres_retrofit import (
    compute_assistant_mask,
    masked_kl,
    move_inputs,
)


CANONICAL_MICRO_ACTIONS = {
    1: (1, 2, 3),
    2: (1,),
    3: (0, 1),
    4: (1,),
    5: (0, 3),
}


def _encoded_stream(args, processor):
    os.environ["RESKIP_DATA_ROOT"] = str(args.data_root)
    import data_v2

    data_v2.DATA_ROOT = Path(args.data_root)
    spec = data_v2.parse_mix_string(args.data_mix)
    raw = data_v2.build_mixed_stream(spec, seed=args.seed)
    failures = 0
    while True:
        try:
            source_name, sample = next(raw)
            encoded = data_v2.encode_sample(
                sample,
                processor,
                args.max_seq,
                compute_assistant_mask,
            )
        except Exception:
            failures += 1
            raw = data_v2.build_mixed_stream(
                spec,
                seed=args.seed + 1_000_000 + failures,
            )
            continue
        if encoded is None:
            continue
        inputs, labels = encoded
        yield inputs, labels, source_name, data_v2.MODALITY.get(source_name, "text")


def _prediction_masks(labels):
    shifted = torch.cat(
        [labels[..., 1:], torch.full_like(labels[:, :1], -100)],
        dim=1,
    )
    prediction = shifted.ne(-100)
    decode = prediction & labels.ne(-100)
    return shifted, prediction, decode


def _token_metrics(logits, teacher_logits, labels, temperature):
    shifted, prediction, _ = _prediction_masks(labels)
    flat_mask = prediction.reshape(-1)
    count = int(flat_mask.sum())
    if count == 0:
        raise RuntimeError("held-out sample has no assistant prediction tokens")
    vocab = logits.shape[-1]
    ce_sum = F.cross_entropy(
        logits.reshape(-1, vocab),
        shifted.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    teacher_ce_sum = F.cross_entropy(
        teacher_logits.reshape(-1, vocab),
        shifted.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    kl_mean = masked_kl(logits, teacher_logits, flat_mask, temperature)
    agreement = logits.argmax(dim=-1).eq(teacher_logits.argmax(dim=-1)) & prediction
    return {
        "tokens": count,
        "ce_sum": float(ce_sum),
        "teacher_ce_sum": float(teacher_ce_sum),
        "kl_sum": float(kl_mean) * count,
        "top1_agreement_count": int(agreement.sum()),
    }


def _profile_metrics(matrix: torch.Tensor) -> dict:
    matrix = matrix.float()
    matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    mean_profile = matrix.mean(dim=0)
    generalized_js = -(
        mean_profile * mean_profile.clamp_min(1.0e-8).log()
    ).sum() + (
        matrix * matrix.clamp_min(1.0e-8).log()
    ).sum(dim=-1).mean()
    pairwise = []
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            midpoint = 0.5 * (matrix[left] + matrix[right])
            pairwise.append(
                0.5
                * (
                    (
                        matrix[left]
                        * (
                            matrix[left].clamp_min(1.0e-8).log()
                            - midpoint.clamp_min(1.0e-8).log()
                        )
                    ).sum()
                    + (
                        matrix[right]
                        * (
                            matrix[right].clamp_min(1.0e-8).log()
                            - midpoint.clamp_min(1.0e-8).log()
                        )
                    ).sum()
                )
            )
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    normalized = singular / singular.sum().clamp_min(1.0e-8)
    effective_rank = torch.exp(
        -(normalized * normalized.clamp_min(1.0e-8).log()).sum()
    )
    return {
        "generalized_js": float(generalized_js),
        "mean_pairwise_js": float(torch.stack(pairwise).mean()),
        "effective_rank": float(effective_rank),
        "singular_values": singular.tolist(),
    }


def _update_route_statistics(
    model,
    alpha_list,
    profile_sums,
    profile_counts,
    historical_sums,
    historical_squares,
    historical_counts,
    num_bins,
):
    head_alphas = capture_head_alpha_list(model, alpha_list)
    for block, head_alpha in enumerate(head_alphas):
        if head_alpha is None:
            continue
        probability = head_alpha.detach().float()
        historical = 1.0 - probability[..., -1]
        historical_sums[block] += float(historical.sum())
        historical_squares[block] += float(historical.square().sum())
        historical_counts[block] += int(historical.numel())
        if probability.shape[-1] < 2:
            continue
        basis = _triangular_depth_basis(
            probability.shape[-1],
            num_bins,
            device=probability.device,
            dtype=probability.dtype,
        )
        token_profiles = torch.einsum("bthn,nk->bthk", probability, basis)
        profile_sums[block] += token_profiles.sum(dim=(0, 1, 2)).cpu()
        profile_counts[block] += int(
            token_profiles.shape[0]
            * token_profiles.shape[1]
            * token_profiles.shape[2]
        )
    return head_alphas


def _apply_query_intervention(model, mode: str, original: torch.Tensor):
    model.router.w_query.copy_(original)
    if mode == "zero_query":
        model.router.w_query[2:].zero_()
    elif mode == "cyclic_query_permutation":
        model.router.w_query[2:].copy_(original[2:].roll(1, dims=0))
    elif mode != "restore":
        raise ValueError(mode)


def _empty_totals():
    return {
        "tokens": 0.0,
        "ce_sum": 0.0,
        "teacher_ce_sum": 0.0,
        "kl_sum": 0.0,
        "top1_agreement_count": 0.0,
    }


def _summarize(values):
    tokens = values["tokens"]
    if tokens <= 0:
        return None
    return {
        "assistant_tokens": int(tokens),
        "student_ce": values["ce_sum"] / tokens,
        "teacher_ce": values["teacher_ce_sum"] / tokens,
        "student_minus_teacher_ce": (
            values["ce_sum"] - values["teacher_ce_sum"]
        ) / tokens,
        "teacher_kl": values["kl_sum"] / tokens,
        "teacher_top1_agreement": values["top1_agreement_count"] / tokens,
    }


def _gate_parameter_statistics(model):
    if not hasattr(model, "residual_strength_gates"):
        return None
    weights = torch.cat(
        [module.weight.detach().float().reshape(-1) for module in model.residual_strength_gates.values()]
    )
    biases = torch.cat(
        [module.bias.detach().float().reshape(-1) for module in model.residual_strength_gates.values()]
    )
    return {
        "additional_parameters": int(weights.numel() + biases.numel()),
        "weight_rms": float(weights.square().mean().sqrt()),
        "weight_max_abs": float(weights.abs().max()),
        "bias_rms": float(biases.square().mean().sqrt()),
        "bias_max_abs": float(biases.abs().max()),
    }


def _update_gate_factor_totals(model, totals):
    if not hasattr(model, "residual_strength_gates"):
        return
    for block, factor in model.residual_strength_gate_values_by_block.items():
        value = factor.detach().float()
        totals[block]["count"] += int(value.numel())
        totals[block]["sum"] += float(value.sum())
        totals[block]["square_sum"] += float(value.square().sum())
        totals[block]["min"] = min(totals[block]["min"], float(value.min()))
        totals[block]["max"] = max(totals[block]["max"], float(value.max()))


def _summarize_gate_factors(totals):
    result = {}
    for block in sorted(totals):
        values = totals[block]
        count = values["count"]
        mean = values["sum"] / count
        variance = max(values["square_sum"] / count - mean * mean, 0.0)
        result[str(block)] = {
            "count": int(count),
            "mean": mean,
            "std": variance ** 0.5,
            "min": values["min"],
            "max": values["max"],
        }
    return result


def _selected_skip_effect(full_logits, skipped_logits, decode_mask):
    positions = torch.where(decode_mask[0])[0]
    if positions.numel() == 0:
        return positions, torch.empty(0), torch.empty(0, dtype=torch.bool)
    reference = full_logits[0, positions].float()
    perturbed = skipped_logits[0, positions].float()
    reference_probability = F.softmax(reference, dim=-1)
    kl = F.kl_div(
        F.log_softmax(perturbed, dim=-1),
        reference_probability,
        reduction="none",
    ).sum(dim=-1)
    flip = reference.argmax(dim=-1).ne(perturbed.argmax(dim=-1))
    return positions.cpu(), kl.cpu(), flip.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="/data/Minko/datasets")
    parser.add_argument("--data-mix", default="v3")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--intervention-samples", type=int, default=32)
    parser.add_argument("--gate-ablation-samples", type=int, default=32)
    parser.add_argument("--risk-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=259123)
    parser.add_argument("--max-seq", type=int, default=1536)
    parser.add_argument("--max-image-pixels", type=int, default=802816)
    parser.add_argument("--role-profile-bins", type=int, default=4)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.gpu not in {0, 1, 2, 3, 4, 5}:
        parser.error("GPU must be one of 0,1,2,3,4,5; cards 6 and 7 are forbidden")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_file = open(output_dir / "heldout_rows.jsonl", "w", encoding="utf-8")
    risk_file = open(output_dir / "skip_risk_rows.jsonl", "w", encoding="utf-8")
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(args.model_path)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and args.max_image_pixels > 0:
        size = dict(getattr(image_processor, "size", {}) or {})
        size["longest_edge"] = int(args.max_image_pixels)
        image_processor.size = size

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
    model, checkpoint = load_role_retrofit(
        base,
        args.checkpoint,
        device=device,
        dtype=dtype,
    )
    model.eval()
    has_gate = bool(checkpoint["config"].get("residual_strength_gate", False))
    if not has_gate:
        args.gate_ablation_samples = 0
        args.risk_samples = 0

    stream = _encoded_stream(args, processor)
    totals = defaultdict(float)
    by_source = defaultdict(lambda: defaultdict(float))
    query_interventions = {
        "zero_query": defaultdict(float),
        "cyclic_query_permutation": defaultdict(float),
    }
    gate_interventions = {
        "identity_gate": defaultdict(float),
        "cyclic_gate_permutation": defaultdict(float),
    }
    original_query = model.router.w_query.detach().clone()
    profile_sums = defaultdict(lambda: torch.zeros(args.role_profile_bins))
    profile_counts = defaultdict(int)
    historical_sums = defaultdict(float)
    historical_squares = defaultdict(float)
    historical_counts = defaultdict(int)
    gate_factor_totals = defaultdict(
        lambda: {"count": 0, "sum": 0.0, "square_sum": 0.0, "min": float("inf"), "max": -float("inf")}
    )
    start = time.time()

    for index in range(args.samples):
        inputs, labels, source_name, modality = next(stream)
        inputs = move_inputs(inputs, device)
        labels = labels.unsqueeze(0).to(device)
        if has_gate:
            set_residual_strength_gate_mode(model, "learned")
        with torch.no_grad():
            teacher_output = teacher(**inputs, use_cache=False)
            full_output = model(
                **inputs,
                labels=labels,
                return_alpha=True,
                return_native_features=has_gate and index < args.risk_samples,
            )
        metrics = _token_metrics(
            full_output.logits,
            teacher_output.logits,
            labels,
            args.kd_temperature,
        )
        for key, value in metrics.items():
            totals[key] += value
            by_source[source_name][key] += value
        head_alphas = _update_route_statistics(
            model,
            full_output.alpha_list,
            profile_sums,
            profile_counts,
            historical_sums,
            historical_squares,
            historical_counts,
            args.role_profile_bins,
        )
        _update_gate_factor_totals(model, gate_factor_totals)
        full_gate_values = {
            block: value.detach().float().clone()
            for block, value in getattr(
                model, "residual_strength_gate_values_by_block", {}
            ).items()
        }

        row = {
            "index": index,
            "source": source_name,
            "modality": modality,
            "sequence_length": int(inputs["input_ids"].shape[1]),
            **metrics,
        }
        if index < args.intervention_samples:
            for mode in query_interventions:
                with torch.no_grad():
                    _apply_query_intervention(model, mode, original_query)
                    intervened = model(**inputs, labels=labels)
                intervention_metrics = _token_metrics(
                    intervened.logits,
                    teacher_output.logits,
                    labels,
                    args.kd_temperature,
                )
                row[f"{mode}_ce_sum"] = intervention_metrics["ce_sum"]
                row[f"{mode}_kl_sum"] = intervention_metrics["kl_sum"]
                for key, value in intervention_metrics.items():
                    query_interventions[mode][key] += value
            with torch.no_grad():
                _apply_query_intervention(model, "restore", original_query)

        if has_gate and index < args.gate_ablation_samples:
            for label, mode in (
                ("identity_gate", "identity"),
                ("cyclic_gate_permutation", "cyclic"),
            ):
                set_residual_strength_gate_mode(model, mode)
                with torch.no_grad():
                    intervened = model(**inputs, labels=labels)
                intervention_metrics = _token_metrics(
                    intervened.logits,
                    teacher_output.logits,
                    labels,
                    args.kd_temperature,
                )
                row[f"{label}_ce_sum"] = intervention_metrics["ce_sum"]
                row[f"{label}_kl_sum"] = intervention_metrics["kl_sum"]
                for key, value in intervention_metrics.items():
                    gate_interventions[label][key] += value
            set_residual_strength_gate_mode(model, "learned")

        if has_gate and index < args.risk_samples:
            _, _, decode_mask = _prediction_masks(labels)
            native_by_block = full_output.native_features_by_block
            for block, offsets in CANONICAL_MICRO_ACTIONS.items():
                head_alpha = head_alphas[block]
                gate_factor = full_gate_values[block].squeeze(-1)
                native = native_by_block[block]
                recent = head_alpha[..., -1].mean(dim=-1).detach().float()
                correction = native["correction_ratio"].detach().float()
                for offset in offsets:
                    with torch.no_grad():
                        skipped = model(
                            **inputs,
                            labels=labels,
                            token_layer_skip_masks={
                                block: {offset: decode_mask}
                            },
                        )
                    positions, token_kl, token_flip = _selected_skip_effect(
                        full_output.logits,
                        skipped.logits,
                        decode_mask,
                    )
                    for local_index, position in enumerate(positions.tolist()):
                        record = {
                            "sample_index": index,
                            "source": source_name,
                            "modality": modality,
                            "block": block,
                            "layer_offset": offset,
                            "token_position": position,
                            "w_recent": float(recent[0, position].cpu()),
                            "historical_mass": float(1.0 - recent[0, position].cpu()),
                            "gate_factor": float(gate_factor[0, position].cpu()),
                            "gate_deviation": float(abs(gate_factor[0, position].cpu() - 1.0)),
                            "correction_ratio": float(correction[0, position].cpu()),
                            "full_to_skip_kl": float(token_kl[local_index]),
                            "top1_flip": bool(token_flip[local_index]),
                        }
                        risk_file.write(json.dumps(record, sort_keys=True) + "\n")
                    risk_file.flush()

        rows_file.write(json.dumps(row, sort_keys=True) + "\n")
        rows_file.flush()
        if index == 0 or (index + 1) % 8 == 0:
            print(
                f"[heldout-rsg] {index + 1}/{args.samples} "
                f"ce={totals['ce_sum'] / totals['tokens']:.5f} "
                f"kl={totals['kl_sum'] / totals['tokens']:.5f} "
                f"risk_samples={min(index + 1, args.risk_samples)} "
                f"elapsed={time.time() - start:.0f}s",
                flush=True,
            )

    rows_file.close()
    risk_file.close()
    block_profiles = {
        block: profile_sums[block] / float(profile_counts[block])
        for block in sorted(profile_sums)
    }
    role_metrics = _profile_metrics(
        torch.stack([block_profiles[block] for block in block_profiles])
    )
    historical_mass = {}
    for block in sorted(historical_counts):
        count = historical_counts[block]
        mean = historical_sums[block] / count
        variance = max(historical_squares[block] / count - mean * mean, 0.0)
        historical_mass[str(block)] = {
            "mean": mean,
            "std": variance ** 0.5,
            "count": count,
        }

    original_query_subset = _empty_totals()
    original_gate_subset = _empty_totals()
    with open(output_dir / "heldout_rows.jsonl", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row["index"] < args.intervention_samples:
                for key in original_query_subset:
                    original_query_subset[key] += row[key]
            if row["index"] < args.gate_ablation_samples:
                for key in original_gate_subset:
                    original_gate_subset[key] += row[key]

    query_original = _summarize(original_query_subset)
    query_summary = {"original_subset": query_original}
    for mode, values in query_interventions.items():
        summary = _summarize(values)
        if summary is not None:
            summary["delta_ce_vs_original_subset"] = (
                summary["student_ce"] - query_original["student_ce"]
            )
            summary["delta_kl_vs_original_subset"] = (
                summary["teacher_kl"] - query_original["teacher_kl"]
            )
        query_summary[mode] = summary

    gate_summary = None
    if has_gate and args.gate_ablation_samples > 0:
        gate_original = _summarize(original_gate_subset)
        gate_summary = {"learned_gate_subset": gate_original}
        for mode, values in gate_interventions.items():
            summary = _summarize(values)
            summary["delta_ce_vs_learned_gate"] = (
                summary["student_ce"] - gate_original["student_ce"]
            )
            summary["delta_kl_vs_learned_gate"] = (
                summary["teacher_kl"] - gate_original["teacher_kl"]
            )
            gate_summary[mode] = summary

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "variant": checkpoint["config"].get("router_variant", "attnres"),
        "method_id": checkpoint["config"].get("method_id", "unknown"),
        "routing_heads": checkpoint["config"].get("routing_heads", 1),
        "residual_strength_gate": has_gate,
        "samples": args.samples,
        "intervention_samples": args.intervention_samples,
        "gate_ablation_samples": args.gate_ablation_samples,
        "risk_samples": args.risk_samples,
        "seed": args.seed,
        "data_mix": args.data_mix,
        "full_path_primary_evaluation": True,
        "overall": _summarize(totals),
        "by_source": {
            source: _summarize(values)
            for source, values in sorted(by_source.items())
        },
        "role_metrics": {
            **role_metrics,
            "block_profiles": {
                str(block): profile.tolist()
                for block, profile in block_profiles.items()
            },
            "historical_mass_by_block": historical_mass,
        },
        "native_query_interventions": query_summary,
        "residual_strength_gate_interventions": gate_summary,
        "learned_residual_strength_gate_parameters": _gate_parameter_statistics(model),
        "learned_residual_strength_factors": _summarize_gate_factors(gate_factor_totals),
        "skip_relevance_actions": {
            str(block): list(offsets)
            for block, offsets in CANONICAL_MICRO_ACTIONS.items()
        } if has_gate else None,
        "skip_relevance_is_posthoc_only": bool(has_gate),
        "elapsed_seconds": time.time() - start,
    }
    with open(
        output_dir / "HELDOUT_FULLPATH_RSG_SUMMARY.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))

    close_stream = getattr(stream, "close", None)
    if close_stream is not None:
        close_stream()
    del model, base, teacher, processor, checkpoint
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
