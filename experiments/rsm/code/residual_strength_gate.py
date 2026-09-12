"""Token-conditioned residual-strength calibration for native AttnRes.

The module is part of the Full AttnRes path.  It scales the existing
adapter-produced AttnRes correction with one zero-initialized scalar predictor
per target block.  It never observes a skip action, skip loss, compute target,
or counterfactual layer-removal label.

For block ``b`` with the original correction ``u_b``:

    g_b = 1 + scale * tanh(v_b^T RMSNorm(u_b) + c_b)
    x_b = h_b + g_b * u_b

Zero initialization gives ``g_b == 1`` exactly, so installing the component
does not change the original AttnRes forward before adaptation.
"""
from __future__ import annotations

import math
import types
from dataclasses import dataclass

import torch
import torch.nn as nn

from qwen3vl_attnres_retrofit import _rms_norm


@dataclass(frozen=True)
class ResidualStrengthGateInstallReport:
    gated_blocks: tuple[int, ...]
    gate_scale: float
    additional_parameters: int


def install_residual_strength_gate(
    model,
    *,
    gate_scale: float = 0.5,
    gated_blocks: tuple[int, ...] | None = None,
) -> ResidualStrengthGateInstallReport:
    """Install the Full-path gate without modifying the frozen backbone."""
    if hasattr(model, "residual_strength_gates"):
        raise RuntimeError("residual-strength gate is already installed")
    gate_scale = float(gate_scale)
    if not 0.0 < gate_scale < 1.0:
        raise ValueError("gate_scale must lie strictly between zero and one")
    if gated_blocks is None:
        # Block 0 has only the embedding source, so its routed delta is exactly
        # zero and a learned strength gate there would be an unused parameter.
        gated_blocks = tuple(range(1, int(model.num_blocks)))
    gated_blocks = tuple(sorted(set(int(block) for block in gated_blocks)))
    if not gated_blocks or any(
        block <= 0 or block >= int(model.num_blocks) for block in gated_blocks
    ):
        raise ValueError("gated_blocks must be a non-empty subset of blocks 1..L-1")

    device = model.gamma.device
    dtype = model.gamma.dtype
    modules = nn.ModuleDict()
    for block in gated_blocks:
        predictor = nn.Linear(int(model.hidden_size), 1, bias=True)
        nn.init.zeros_(predictor.weight)
        nn.init.zeros_(predictor.bias)
        modules[str(block)] = predictor
    model.residual_strength_gates = modules.to(device=device, dtype=dtype)
    model.residual_strength_gate_scale = gate_scale
    model.residual_strength_gate_blocks = gated_blocks
    model.residual_strength_gate_mode = "learned"
    model.residual_strength_gate_values_by_block = {}

    original_compute = model._compute_block_input

    def _gated_compute_block_input(
        self,
        block_idx,
        prev_block,
        completed,
        collect_alpha,
        compute_entropy=False,
        compute_correction_ratio=False,
        compute_native_features=False,
        return_fused_native_inputs=False,
        minimal_native_features=False,
    ):
        result = original_compute(
            block_idx,
            prev_block,
            completed,
            collect_alpha,
            compute_entropy=compute_entropy,
            compute_correction_ratio=compute_correction_ratio,
            compute_native_features=compute_native_features,
            return_fused_native_inputs=return_fused_native_inputs,
        )
        (
            corrected,
            alpha,
            routed,
            entropy,
            correction_ratio,
            native_features,
            fused_native_inputs,
        ) = result
        block_idx = int(block_idx)
        key = str(block_idx)
        if key not in self.residual_strength_gates:
            return result

        pre_gate_update = corrected - prev_block
        mode = str(self.residual_strength_gate_mode)
        if mode == "identity":
            factor = torch.ones_like(pre_gate_update[..., :1])
        else:
            source_block = block_idx
            if mode == "cyclic":
                blocks = self.residual_strength_gate_blocks
                source_block = blocks[(blocks.index(block_idx) + 1) % len(blocks)]
            elif mode != "learned":
                raise ValueError(f"unknown residual-strength gate mode: {mode}")
            predictor = self.residual_strength_gates[str(source_block)]
            gate_input = _rms_norm(pre_gate_update)
            # The zero-initialized projection receives an RMS-normalized
            # hidden vector. Scale its dot product exactly like attention so
            # one Adam step cannot immediately saturate tanh in width 2048.
            raw = predictor(gate_input) / math.sqrt(float(self.hidden_size))
            factor = 1.0 + self.residual_strength_gate_scale * torch.tanh(
                raw.float()
            ).to(pre_gate_update.dtype)

        gated_update = factor * pre_gate_update
        corrected = prev_block + gated_update
        self.residual_strength_gate_values_by_block[block_idx] = factor.detach()

        if native_features is not None:
            if native_features.get("_minimal_native_features", False):
                native_features = {
                    **native_features,
                    "residual_strength_factor": factor.squeeze(-1).float(),
                }
            else:
                previous_scale = (
                    prev_block.float().pow(2).mean(dim=-1).sqrt().clamp_min(1.0e-8)
                )
                native_features = {
                    **native_features,
                    "correction_ratio": (
                        gated_update.float().pow(2).mean(dim=-1).sqrt()
                        / previous_scale
                    ),
                    "residual_strength_factor": factor.squeeze(-1).float(),
                }
        if compute_correction_ratio:
            update_rms = gated_update.float().pow(2).mean().sqrt()
            base_rms = prev_block.float().pow(2).mean().sqrt().clamp_min(1.0e-8)
            correction_ratio = (update_rms / base_rms).detach()
        return (
            corrected,
            alpha,
            routed,
            entropy,
            correction_ratio,
            native_features,
            fused_native_inputs,
        )

    model._compute_block_input = types.MethodType(
        _gated_compute_block_input,
        model,
    )
    added = sum(parameter.numel() for parameter in modules.parameters())
    return ResidualStrengthGateInstallReport(
        gated_blocks=gated_blocks,
        gate_scale=gate_scale,
        additional_parameters=int(added),
    )


def residual_strength_gate_statistics(model) -> dict:
    """Summarize the factors captured by the latest Full forward."""
    result = {}
    for block in model.residual_strength_gate_blocks:
        value = model.residual_strength_gate_values_by_block.get(block)
        if value is None:
            continue
        value = value.detach().float().reshape(-1)
        result[str(block)] = {
            "count": int(value.numel()),
            "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)),
            "min": float(value.min()),
            "max": float(value.max()),
        }
    return result


def set_residual_strength_gate_mode(model, mode: str) -> None:
    mode = str(mode)
    if mode not in {"learned", "identity", "cyclic"}:
        raise ValueError(mode)
    if not hasattr(model, "residual_strength_gates"):
        if mode != "identity":
            raise RuntimeError("model has no residual-strength gate")
        return
    model.residual_strength_gate_mode = mode
