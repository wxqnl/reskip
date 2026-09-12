"""Checkpoint construction helpers for residual-strength-gated AttnRes."""
from __future__ import annotations

from pathlib import Path

import torch

from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit
from role_structured_router import install_role_router
from residual_strength_gate import install_residual_strength_gate


def load_role_retrofit(
    base_model,
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = Qwen3VLAttnResRetrofit(
        base_model,
        num_blocks=int(config.get("num_blocks", 7)),
        skippable_blocks=checkpoint.get("skippable_blocks"),
        adapter_rank=int(config.get("adapter_rank", 256)),
        no_adapter=bool(config.get("no_adapter", False)),
        bridge_mode=config.get("bridge_mode", "adapter"),
        compat_peak=float(config.get("compat_peak", 1.0)),
        identity_anchored_source_calibration=bool(
            config.get("identity_anchored_source_calibration", False)
        ),
    ).move_retrofit_modules(device=device, dtype=dtype)
    install_role_router(
        model,
        variant=str(config.get("router_variant", "attnres")),
        num_heads=int(config.get("routing_heads", 4)),
        num_basis=int(config.get("role_basis_count", 4) or 4),
        role_scale=float(config.get("role_logit_scale", 1.0)),
        random_basis_seed=int(config.get("random_basis_seed", 314159)),
    )
    if bool(config.get("residual_strength_gate", False)):
        install_residual_strength_gate(
            model,
            gate_scale=float(config.get("residual_strength_gate_scale", 0.5)),
            gated_blocks=tuple(
                int(block)
                for block in config.get(
                    "residual_strength_gate_blocks",
                    range(1, int(config.get("num_blocks", 7))),
                )
            ),
        )
    model.router.load_state_dict(checkpoint["router"], strict=True)
    model.adapters.load_state_dict(checkpoint["adapters"], strict=True)
    if bool(config.get("residual_strength_gate", False)):
        model.residual_strength_gates.load_state_dict(
            checkpoint["residual_strength_gates"],
            strict=True,
        )
    with torch.no_grad():
        model.gamma.copy_(checkpoint["gamma"].to(device=device, dtype=dtype))
    model.freeze_base()
    model.eval()
    return model, checkpoint
