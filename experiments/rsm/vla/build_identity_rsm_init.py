#!/usr/bin/env python3
"""Build an RSM-compatible state from trained AttnRes plus identity RSM gates."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attnres-state", required=True)
    parser.add_argument("--rsm-template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method-id", required=True)
    return parser.parse_args()


def require_mapping(checkpoint: dict, key: str) -> dict:
    value = checkpoint.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"checkpoint section {key!r} is missing or empty")
    return value


def main() -> None:
    args = parse_args()
    attnres_path = Path(args.attnres_state).resolve()
    template_path = Path(args.rsm_template).resolve()
    output_path = Path(args.output).resolve()

    attnres = torch.load(attnres_path, map_location="cpu", weights_only=True)
    template = torch.load(template_path, map_location="cpu", weights_only=True)
    if not isinstance(attnres, dict) or not isinstance(template, dict):
        raise TypeError("both source checkpoints must be dictionaries")

    attnres_cfg = dict(attnres.get("config", {}) or {})
    template_cfg = dict(template.get("config", {}) or {})
    for field in ("num_blocks", "adapter_rank"):
        if attnres_cfg.get(field) != template_cfg.get(field):
            raise ValueError(
                f"source mismatch for {field}: "
                f"AttnRes={attnres_cfg.get(field)!r}, RSM={template_cfg.get(field)!r}"
            )
    if bool(attnres_cfg.get("residual_strength_gate", False)):
        raise ValueError("AttnRes source unexpectedly already contains RSM")
    if not bool(template_cfg.get("residual_strength_gate", False)):
        raise ValueError("RSM template is not marked as residual-strength gated")

    require_mapping(attnres, "router")
    require_mapping(attnres, "adapters")
    template_gates = require_mapping(template, "residual_strength_gates")

    corrected = copy.deepcopy(attnres)
    corrected["residual_strength_gates"] = {
        name: torch.zeros_like(tensor) for name, tensor in template_gates.items()
    }
    corrected_cfg = copy.deepcopy(attnres_cfg)
    corrected_cfg.update(
        {
            "residual_strength_gate": True,
            "residual_strength_gate_scale": float(
                template_cfg.get("residual_strength_gate_scale", 0.5)
            ),
            "residual_strength_gate_blocks": list(
                template_cfg["residual_strength_gate_blocks"]
            ),
            "method_id": args.method_id,
            "initialization": "trained_vlm_attnres_router_adapters_plus_identity_rsm_gate",
            "source_attnres_state": str(attnres_path),
            "source_rsm_shape_template": str(template_path),
            "rsm_gate_identity_initialized": True,
            "primary_purpose": "joint_full_path_residual_adaptation",
            "skip_specific_parameters": 0,
            "skip_supervision": False,
            "second_stage_training": False,
        }
    )
    corrected["config"] = corrected_cfg

    if any(torch.count_nonzero(value).item() for value in corrected["residual_strength_gates"].values()):
        raise AssertionError("identity RSM gates must be exactly zero initialized")
    for section in ("router", "adapters"):
        for name, source in attnres[section].items():
            if not torch.equal(source, corrected[section][name]):
                raise AssertionError(f"{section}.{name} changed while building init state")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    torch.save(corrected, output_path)
    parameter_count = sum(
        tensor.numel() for tensor in corrected["residual_strength_gates"].values()
    )
    print(f"wrote {output_path}")
    print(f"identity_rsm_parameters={parameter_count}")
    print(f"num_blocks={corrected_cfg['num_blocks']}")


if __name__ == "__main__":
    main()
