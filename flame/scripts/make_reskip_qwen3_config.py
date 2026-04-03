from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_reskip_qwen3_config(
    base_vlm_path: Path,
    *,
    use_attn_res: bool,
    attn_res_num_blocks: int,
    attn_res_temperature: float,
) -> dict:
    config_path = base_vlm_path / "config.json"
    with config_path.open() as f:
        raw_cfg = json.load(f)

    text_cfg = raw_cfg.get("text_config", raw_cfg)
    cfg = dict(text_cfg)
    cfg["model_type"] = "reskip_qwen3"
    cfg["use_attn_res"] = use_attn_res
    cfg["attn_res_num_blocks"] = attn_res_num_blocks
    cfg["attn_res_temperature"] = attn_res_temperature
    cfg["attn_res_output_norm"] = True
    cfg["fuse_norm"] = False
    cfg["fuse_cross_entropy"] = False
    cfg["fuse_linear_cross_entropy"] = False
    cfg.pop("architectures", None)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a flame-compatible reskip_qwen3 config from a local Qwen3-VL base model."
    )
    parser.add_argument("--base-vlm-path", required=True, help="Local StarVLA/Qwen3-VL base model directory.")
    parser.add_argument("--output", required=True, help="Output JSON config path.")
    parser.add_argument("--use-attn-res", action="store_true", help="Enable AttnRes routing in the output config.")
    parser.add_argument("--attn-res-num-blocks", type=int, default=8)
    parser.add_argument("--attn-res-temperature", type=float, default=1.0)
    args = parser.parse_args()

    cfg = build_reskip_qwen3_config(
        Path(args.base_vlm_path),
        use_attn_res=args.use_attn_res,
        attn_res_num_blocks=args.attn_res_num_blocks,
        attn_res_temperature=args.attn_res_temperature,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"Saved flame config to {output_path}")


if __name__ == "__main__":
    main()
