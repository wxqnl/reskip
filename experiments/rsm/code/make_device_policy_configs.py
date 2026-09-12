"""Create device-side and force-Full controls from one frozen host policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    policy = json.loads(args.policy.read_text())

    device = {
        **policy,
        "device_conditional_runtime": True,
        "device_conditional_max_positions": 8192,
        "device_fused_native_features": False,
        "selector_description": (
            "frozen RSM-ReSkip rule executed by the GPU-side conditional runtime"
        ),
    }
    force_full = {
        **device,
        "feature_lower_bounds": {
            str(block): {"w_recent": 2.0}
            for block in policy["eligible_blocks"]
        },
        "feature_upper_bounds": {
            str(block): {} for block in policy["eligible_blocks"]
        },
        "selector_description": (
            "matched device-graph control with every skip decision forced false"
        ),
    }
    (args.output_dir / "rsm_reskip_device.json").write_text(
        json.dumps(device, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "force_full_device.json").write_text(
        json.dumps(force_full, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "source_policy": str(args.policy.resolve()),
        "device_policy": str((args.output_dir / "rsm_reskip_device.json").resolve()),
        "force_full_control": str((args.output_dir / "force_full_device.json").resolve()),
        "thresholds_changed_for_device_policy": False,
    }
    (args.output_dir / "CONFIG_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
