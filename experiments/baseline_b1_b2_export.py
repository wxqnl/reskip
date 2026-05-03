"""Export 340M skip-strategy baselines for B1 (decision rule) + B2 (threshold rule).

Cells produced:
  Dynamic at q=0.5, P={3,5}, M=1 (matched ~12.5% block skip rate):
    - recent_weight_gt   (current ReSkip rule)
    - entropy_lt         (skip when phase1 router is confident)
    - recent_minus_embed_gt  (skip when recent dominates over embed)

  Static at every-token, M=1:
    - static_p3_every    (skip position 3 always)
    - static_p5_every    (skip position 5 always)

Used together with attnres_full (no-skip baseline, already exists) and a runtime
random-skip wrapper for the random-rate-matched baseline.

Re-uses calibrate() from export_ablation_skip_models.py but extended to capture
recent_weight, embed_weight, and entropy in one pass (instead of 3 loads).
"""
from __future__ import annotations

import argparse
import torch
from pathlib import Path

from flame_reskip_common import (
    build_text_dataloader,
    load_model_and_tokenizer,
    save_json,
)


DISABLE = 1e9
NUM_POSITIONS = 8  # 340M reskip has 8 attn-res blocks


def _entry_metric(entry: dict, strategy: str):
    recent = entry.get("avg_phase1_recent_weight")
    embed = entry.get("avg_phase1_embed_weight")
    entropy = entry.get("avg_phase1_entropy")
    if strategy == "recent_weight_gt":
        return recent
    if strategy == "entropy_lt":
        return entropy
    if strategy == "recent_minus_embed_gt":
        if recent is None or embed is None:
            return None
        return float(recent) - float(embed)
    raise ValueError(f"unknown strategy {strategy}")


@torch.no_grad()
def calibrate_all(model, dataloader, device, *, num_batches, num_positions):
    """Single forward pass per batch; collect per-position values for all 3 strategies."""
    disabled = [DISABLE] * num_positions
    per_pos = {
        s: [[] for _ in range(num_positions)]
        for s in ("recent_weight_gt", "entropy_lt", "recent_minus_embed_gt")
    }
    n = 0
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        am = batch.get("attention_mask")
        cu = batch.get("cu_seqlens")
        if am is not None:
            am = am.to(device)
            if torch.all(am):
                am = None
        if cu is not None:
            cu = cu.to(device)
        out = model(
            input_ids=ids, attention_mask=am, labels=labels,
            cu_seqlens=cu, use_cache=False, return_dict=True,
            return_routing_info=True, enable_skipping=False,
            dynamic_skip_strategy="recent_weight_gt",
            dynamic_skip_granularity="block",
            dynamic_skip_probe_mode="all",  # 'all' so we capture entropy + embed too
            dynamic_skip_position_thresholds=disabled,
            dynamic_skip_max_skips=0,
        )
        ri = getattr(out, "routing_info", None)
        if ri:
            for entry in ri.get("execution_trace", []):
                pos = int(entry["position"])
                if pos >= num_positions:
                    continue
                for strategy in per_pos:
                    val = _entry_metric(entry, strategy)
                    if val is not None:
                        per_pos[strategy][pos].append(float(val))
        n += 1
        if n >= num_batches:
            break
    return per_pos


def quantile_thresholds(values_per_pos, quantile, *, direction, allowed_positions):
    """Return per-position thresholds. direction='gt' or 'lt' decides quantile direction.
    For 'gt' strategy, we want skip when val>thr → use UPPER quantile (q=0.5 → median).
    For 'lt' strategy, we want skip when val<thr → use LOWER quantile (q=0.5 → median).
    Same q value yields ~equal firing rate either way at q=0.5.
    """
    out = []
    for pos, values in enumerate(values_per_pos):
        if pos not in allowed_positions or pos == 0 or pos == len(values_per_pos) - 1 or not values:
            out.append(DISABLE)
            continue
        t = torch.tensor(values, dtype=torch.float32)
        # In both gt and lt cases, q=0.5 means median; pick that.
        # For 'gt' strategy (skip when val>thr): use 1-q for upper-tail interpretation? No.
        # We define q as "fraction of tokens that fire skip at this position".
        # gt strategy: skip if val > thr → 50% fire when thr = median.
        # lt strategy: skip if val < thr → 50% fire when thr = median.
        # So same q → same fire rate; both use torch.quantile(t, 1-q) for gt, q for lt.
        if direction == "gt":
            # threshold at (1-q) percentile so q fraction exceeds it
            q_value = 1.0 - quantile
        else:
            q_value = quantile
        out.append(float(torch.quantile(t, q_value).item()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="flame/saves/reskip_transformer-340M")
    p.add_argument("--dataset", default="/home/user01/Minko/datasets/fineweb_edu_100BT")
    p.add_argument("--seq_len", type=int, default=8192)
    p.add_argument("--cal_batches", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_base", default="outputs")
    p.add_argument("--quantile", type=float, default=0.5,
                   help="Per-position skip-firing rate at allowed positions; 0.5 ≈ 12.5% block rate with M=1.")
    p.add_argument("--positions", default="3,5",
                   help="Comma-separated allowed positions; default {3,5}.")
    p.add_argument("--max_skips", type=int, default=1)
    args = p.parse_args()

    allowed = set(int(x) for x in args.positions.split(","))
    print(f"[export] model={args.model_path} q={args.quantile} positions={allowed} M={args.max_skips}")

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    decoder = model.model
    assert decoder.num_block_positions == NUM_POSITIONS, decoder.num_block_positions

    loader = build_text_dataloader(
        tokenizer=tokenizer, dataset=args.dataset, dataset_name=None,
        dataset_split="train", data_dir=None, data_files=None,
        seq_len=args.seq_len, context_len=args.seq_len,
        batch_size=1, num_workers=2, streaming=True, varlen=False, seed=0,
    )

    print("[export] calibrating per-position metrics...")
    per_pos = calibrate_all(model, loader, args.device,
                            num_batches=args.cal_batches,
                            num_positions=NUM_POSITIONS)
    for s in per_pos:
        print(f"  {s}: per-pos counts = {[len(v) for v in per_pos[s]]}")

    # Tag with q*100 for path
    qtag = f"q{int(args.quantile*100):03d}"
    out_root = Path(args.output_base)

    # ---- 3 dynamic cells (B1.b/B2.a, B2.b, B2.c) ----
    dynamic_cells = [
        ("recent_weight_gt", "gt"),
        ("entropy_lt", "lt"),
        ("recent_minus_embed_gt", "gt"),
    ]
    for strategy, direction in dynamic_cells:
        thresholds = quantile_thresholds(per_pos[strategy], args.quantile,
                                         direction=direction, allowed_positions=allowed)
        out_dir = out_root / f"reskip_340M_b1b2_{strategy}_{qtag}_M{args.max_skips}"
        out_dir.mkdir(parents=True, exist_ok=True)

        decoder.clear_skip_keep_mask()
        decoder.clear_dynamic_skip_policy()
        decoder.set_dynamic_skip_policy(
            strategy=strategy,
            probe_mode="all",
            position_thresholds=thresholds,
            max_skips=args.max_skips,
        )
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        save_json(out_dir / "skip_policy.json", {
            "strategy": strategy,
            "probe_mode": "all",
            "positions": sorted(allowed),
            "max_skips": args.max_skips,
            "quantile": args.quantile,
            "thresholds": thresholds,
        })
        print(f"[export] dynamic: {out_dir}")
        print(f"  thresholds={[f'{t:.4f}' for t in thresholds]}")

    # ---- 2 static cells (B1.c, B1.d) ----
    static_cells = [
        ("static_p3_every", [True, True, True, False, True, True, True, True]),
        ("static_p5_every", [True, True, True, True, True, False, True, True]),
    ]
    for name, keep_mask in static_cells:
        out_dir = out_root / f"reskip_340M_b1_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        decoder.clear_dynamic_skip_policy()
        decoder.clear_skip_keep_mask()
        decoder.set_skip_keep_mask(keep_mask)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        save_json(out_dir / "skip_policy.json", {
            "strategy": "static_keep_mask",
            "keep_mask": [int(b) for b in keep_mask],
            "rate_block": (NUM_POSITIONS - sum(keep_mask)) / NUM_POSITIONS,
        })
        print(f"[export] static: {out_dir}, rate={(NUM_POSITIONS-sum(keep_mask))/NUM_POSITIONS:.3f}")

    # Reset runtime state for cleanliness
    decoder.clear_dynamic_skip_policy()
    decoder.clear_skip_keep_mask()
    print("[export] done.")


if __name__ == "__main__":
    main()
