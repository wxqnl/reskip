"""Build point-estimate and seed-aggregated tables for all four model families."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path("/data/Minko/experiments/reskip_llm_benchmark_20260911")
METRICS = (
    ("LAMBADA", "lambada_openai", "acc,none"),
    ("HellaSwag", "hellaswag", "acc_norm,none"),
    ("PIQA", "piqa", "acc_norm,none"),
    ("ARC-E", "arc_easy", "acc_norm,none"),
    ("ARC-C", "arc_challenge", "acc_norm,none"),
    ("OpenBookQA", "openbookqa", "acc_norm,none"),
    ("WinoGrande", "winogrande", "acc,none"),
)
POINT_ROWS = (
    ("Qwen3-VL-2B", "Base", None, "qwen3vl_2b_base"),
    ("Qwen3-VL-2B", "Full AttnRes", 0, "qwen3vl_2b_full_attnres_seed0"),
    ("Qwen3-VL-2B", "Full RSM-AttnRes", 0, "qwen3vl_2b_full_rsm_seed0"),
    ("Qwen3-VL-4B", "Base", None, "qwen3vl_4b_base"),
    ("Qwen3-VL-4B", "Full AttnRes", 2, "qwen3vl_4b_full_attnres_seed2"),
    ("Qwen3-VL-4B", "Full RSM-AttnRes", 2, "qwen3vl_4b_full_rsm_seed2"),
    ("InternVL3.5-2B", "Base", None, "internvl3p5_2b_base"),
    ("InternVL3.5-2B", "Full AttnRes", 0, "internvl3p5_2b_full_attnres_seed0"),
    ("InternVL3.5-2B", "Full RSM-AttnRes", 0, "internvl3p5_2b_full_rsm_seed0"),
    ("SmolVLM2-2.2B", "Base", None, "smolvlm2_2p2b_base"),
    *tuple(("SmolVLM2-2.2B", "Full AttnRes", seed, f"smolvlm2_2p2b_full_attnres_seed{seed}") for seed in range(3)),
    *tuple(("SmolVLM2-2.2B", "Full RSM-AttnRes", seed, f"smolvlm2_2p2b_full_rsm_seed{seed}") for seed in range(3)),
)


def load_row(family: str, method: str, seed: int | None, label: str) -> dict:
    path = ROOT / "results" / label / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing completed cell: {path}")
    payload = json.loads(path.read_text())
    scores = {
        display: float(payload["results"][task][metric]) * 100.0
        for display, task, metric in METRICS
    }
    return {
        "Family": family,
        "Method": method,
        "Seed": seed,
        **scores,
        "Average": statistics.fmean(scores.values()),
        "Source": str(path),
    }


def main() -> None:
    rows = [load_row(*spec) for spec in POINT_ROWS]
    out = ROOT / "tables"
    out.mkdir(exist_ok=True)
    with (out / "llm_benchmark_extended_seedwise.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = []
    for family in ("Qwen3-VL-2B", "Qwen3-VL-4B", "InternVL3.5-2B", "SmolVLM2-2.2B"):
        for method in ("Base", "Full AttnRes", "Full RSM-AttnRes"):
            group = [row for row in rows if row["Family"] == family and row["Method"] == method]
            means = {
                display: statistics.fmean(float(row[display]) for row in group)
                for display, *_ in METRICS
            }
            macro_values = [float(row["Average"]) for row in group]
            grouped.append(
                {
                    "Family": family,
                    "Method": method,
                    "Seeds": len(group),
                    **means,
                    "Average": statistics.fmean(macro_values),
                    "AverageSD": statistics.stdev(macro_values) if len(group) > 1 else None,
                }
            )

    with (out / "llm_benchmark_extended.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)

    fields = [display for display, *_ in METRICS]
    lines = [
        "| Family | Method | n | " + " | ".join(fields) + " | Average |",
        "|---|---|---:|" + "---:|" * (len(fields) + 1),
    ]
    for row in grouped:
        metrics = " | ".join(f"{row[field]:.2f}" for field in fields)
        average = f"{row['Average']:.2f}"
        if row["AverageSD"] is not None:
            average += f" ± {row['AverageSD']:.2f}"
        lines.append(
            f"| {row['Family']} | {row['Method']} | {row['Seeds']} | {metrics} | {average} |"
        )
    path = out / "llm_benchmark_extended.md"
    path.write_text("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    main()
