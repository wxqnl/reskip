"""Generate the final paper-facing VLM report from immutable artifacts."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path("/data/Minko")
EXP = ROOT / "experiments/attnres_rsm_vlm_scale_generalization_20260831"
RSM2 = ROOT / "experiments/attnres_residual_strength_gate_2b_20260831"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def display(mean: float, std: float | None, n: int) -> str:
    if std is None or n == 1:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def main() -> None:
    quality = load(EXP / "reports/VLM_FINAL_QUALITY_TABLES.json")
    protocol = load(EXP / "protocol/EXPERIMENT_MATRIX.json")
    amendment4 = load(EXP / "protocol/PROTOCOL_AMENDMENT_004.json")
    seed1_screen = load(
        EXP / "reskip/qwen3vl_4b_seed1_conservative_screen/SCREEN_DECISION.json"
    )
    seed2_gate = load(
        EXP / "eval/reskip/qwen3vl_4b_seed2_capped/FORMAL_QUALITY_GATE.json"
    )
    runtime2 = load(RSM2 / "FINAL_RUNTIME_SUMMARY.json")
    fixed4_path = EXP / "runtime/qwen3vl_4b_seed2_capped/fixed64_12_per_task_r3/BENCHMARK_NATURAL_RUNTIME.json"
    natural4_path = EXP / "runtime/qwen3vl_4b_seed2_capped/natural12_per_task_r3/BENCHMARK_NATURAL_RUNTIME.json"
    equivalence4_path = EXP / "runtime/qwen3vl_4b_seed2_capped/host_device_equivalence_1_per_task_r3/HOST_DEVICE_EQUIVALENCE.json"
    skip4_path = EXP / "eval/reskip/qwen3vl_4b_seed2_capped/skip_stats.json"
    fixed4, natural4 = load(fixed4_path), load(natural4_path)
    equivalence4, skip4 = load(equivalence4_path), load(skip4_path)
    heldout2 = load(
        RSM2
        / "S1_rsg_h1_screen_seed259123_n128/HELDOUT_FULLPATH_RSG_SUMMARY.json"
    )
    heldout4 = load(
        EXP
        / "reskip/qwen3vl_4b_seed0_heldout_seed259123/HELDOUT_FULLPATH_RSG_SUMMARY.json"
    )
    increment2 = load(
        RSM2 / "screen_evidence_seed259123_v2/GATE_SKIP_INCREMENT.json"
    )
    increment4 = load(
        EXP
        / "reskip/qwen3vl_4b_seed0_heldout_seed259123/GATE_SKIP_INCREMENT.json"
    )

    rows = {
        (row["family"], row["system"], row["seed"]): row
        for row in quality["rows"]
    }
    summaries = {
        (row["family"], row["system"]): row
        for row in quality["summaries"]
    }

    def summary(family: str, system: str) -> dict:
        return summaries[(family, system)]

    block4 = skip4["avg_partial_decoder_layer_skips_per_decode_forward"] / 4.0
    fixed4_base = fixed4["macro"]["rsm_reskip_vs_base"]
    fixed4_full = fixed4["macro"]["rsm_reskip_vs_full_rsm"]
    natural4_base = natural4["macro"]["rsm_reskip_vs_base"]
    natural4_full = natural4["macro"]["rsm_reskip_vs_full_rsm"]
    runtime_rows = [
        {
            "scale": "Qwen3-VL-2B",
            "base_score": runtime2["quality"]["benchmark_macro_average"]["base_paper"],
            "full_score": runtime2["quality"]["benchmark_macro_average"]["full_rsm_attnres"],
            "reskip_score": runtime2["quality"]["benchmark_macro_average"]["rsm_reskip"],
            "block_equivalents": runtime2["quality"]["formal_rollout_block_equivalents_per_decode_token"],
            "fixed64_reskip_vs_base": runtime2["fixed_64_token_runtime"]["matched_length_speedups"]["rsm_reskip_vs_base"]["geometric_mean"],
            "fixed64_reskip_vs_base_95ci": runtime2["fixed_64_token_runtime"]["matched_length_speedups"]["rsm_reskip_vs_base"]["bootstrap_95ci"],
            "fixed64_matched_samples": runtime2["fixed_64_token_runtime"]["matched_length_speedups"]["rsm_reskip_vs_base"]["requests"],
            "fixed64_full_vs_base": (
                runtime2["fixed_64_token_runtime"]["matched_length_speedups"]["rsm_reskip_vs_base"]["geometric_mean"]
                / runtime2["fixed_64_token_runtime"]["matched_length_speedups"]["rsm_reskip_vs_full_rsm"]["geometric_mean"]
            ),
            "natural_reskip_vs_base": runtime2["natural_generation_runtime"]["rsm_reskip_vs_base_matched_length"]["pooled_sample_geometric_mean"],
            "natural_reskip_vs_base_95ci": runtime2["natural_generation_runtime"]["rsm_reskip_vs_base_matched_length"]["bootstrap_95ci"],
            "natural_matched_samples": runtime2["natural_generation_runtime"]["rsm_reskip_vs_base_matched_length"]["matched_requests"],
            "natural_full_vs_base": (
                runtime2["natural_generation_runtime"]["rsm_reskip_vs_base_matched_length"]["pooled_sample_geometric_mean"]
                / runtime2["natural_generation_runtime"]["rsm_reskip_vs_full_rsm_matched_length"]["pooled_sample_geometric_mean"]
            ),
        },
        {
            "scale": "Qwen3-VL-4B",
            "base_score": rows[("Qwen3-VL-4B", "Base", None)]["macro_average"],
            "full_score": rows[("Qwen3-VL-4B", "Full RSM-AttnRes", 2)]["macro_average"],
            "reskip_score": rows[("Qwen3-VL-4B", "RSM-ReSkip", 2)]["macro_average"],
            "block_equivalents": block4,
            "fixed64_reskip_vs_base": fixed4_base["matched_length_pooled_sample_geometric_mean"],
            "fixed64_reskip_vs_base_95ci": fixed4_base["matched_length_pooled_sample_bootstrap_95ci"],
            "fixed64_matched_samples": fixed4_base["matched_length_sample_count"],
            "fixed64_full_vs_base": (
                fixed4_base["matched_length_pooled_sample_geometric_mean"]
                / fixed4_full["matched_length_pooled_sample_geometric_mean"]
            ),
            "natural_reskip_vs_base": natural4_base["matched_length_pooled_sample_geometric_mean"],
            "natural_reskip_vs_base_95ci": natural4_base["matched_length_pooled_sample_bootstrap_95ci"],
            "natural_matched_samples": natural4_base["matched_length_sample_count"],
            "natural_full_vs_base": (
                natural4_base["matched_length_pooled_sample_geometric_mean"]
                / natural4_full["matched_length_pooled_sample_geometric_mean"]
            ),
        },
    ]

    quality_table = []
    per_task_quality_table = []
    formal_tasks = quality["formal_tasks"]
    for family in (
        "Qwen3-VL-2B",
        "Qwen3-VL-4B",
        "SmolVLM2-2.2B",
        "InternVL3.5-2B",
    ):
        for system in ("Base", "Full AttnRes", "Full RSM-AttnRes", "RSM-ReSkip"):
            record = summaries.get((family, system))
            if record is None:
                continue
            quality_table.append(
                {
                    "family": family,
                    "system": system,
                    **record,
                }
            )
            members = [
                row
                for row in quality["rows"]
                if row["family"] == family and row["system"] == system
            ]
            per_task_quality_table.append(
                {
                    "family": family,
                    "system": system,
                    "n": len(members),
                    **{
                        task: statistics.fmean(row[task] for row in members)
                        for task in formal_tasks
                    },
                    "macro_average": statistics.fmean(
                        row["macro_average"] for row in members
                    ),
                }
            )

    gate_ablation = []
    for scale, record in (("2B", heldout2), ("4B", heldout4)):
        values = record["residual_strength_gate_interventions"]
        learned = values["learned_gate_subset"]["teacher_kl"]
        gate_ablation.append(
            {
                "scale": scale,
                "learned_gate_kl": learned,
                "identity_gate_kl": values["identity_gate"]["teacher_kl"],
                "cyclic_gate_kl": values["cyclic_gate_permutation"]["teacher_kl"],
                "identity_minus_learned_kl": values["identity_gate"]["teacher_kl"] - learned,
                "cyclic_minus_learned_kl": values["cyclic_gate_permutation"]["teacher_kl"] - learned,
            }
        )

    auc_diagnostics = []
    for scale, record in (("2B", increment2), ("4B", increment4)):
        deltas = []
        for threshold in (0.01, 0.05, 0.10):
            item = record["thresholds"][str(threshold)]
            delta = item["macro_augmented_auc"] - item["macro_native_auc"]
            deltas.append(delta)
            auc_diagnostics.append(
                {
                    "scale": scale,
                    "unsafe_kl_threshold": threshold,
                    "native_auc": item["macro_native_auc"],
                    "augmented_auc": item["macro_augmented_auc"],
                    "auc_delta": delta,
                }
            )

    claims = {
        "quality_rows": quality_table,
        "per_task_quality_rows": per_task_quality_table,
        "matched_seed_deltas": quality["matched_seed_deltas"],
        "matched_seed_summaries": quality["matched_seed_summaries"],
        "runtime_rows": runtime_rows,
        "gate_ablation": gate_ablation,
        "skip_auc_diagnostics": auc_diagnostics,
        "training_constraints": protocol["hard_constraints"],
        "material_passport": protocol["material_passport"],
        "claim_boundary": [
            "Full RSM-AttnRes is trained only on the ordinary Full path; no skip branch, skip labels, compute target, external controller, or second-stage training is used.",
            "RSM is a Full-path AttnRes residual calibration mechanism. Token-level ReSkip is a frozen inference-time by-product and is evaluated only after the Full-path quality gate.",
            "The RSM factor improves cross-fitted skip-risk AUC on 2B but not on 4B; it is therefore not claimed as a universally better skip classifier.",
            "Natural-generation speed claims use matched output lengths; the fixed-64 panel isolates decode-dominant end-to-end runtime on real benchmark images and prompts.",
            "SmolVLM2 is the accepted cross-architecture Full-path confirmation: RSM improves matched Full AttnRes by +0.376 ± 0.116 pp over three seeds (3/3 positive), while its mean is only +0.024 pp above Frozen Base. The matched benefit is repeatable, but the absolute Base margin is narrow.",
            "InternVL3.5 and the current Granite Vision 4.1-4B transfer fail their frozen continuation gates. Granite RSM recovers +0.104 pp over matched Full but remains -1.895 pp below Base; both are retained as diagnostic family boundaries rather than paper-quality successes.",
            "The generic SmolVLM2 hook is a Full-path architecture-transfer probe, not a ReSkip speed claim. It cannot provide the device-side token-level conditional used by the Qwen runtime, so a request-level static-skip surrogate is intentionally not reported.",
            "The failed 4B seed-0 aggressive formal policy and the stopped seed-1 development screen are retained in the policy audit; they are not silently replaced by the final seed-2 capped policy.",
        ],
        "canonical_sources": {
            "quality": str(EXP / "reports/VLM_FINAL_QUALITY_TABLES.json"),
            "qwen2_runtime": str(RSM2 / "FINAL_RUNTIME_SUMMARY.json"),
            "qwen4_fixed64_runtime": str(fixed4_path),
            "qwen4_natural_runtime": str(natural4_path),
            "qwen4_host_device_equivalence": str(equivalence4_path),
            "qwen4_skip_stats": str(skip4_path),
            "generic_vlm_repair_audit": str(
                EXP / "reports/diagnostics/GENERIC_VLM_REPAIR_AUDIT.json"
            ),
            "figure": str(EXP / "figures/vlm_scale_method_evidence.pdf"),
        },
    }

    manifest = {
        "experiment_id": EXP.name,
        "canonical": [
            {
                "scope": "Qwen3-VL-2B quality and runtime",
                "status": "accepted",
                "path": str(RSM2 / "FINAL_RUNTIME_SUMMARY.json"),
            },
            {
                "scope": "Qwen3-VL-4B Full-path quality",
                "status": "accepted",
                "path": str(EXP / "reports/VLM_FINAL_QUALITY_TABLES.json"),
            },
            {
                "scope": "Qwen3-VL-4B token-level ReSkip quality",
                "status": "accepted",
                "path": str(
                    EXP
                    / "eval/reskip/qwen3vl_4b_seed2_capped/FORMAL_QUALITY_GATE.json"
                ),
            },
            {
                "scope": "Qwen3-VL-4B fixed-64 runtime",
                "status": "accepted",
                "path": str(fixed4_path),
            },
            {
                "scope": "Qwen3-VL-4B natural-generation runtime",
                "status": "accepted",
                "path": str(natural4_path),
            },
            {
                "scope": "SmolVLM2 cross-architecture Full-path confirmation",
                "status": "accepted",
                "path": str(
                    EXP / "reports/diagnostics/GENERIC_VLM_REPAIR_AUDIT.json"
                ),
            },
        ],
        "diagnostic_only": [
            {
                "scope": "Qwen3-VL-4B seed-0 heldout gate interventions and AUC",
                "path": str(EXP / "reskip/qwen3vl_4b_seed0_heldout_seed259123"),
            },
            {
                "scope": "Qwen3-VL-4B seed-2 heldout block-risk ranking",
                "path": str(EXP / "reskip/qwen3vl_4b_seed2_heldout_seed259123"),
            },
        ],
        "excluded_from_claims": [
            {
                "scope": "Qwen3-VL-4B seed-0 aggressive ReSkip",
                "status": "failed formal quality gate; no runtime",
                "path": str(EXP / "eval/reskip/qwen3vl_4b_seed0"),
            },
            {
                "scope": "Qwen3-VL-4B seed-1 conservative ReSkip",
                "status": "stopped at development screen; no formal claim",
                "path": str(EXP / "reskip/qwen3vl_4b_seed1_conservative_screen"),
            },
            {
                "scope": "InternVL3.5 bounded adaptation repair",
                "status": "failed Amendment-008 continuation gate; diagnostic only",
                "path": str(EXP / "eval/full_rsm_lr5e4"),
            },
            {
                "scope": "Granite Vision 4.1-4B current-model transfer",
                "status": "failed Amendment-011 seed-0 promotion gate; no seed-1/2 search",
                "path": str(EXP / "eval/full_rsm_granite4_lr1e3_noent"),
            },
            {
                "scope": "Granite Vision 3.3-2B transfer preflight",
                "status": "stopped before a benchmark result or checkpoint because native dynamic-resolution sequences violated the frozen length setup",
                "path": str(EXP / "train/_failed_preflight"),
            },
        ],
        "generated_paper_artifacts": {
            "report": str(EXP / "reports/VLM_PAPER_READY_RESULTS.md"),
            "claims": str(EXP / "reports/VLM_FINAL_CLAIMS.json"),
            "figure_png": str(EXP / "figures/vlm_scale_method_evidence.png"),
            "figure_pdf": str(EXP / "figures/vlm_scale_method_evidence.pdf"),
        },
    }

    lines = [
        "# VLM experiment package: RSM-AttnRes and native token-level ReSkip",
        "",
        "## Main result",
        "",
        "RSM is trained as part of ordinary Full-path AttnRes adaptation. It uses no skip branch, skip label, compute target, external controller, or second training stage. ReSkip is evaluated only as a frozen token-level inference consequence after the Full-path quality gate.",
        "",
        "## Official six-task quality",
        "",
        "All scores are unweighted macro averages over AI2D, MMBench, MMMU, MMStar, OCRBench, and RealWorldQA.",
        "",
        "| Model | System | Seeds | Macro score |",
        "|---|---|---:|---:|",
    ]
    for record in quality_table:
        lines.append(
            "| {family} | {system} | {n} | {score} |".format(
                family=record["family"],
                system=record["system"],
                n=record["n"],
                score=display(
                    record["macro_average_mean"],
                    record["macro_average_sample_std"],
                    record["n"],
                ),
            )
        )

    lines.extend(
        [
            "",
            "### Per-task scores used for the paper table",
            "",
            "Multi-seed systems are arithmetic means over matched seeds; ReSkip rows are the frozen confirmatory seed shown above.",
            "",
            "| Model | System | n | AI2D | MMBench | MMMU | MMStar | OCRBench | RealWorldQA | Macro |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in per_task_quality_table:
        lines.append(
            f"| {record['family']} | {record['system']} | {record['n']} | "
            f"{record['AI2D']:.3f} | {record['MMBench']:.3f} | "
            f"{record['MMMU']:.3f} | {record['MMStar']:.3f} | "
            f"{record['OCRBench']:.3f} | {record['RealWorldQA']:.3f} | "
            f"{record['macro_average']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Matched-seed effect of RSM",
            "",
            "| Model | Matched seeds | RSM − Full AttnRes (pp) |",
            "|---|---:|---:|",
        ]
    )
    for record in quality["matched_seed_summaries"]:
        lines.append(
            f"| {record['family']} | {record['n_matched_seeds']} | "
            f"{display(record['rsm_minus_attnres_pp_mean'], record['rsm_minus_attnres_pp_sample_std'], record['n_matched_seeds'])} |"
        )

    lines.extend(
        [
            "",
            "## Token-level ReSkip quality and speed",
            "",
            "| Scale | Base score | Full RSM | RSM-ReSkip | Δ vs Full (pp) | Blocks/token | Fixed-64 Full/Base | Fixed-64 ReSkip/Base | Natural Full/Base | Natural ReSkip/Base |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in runtime_rows:
        lines.append(
            f"| {record['scale']} | {record['base_score']:.3f} | {record['full_score']:.3f} | "
            f"{record['reskip_score']:.3f} | {record['reskip_score'] - record['full_score']:+.3f} | "
            f"{record['block_equivalents']:.3f} | {record['fixed64_full_vs_base']:.3f}× | "
            f"{record['fixed64_reskip_vs_base']:.3f}× | {record['natural_full_vs_base']:.3f}× | "
            f"{record['natural_reskip_vs_base']:.3f}× |"
        )

    lines.extend(
        [
            "",
            "Runtime uncertainty uses paired per-request ratios and a 10,000-resample bootstrap:",
            "",
            "| Scale | Fixed-64 ReSkip/Base (95% CI) | Matched n | Natural ReSkip/Base (95% CI) | Matched n |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for record in runtime_rows:
        fixed_ci = record["fixed64_reskip_vs_base_95ci"]
        natural_ci = record["natural_reskip_vs_base_95ci"]
        lines.append(
            f"| {record['scale']} | {record['fixed64_reskip_vs_base']:.3f}× "
            f"[{fixed_ci[0]:.3f}, {fixed_ci[1]:.3f}] | {record['fixed64_matched_samples']} | "
            f"{record['natural_reskip_vs_base']:.3f}× "
            f"[{natural_ci[0]:.3f}, {natural_ci[1]:.3f}] | {record['natural_matched_samples']} |"
        )

    lines.extend(
        [
            "",
            "## 4B policy-development audit",
            "",
            "| Unit | Evidence level | Δ vs matched Full (pp) | Blocks/token | Decision |",
            "|---|---|---:|---:|---|",
            f"| seed0 aggressive | formal | {amendment4['observed_seed0_result']['delta_pp']:+.3f} | {amendment4['observed_seed0_result']['block_equivalents_per_token']:.3f} | failed quality gate; no runtime |",
            f"| seed1 conservative | development screen | {seed1_screen['delta_vs_full_pp']:+.3f} | {seed1_screen['block_equivalents_per_token']:.3f} | exceeded compute interval; no formal run |",
            f"| seed2 capped | formal confirmation | {seed2_gate['delta_vs_full_pp']:+.3f} | {seed2_gate['block_equivalents_per_token']:.3f} | passed; runtime allowed |",
            "",
            "## Causal and sensitivity evidence",
            "",
            "| Scale | Learned gate KL | Identity gate KL | Cyclic gate KL | Identity − learned |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for record in gate_ablation:
        lines.append(
            f"| {record['scale']} | {record['learned_gate_kl']:.5f} | "
            f"{record['identity_gate_kl']:.5f} | {record['cyclic_gate_kl']:.5f} | "
            f"{record['identity_minus_learned_kl']:+.5f} |"
        )

    mean_auc2 = statistics.fmean(
        row["auc_delta"] for row in auc_diagnostics if row["scale"] == "2B"
    )
    mean_auc4 = statistics.fmean(
        row["auc_delta"] for row in auc_diagnostics if row["scale"] == "4B"
    )
    lines.extend(
        [
            "",
            f"The RSM feature adds {mean_auc2:+.4f} mean cross-fitted AUC on 2B but {mean_auc4:+.4f} on 4B. The 4B negative diagnostic is retained: the Full-path benefit is real, but the learned factor is not a universally stronger skip-risk discriminator.",
            "",
            "## Claim boundary",
            "",
        ]
    )
    exact_pairs = equivalence4["summary"]["generated_token_ids_exact"]
    expected_pairs = equivalence4["summary"]["expected_pairs"]
    lines.insert(
        len(lines) - 3,
        f"The 4B device runtime matches the frozen host selector on {exact_pairs}/{expected_pairs} paired generations, including exact token IDs, decoded responses, block-equivalents, and shared skip-action statistics.",
    )
    lines.extend(f"- {item}" for item in claims["claim_boundary"])
    lines.extend(
        [
            "",
            "## Material Passport",
            "",
            f"- Models: {', '.join(item['name'] for item in protocol['material_passport']['models'])}.",
            f"- Training data: {protocol['material_passport']['training_data']['name']} at `{protocol['material_passport']['training_data']['local_root']}`.",
            f"- Evaluation: {protocol['material_passport']['evaluation_data']['name']}.",
            f"- Software snapshot: `{protocol['material_passport']['software']['code_snapshot']}`.",
            "- Hardware: node42, NVIDIA H100 80GB HBM3; batch size 1 for runtime measurements.",
            "",
            "## Canonical artifacts",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in claims["canonical_sources"].items())

    reports = EXP / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "VLM_FINAL_CLAIMS.json").write_text(
        json.dumps(claims, indent=2) + "\n"
    )
    (reports / "RESULTS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (reports / "VLM_PAPER_READY_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
