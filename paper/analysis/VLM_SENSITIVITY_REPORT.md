# VLM sensitivity analysis

This package follows frozen Amendments 013–015. Amendment 014 harmonized the reused Qwen3-VL-2B q45 runtime anchor from its historical MMStar-only scope to the same six-task fixed-64 protocol before any new q45 runtime result was observed. Amendment 015 registered one q10 quiescent rerun after an environment audit identified overlap with other project evaluations; both q10 estimates are retained, and the rerun is used for the paper runtime row. Neither amendment changed models, quality cells, policies, thresholds, or acceptance criteria.

## Gate-strength sweep (Qwen3-VL-2B, seed 0)

| RSM scale | Six-task macro | Factor range | Mean within-block SD | Source |
|---:|---:|---:|---:|---|
| 0.00 | 67.357 | 1.000–1.000 | 0.000 | reused matched Full AttnRes seed 0 |
| 0.25 | 67.615 | 0.750–1.250 | 0.115 | new pre-registered cell |
| 0.50 | 67.970 | 0.500–1.500 | 0.159 | reused accepted RSM seed 0 |
| 0.75 | 67.892 | 0.250–1.688 | 0.105 | new pre-registered cell |

## Frozen token-level ReSkip Pareto sweep

| Model | Target coverage | Observed block-eq/token | Macro | Δ vs Full RSM (pp) | Worst task Δ (pp) | Registered macro gate |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-VL-2B | 0.10 | 0.688 | 67.939 | -0.031 | -0.700 | PASS |
| Qwen3-VL-2B | 0.25 | 0.809 | 67.823 | -0.148 | -1.600 | PASS |
| Qwen3-VL-2B | 0.45 | 0.959 | 67.713 | -0.257 | -1.400 | PASS |
| Qwen3-VL-2B | 0.65 | 1.019 | 67.661 | -0.309 | -2.000 | FAIL |
| Qwen3-VL-4B | 0.10 | 0.488 | 73.082 | -0.174 | -0.830 | PASS |
| Qwen3-VL-4B | 0.25 | 0.767 | 73.005 | -0.251 | -0.800 | PASS |
| Qwen3-VL-4B | 0.45 | 1.089 | 72.743 | -0.512 | -2.500 | FAIL |
| Qwen3-VL-4B | 0.65 | 1.298 | 72.644 | -0.612 | -3.100 | FAIL |

## Fixed-64 runtime sensitivity

| Model | Target coverage | Formal block-eq/token | Speedup vs Base | 95% bootstrap CI | Speedup vs Full RSM | n |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B | 0.10 | 0.688 | 1.0259× | [1.0047, 1.0505] | 1.1865× | 72 |
| Qwen3-VL-2B | 0.45 | 0.959 | 1.0438× | [1.0357, 1.0532] | 1.1752× | 72 |
| Qwen3-VL-2B | 0.65 | 1.019 | 1.0551× | [1.0462, 1.0659] | 1.2008× | 72 |
| Qwen3-VL-4B | 0.10 | 0.488 | 0.9900× | [0.9841, 0.9973] | 1.0503× | 72 |
| Qwen3-VL-4B | 0.25 | 0.767 | 1.0286× | [1.0227, 1.0363] | 1.0715× | 72 |
| Qwen3-VL-4B | 0.65 | 1.298 | 1.0657× | [1.0617, 1.0694] | 1.1177× | 72 |

### Post-audit q10 runtime validation (Amendment 015)

| Qwen3-VL-2B q10 run | Speedup vs Base | 95% bootstrap CI | Speedup vs Full RSM | Use |
|---|---:|---:|---:|---|
| Original concurrent run | 1.1340× | [1.1055, 1.1649] | 1.3493× | preserved audit only |
| Quiescent rerun | 1.0259× | [1.0047, 1.0505] | 1.1865× | paper runtime row |

## Interpretation boundary

- Gate-scale cells were fixed before the new evaluations and the accepted scale remains 0.5 regardless of outcome.
- Block/token measurements are inference-only analyses of accepted Full-path checkpoints; no skip branch, skip labels, or compute target entered training.
- The Pareto x-axis is observed removed block-equivalents per decode token, not requested calibration coverage.
- The registered -0.3 pp band applies to the six-task macro only; per-task deltas are shown descriptively and tasks are not treated as iid replicates.
- Fixed-64 runtime keeps real benchmark images and prompts but standardizes output length; it is a speed test, not an accuracy score.
- Runtime is paired within each policy cell on one H100, but coverage cells may run on different physical H100s. Cross-coverage wall-clock ordering is therefore descriptive: changes in the fired block/action mix and device-side conditional or KV-maintenance overhead can make speed non-monotone in removed block-equivalents, while the saved data do not identify a unique cause.
