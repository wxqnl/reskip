# Statistical validation report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-01
- Verification Status: ANALYZED
- Version Label: validation_v1

## Validation Report

- **Source**: Amendments 013–015 sensitivity package
- **Overall Confidence**: CAUTION

The causal block intervention and paired runtime estimates are informative, but the gate-scale sweep uses one training seed and the four Pareto levels are sensitivity cells rather than independent replications. The report therefore avoids null-hypothesis significance claims for quality.

### Statistical Findings

| Metric | Test | Value | Effect size / precision | Confidence |
|---|---|---:|---|---|
| Qwen3-VL-2B: gate deviation vs single-block identity KL | Spearman across blocks; 10k paired-request bootstrap | ρ=0.200, 95% CI [0.029, 0.429] | small; only 6 blocks | CAUTION |
| Qwen3-VL-4B: gate deviation vs single-block identity KL | Spearman across blocks; 10k paired-request bootstrap | ρ=0.667, 95% CI [0.405, 0.738] | large; only 8 blocks | CAUTION |
| Qwen3-VL-2B q10: fixed-64 speedup vs Base | paired request geometric mean; 10k bootstrap | 1.0259×, 95% CI [1.0047, 1.0505] | n=72; same-host process | SOLID |
| Qwen3-VL-2B q45: fixed-64 speedup vs Base | paired request geometric mean; 10k bootstrap | 1.0438×, 95% CI [1.0357, 1.0532] | n=72; same-host process | SOLID |
| Qwen3-VL-2B q65: fixed-64 speedup vs Base | paired request geometric mean; 10k bootstrap | 1.0551×, 95% CI [1.0462, 1.0659] | n=72; same-host process | SOLID |
| Qwen3-VL-4B q10: fixed-64 speedup vs Base | paired request geometric mean; 10k bootstrap | 0.9900×, 95% CI [0.9841, 0.9973] | n=72; same-host process | CAUTION |
| Qwen3-VL-4B q25: fixed-64 speedup vs Base | paired request geometric mean; 10k bootstrap | 1.0286×, 95% CI [1.0227, 1.0363] | n=72; same-host process | SOLID |
| Qwen3-VL-4B q65: fixed-64 speedup vs Base | paired request geometric mean; 10k bootstrap | 1.0657×, 95% CI [1.0617, 1.0694] | n=72; same-host process | SOLID |

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Single-seed sweep | Gate-scale quality sensitivity has one training seed; it establishes local robustness, not a population variance estimate. | Figure S1 |
| Task heterogeneity | Six benchmark tasks are not iid replicates. Per-task ranges are descriptive and are not confidence intervals. | Figure S3 |
| Multiple descriptive cells | No p-values are reported and no cell is selected by significance. The accepted operating point was fixed before this sweep. | Gate and Pareto sweeps |
| Audited runtime replacement | The original q10 concurrent-run estimate is preserved but not used in Figure S4; Amendment 015 fixed one quiescent rerun before its result was available. | Qwen3-VL-2B q10 runtime |
| Runtime environment | Each cell is paired on one H100, but cells may use different physical H100s on the same host. Bootstrap intervals quantify paired-request variation within a cell; they do not cover cross-GPU, cross-host, or independent-process variance, so cross-coverage ordering is descriptive. | Figure S4 |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
| Simpson's paradox | NOTE | Macro and all six per-task deltas are retained, so aggregate/task direction differences remain visible. | Keep the per-task table with the macro. |
| Ecological fallacy | NOTE | Claims concern model/task-level behavior; no individual-level inference is made. | Preserve the stated unit of analysis. |
| Berkson's paradox | NOTE | Models, checkpoints, tasks, and cells were frozen without conditioning on new sensitivity outcomes. | Retain the frozen protocol and failed-launch audit. |
| Collider bias | NOTE | No covariate adjustment or conditioned regression is used. | No action required. |
| Base-rate neglect | NOTE | No sensitivity/specificity or diagnostic predictive value is claimed. | No action required. |
| Regression to the mean | NOTE | Cells were not chosen from extreme new results, and the accepted point was not re-selected. | Do not reinterpret a better extreme as a new default. |
| Survivorship bias | NOTE | Every frozen cell is reported; API-failed launches are preserved separately and rerun at the same specification. | Keep `_failed_launch` outside canonical results. |
| Look-elsewhere effect | NOTE | Four gate scales and four coverage levels were fixed before results; no intermediate coverage was inserted. | Treat any secondary pattern as exploratory. |
| Garden of forking paths | NOTE | Amendment 013 fixes models, seeds, tasks, coverages, blocks, actions, metrics, and stopping rules. | Cite the amendment in the appendix. |
| Correlation ≠ causation | CAUTION | The gate-deviation/KL association is correlational across blocks. The identity intervention supports functional relevance but not a universal training-causal claim. | Use 'associated with' for the correlation and reserve causal language for the explicit ablation. |
| Reverse causality | NOTE | No temporal directional claim is derived from the across-block association. | No action required. |

### Reproducibility

- **Method**: deterministic artifact rebuild plus environment-sensitive paired runtime reruns
- **Verdict**: PARTIALLY_REPRODUCIBLE

All tables and figures rebuild deterministically from saved result JSON. Runtime confidence intervals are reproducible from saved raw rows, while exact wall-clock values remain hardware- and process-state-sensitive.
