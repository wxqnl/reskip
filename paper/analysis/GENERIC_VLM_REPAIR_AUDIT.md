# Generic VLM AttnRes diagnosis and bounded repair audit

## Outcome

- SmolVLM2-2.2B: three-seed no-entropy RSM=55.583 ± 0.236, matched Full=55.207 ± 0.340, paired gain=+0.376 ± 0.116 pp; decision: **accept paper replacement**.
- InternVL3.5-2B: bounded lr=5e-4 RSM=66.708, matched Full=67.042, delta=-0.334 pp and delta vs Base=-0.870 pp; decision: **stop recipe refinement**.
- Granite Vision 4.1-4B: pre-registered seed-0 RSM=64.560, matched Full=64.456, delta=+0.104 pp and delta vs Base=-1.895 pp; decision: **stop current-model transfer**.

## Why the original supplemental runs were nearly flat

The original generic recipe used lr=2e-4, inherited from an older skip-KL screen. Under the present Full-path-only objective it barely moved the new mechanism: SmolVLM2 gate-weight RMS was 0.0457 and InternVL was 0.0298. Their route entropies were 1.0937/1.0965 and 1.2187/1.2187 relative to the uniform-routing ceilings. The source mixtures therefore remained almost uniform, leaving too little block/source differentiation for RSM to exploit.

Raising the learning rate made both routing and gates specialize, proving that the flat result was under-adaptation rather than a dead implementation. The high-rate InternVL benchmark nevertheless fell below Base, so stronger specialization is not monotonically better: the usable region depends on model family. Removing the uniform-routing reward helps SmolVLM2 but hurts InternVL, which is direct evidence against a universal one-recipe claim.

For the three independently trained SmolVLM2 RSM seeds, route-entropy EMA stays in 0.7970–0.8204 and gate-weight RMS in 0.1250–0.1323, compared with 1.0937 and 0.0457 in the original run. Thus the repaired mechanism signal is repeatable even before consulting benchmark scores.

Granite Vision 4.1-4B is the sole current-generation replacement test. Its recipe was frozen before the weight download completed. Under the exact tokenizer-regex setting, max sequence length 2048 accepts 15/16 preflight VLM samples at their model-native dynamic resolution and rejects one 2063-token example; no one-tile, truncation, or low-resolution shortcut is used.

## Formal macro results

| Family | Recipe | System | Seeds | Macro | Status |
|---|---|---|---:|---:|---|
| SmolVLM2-2.2B | Frozen pretrained | Base | 1 | 55.559 | frozen baseline |
| SmolVLM2-2.2B | lr=2e-4, entropy=0.02 | Full AttnRes | 1 | 55.493 | historical control |
| SmolVLM2-2.2B | lr=2e-4, entropy=0.02 | Full RSM-AttnRes | 1 | 55.696 | historical control |
| SmolVLM2-2.2B | lr=1e-3, entropy=0.02 | Full AttnRes | 1 | 55.724 | rejected seed-0 diagnostic |
| SmolVLM2-2.2B | lr=1e-3, entropy=0.02 | Full RSM-AttnRes | 1 | 55.573 | rejected seed-0 diagnostic |
| SmolVLM2-2.2B | lr=1e-3, entropy=0 | Full AttnRes | 3 | 55.207 ± 0.340 | three-seed confirmation |
| SmolVLM2-2.2B | lr=1e-3, entropy=0 | Full RSM-AttnRes | 3 | 55.583 ± 0.236 | three-seed confirmation |
| InternVL3.5-2B | Frozen pretrained | Base | 1 | 67.578 | frozen baseline |
| InternVL3.5-2B | lr=2e-4, entropy=0.02 | Full AttnRes | 1 | 67.609 | historical control |
| InternVL3.5-2B | lr=2e-4, entropy=0.02 | Full RSM-AttnRes | 1 | 67.434 | historical control |
| InternVL3.5-2B | lr=1e-3, entropy=0.02 | Full AttnRes | 1 | 66.796 | rejected seed-0 diagnostic |
| InternVL3.5-2B | lr=1e-3, entropy=0.02 | Full RSM-AttnRes | 1 | 67.284 | rejected seed-0 diagnostic |
| InternVL3.5-2B | lr=1e-3, entropy=0 | Full AttnRes | 1 | 67.131 | rejected seed-0 diagnostic |
| InternVL3.5-2B | lr=1e-3, entropy=0 | Full RSM-AttnRes | 1 | 66.951 | rejected seed-0 diagnostic |
| InternVL3.5-2B | lr=5e-4, entropy=0.02 | Full AttnRes | 1 | 67.042 | bounded midpoint seed-0 test |
| InternVL3.5-2B | lr=5e-4, entropy=0.02 | Full RSM-AttnRes | 1 | 66.708 | bounded midpoint seed-0 test |
| Granite Vision 4.1-4B | Frozen pretrained | Base | 1 | 66.456 | frozen current-model baseline |
| Granite Vision 4.1-4B | lr=1e-3, entropy=0 | Full AttnRes | 1 | 64.456 | pre-registered current-model seed-0 transfer |
| Granite Vision 4.1-4B | lr=1e-3, entropy=0 | Full RSM-AttnRes | 1 | 64.560 | pre-registered current-model seed-0 transfer |

## SmolVLM2 paired-seed confirmation

| Seed | Full AttnRes | Full RSM-AttnRes | RSM − Full (pp) |
|---:|---:|---:|---:|
| 0 | 55.505 | 55.819 | +0.315 |
| 1 | 55.279 | 55.582 | +0.304 |
| 2 | 54.837 | 55.347 | +0.511 |

Acceptance required a mean paired gain of at least +0.15 pp, RSM mean above Base, and positive paired gains on at least two of three seeds. All cells are reported regardless of the decision.

## Mechanism evidence

![Generic VLM repair mechanism](../../figures/generic_vlm_repair_mechanism.png)

The figure is descriptive rather than causal proof. Its strongest causal controls are the matched Full/RSM cells and the pre-registered entropy ablation; route entropy and gate norm diagnose whether adaptation actually created specialization.

## Claim boundary

- RSM remains part of ordinary Full-path AttnRes adaptation. There is no skip branch, skip supervision, compute target, warm start, or second stage in any reported repair run.
- The RSM addition is 10,245 parameters on SmolVLM2 and 12,294 on InternVL (about 0.00046% and 0.00052% of the respective base models). These parameters are optimized jointly with AttnRes, not by a separate skip-training phase.
- The supplemental models remain architecture-transfer tests. Their generic hook does not establish a device-side token-level ReSkip speed claim.
- Rejected high-rate and no-entropy InternVL cells are diagnostics, not candidate results. The single lr=5e-4 point was frozen as a bounded midpoint; the protocol forbids further recipe search if it fails.
- Granite Vision 3.3 stopped at input-length preflight before any complete benchmark cell or checkpoint. It is stored only as a failed preflight, not reported as a model result.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-01
- Verification Status: VERIFIED after complete six-task artifacts
- Models: `/data/Minko/models/SmolVLM2-2.2B-Instruct`; `/data/Minko/models/InternVL3_5-2B-HF`; `/data/Minko/models/granite-vision-4.1-4b`
- Current-model source: `https://huggingface.co/ibm-granite/granite-vision-4.1-4b`
- Training data: `/data/Minko/datasets`, v3 mixed adaptation stream
- Evaluation: local lmms-eval caches for AI2D, MMBench, MMMU, MMStar, OCRBench, and RealWorldQA
- Hardware: node42 NVIDIA H100 GPUs 0-6; GPU 7 excluded
- Protocols: `/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831/protocol/PROTOCOL_AMENDMENT_006.json`, `/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831/protocol/PROTOCOL_AMENDMENT_007.json`, `/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831/protocol/PROTOCOL_AMENDMENT_008.json`, `/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831/protocol/PROTOCOL_AMENDMENT_009.json`, `/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831/protocol/PROTOCOL_AMENDMENT_010.json`, `/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831/protocol/PROTOCOL_AMENDMENT_011.json`, `/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831/protocol/PROTOCOL_AMENDMENT_012.json`
- Software: `/data/Minko/experiments/attnres_rsm_vlm_scale_generalization_20260831/code`
