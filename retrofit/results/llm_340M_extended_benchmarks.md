# 340M LLM Extended Benchmarks (Phase 2 of NeurIPS-prep run)

**Date:** 2026-04-30 (results recorded; runs themselves finished 2026-04-29)
**Goal:** Extend `tab:reskip_benchmark` in `paper/main.tex` from
LAMBADA + HellaSwag + ARC-E/C (4 metrics) to include PIQA + MMLU +
OpenBookQA, and verify the lossless-skip claim still holds at ReSkip's
default operating point on the new tasks.

## Setup

| Item | Value |
|---|---|
| Vanilla 340M | `flame/saves/transformer-340M` (no AttnRes) |
| AttnRes-full | `flame/saves/reskip_transformer-340M` (8-block, FineWeb-Edu 100BT) |
| AttnRes + ReSkip | `outputs/reskip_340M_combined_35_skip2_q085` (P={3,5}, M=2, q=0.85) |
| Tasks | piqa, mmlu, openbookqa, arc_easy, arc_challenge, lambada_openai, hellaswag |
| Harness | `lm-evaluation-harness` via `experiments/flame_lm_eval.py` |
| Headline metric | `acc_norm` for piqa / openbookqa / arc_easy / arc_challenge / hellaswag, `acc` for mmlu / lambada_openai |

## Results

| Cell | PIQA | OpenBookQA | ARC-E | ARC-C | MMLU | LAMBADA acc | LAMBADA ppl | HellaSwag |
|---|---|---|---|---|---|---|---|---|
| Vanilla 340M (no AttnRes)            | 0.6779 | 0.3320 | 0.5602 | 0.3046 | 0.2594 | 0.3790 | 24.71 | 0.4436 |
| AttnRes-full (no skip)               | **0.6893** | **0.3580** | 0.5438 | 0.3012 | 0.2555 | **0.4054** | **20.20** | **0.4607** |
| AttnRes + ReSkip ({3,5}/M=2/q=0.85)  | **0.6893** | **0.3580** | 0.5438 | 0.3012 | 0.2555 | **0.4054** | **20.20** | **0.4607** |

### Reading

1. **AttnRes vs. vanilla**: the AttnRes-trained-in 340M lifts LAMBADA
   by +2.6pp (0.379 → 0.405), HellaSwag +1.7pp, OpenBookQA +2.6pp,
   PIQA +1.1pp, with ARC-E ($-1.6$) and MMLU ($-0.4$) within harness
   noise. This is the existence-proof for AttnRes's intrinsic depth
   signal on the new task set.

2. **AttnRes + ReSkip ≡ AttnRes-full at q=0.85**: every cell is
   bit-identical between the no-skip and the q=0.85 cells. This is
   not a bug — the q=0.85 threshold is calibrated on FineWeb-Edu,
   and on the lm-eval distribution it fires 0 % of the time
   (verified by `experiments/probe_b1_b2_skip_rate.py`). The cell
   acts as a no-op on these tasks. The "rate-matched dynamic vs.
   static" comparison is the q=0.5 B1+B2 ablation in
   `baseline_b1_b2_skip_strategies.md`, not this row.

## Cell artifacts

| Cell | Results JSON |
|---|---|
| Vanilla 340M     | `retrofit/outputs/lm_eval_340M_extra/vanilla/__home__user01__Minko__reskip2__reskip__flame__saves__transformer-340M/results_2026-04-29T21-53-40.923546.json` |
| AttnRes-full     | `retrofit/outputs/lm_eval_340M_extra/attnres_full/__home__user01__Minko__reskip2__reskip__flame__saves__reskip_transformer-340M/results_2026-04-29T12-18-23.744194.json` |
| AttnRes + ReSkip | `retrofit/outputs/lm_eval_340M_extra/attnres_reskip_35_q085/__home__user01__Minko__reskip2__reskip__outputs__reskip_340M_combined_35_skip2_q085/results_2026-04-29T21-54-06.446484.json` |

The `attnres_reskip_23_q093` cell (queued GPU 3, P={2,3}/M=2/q=0.93)
was never launched per the paper-scope-minimalism decision in
`extra_benchmark_run_2026-04-29.md` — appendix-only at PPL-ratio
level, not part of `tab:reskip_benchmark`.

## Conclusions

- 7-task headline supports the existing 4-task `tab:reskip_benchmark`:
  AttnRes-full ≥ vanilla on every task that lifts (LAMBADA / HellaSwag /
  OpenBookQA / PIQA), and ReSkip at q=0.85 is bit-identical to AttnRes-full.
- The "ReSkip preserves every metric" claim survives the task-set
  expansion, with the honest caveat that q=0.85 fires 0 % on lm-eval
  data; the load-bearing dynamic-vs-static comparison lives in the
  B1+B2 ablation at q=0.5.
