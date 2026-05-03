# M=4 ReSkip ablation — narrow vs wide P, 2B+4B (2026-05-01)

## Why we ran this

Reviewer/internal question: paper uses $M_{\max}{=}2$ everywhere. Is M an unswept knob? What if we relax it? This doc documents the structural answer.

## Setup

- 2B retrofit: `outputs/block_v3/2B_L4_v3_10k/retrofit_attnres_state.pt`, 7 blocks ($L{=}4$).
- 4B retrofit: `outputs/block_v3/4B_L4_v3_10k/retrofit_attnres_state.pt`, 9 blocks ($L{=}4$).
- LAMBADA-500 acc/ppl + bench_cache_regime.py (prefill 1024/2048, decode 64 tok, warmup 2 / timed 10).
- Calibrate-32 on LAMBADA prefix split disjoint from eval.
- Sweep $q \in \{0.5, 0.7, 0.85, 0.99\}$, $M_{\max}{=}4$.
- Two P regimes per scale:
  - **narrow** = paper canonical ($\{1,4\}$ on 2B, $\{1,2\}$ on 4B). Picked from per-block VLA action-drift in App. E.
  - **wide** = $\{0,1,…,N{-}1\}$ (all blocks eligible).
- Logs: `retrofit/outputs/m4_ablation/{2B,4B}_{narrow,wide}_M4.log`.

## TL;DR

1. **Narrow + M=4 ≡ narrow + M=2.** $|\mathcal{P}|=2$ caps avg actual skips at $\le 2.0$, and at the most aggressive setting (q=0.5) the observed avg is only **1.06/2** on 2B and **0.64/2** on 4B. So $M_{\max}{=}4$ never binds for paper's narrow P. M=4 numbers reproduce the paper's M=2 cells to two decimals.
2. **Wide + M=4 destroys accuracy.** At any q, wide-P + M=4 collapses LAMBADA: 2B drops to 0.45 (-12 pp); 4B drops to 0.54 (-12 pp). Block 0 / last block / action-head input are fragile.
3. **No new operating point opens up.** Eager+skip narrow lands at ~1.05× base decode/tok (slightly slower); eager+skip wide beats base on speed but accuracy is unusable. The compile-no-skip 1.029× iso-cost (Tab. retrofit_latency_full row 8) remains the only accuracy-safe ≤ base point.

## Narrow M=4 — LAMBADA + speed

### 2B narrow $\mathcal{P}{=}\{1,4\}$, M=4

| q | LAMBADA acc | ppl | avg skips / 2 | prefill 1024 vs base | decode/tok 1024 | prefill 2048 | decode/tok 2048 |
|---|---|---|---|---|---|---|---|
| 0.5  | 0.4120 | 12.55 | 1.06 | 1.028× | 1.074× | 0.963× | 1.176× |
| 0.7  | 0.4180 | 11.84 | 0.96 | 1.077× | 1.142× | 0.957× | 1.038× |
| 0.85 | 0.5600 | 5.26  | 0.19 | 1.031× | 1.071× | 0.956× | 1.059× |
| 0.99 | 0.5600 | 5.08  | 0.15 | 1.031× | 1.070× | 0.959× | 1.054× |

Paper-canonical M=2 q=0.99 cell (no-skip baseline 0.5650): the M=4 q=0.99 cell here is `0.5600`, identical (within noise) — confirming M=4≡M=2.

Decode/tok stays in the 1.04–1.18× band. **Eager+skip narrow does not beat base on decode** even at the most aggressive threshold.

### 4B narrow $\mathcal{P}{=}\{1,2\}$, M=4

| q | LAMBADA acc | ppl | avg skips / 2 | prefill 1024 vs base | decode/tok 1024 | prefill 2048 | decode/tok 2048 |
|---|---|---|---|---|---|---|---|
| 0.5  | 0.4920 | 7.13 | 0.64 | (router floor) | (router floor) | 2.187× | 1.099× |
| 0.7  | 0.5180 | 6.27 | 0.47 | — | — | 2.180× | 1.041× |
| 0.85 | 0.6160 | 3.57 | 0.04 | — | — | 2.193× | 1.149× |
| 0.99 | 0.6160 | 3.57 | 0.04 | — | — | 2.187× | 1.146× |

4B has 9 blocks vs 2B's 7, so the per-block router stack cost dominates eager. Even at q=0.99 the prefill is 2.19× base. Decode/tok 1.04-1.15× base.

**Note**: q=0.85 and q=0.99 give identical LAMBADA on 4B because both calibrate to a τ that fires only ~4% of inputs.

## Wide M=4 — LAMBADA + speed (already in `q1_compiled_rerun/eager_widePM_*`)

### 2B wide $\mathcal{P}{=}\{0,…,6\}$, M=4

| q | LAMBADA acc | ppl | avg skips / 7 | prefill 2048 vs base | decode/tok 2048 |
|---|---|---|---|---|---|
| 0.5  | 0.1200 | 1219.9 | 2.70 | 0.780× | 0.994× |
| 0.7  | 0.1880 | 294.6  | 2.22 | 0.780× | 0.955× |
| 0.85 | 0.3520 | 39.8   | 0.98 | 0.873× | 0.990× |
| 0.99 | 0.4540 | 14.7   | 0.53 | 0.777× | 0.974× |

### 4B wide $\mathcal{P}{=}\{0,…,8\}$, M=4

| q | LAMBADA acc | ppl | avg skips / 9 |
|---|---|---|---|
| 0.5  | 0.1000 | 4161.5 | 3.37 |
| 0.7  | 0.1940 | 1076.2 | 2.64 |
| 0.85 | 0.4660 | 17.2   | 0.99 |
| 0.99 | 0.5420 | 6.59   | 0.50 |

Wide M=4 at q=0.99 IS faster than base on both prefill and decode for 2B (0.78× / 0.97×). But LAMBADA collapses from 0.57 (no-skip) to 0.45 (-12 pp). Same on 4B: 0.66 → 0.54.

## Why wide-P collapses

Block 0 and the last block (block 6 on 2B / block 8 on 4B = penultimate / final) are structurally fragile under the AttnRes formulation: block 0 has no upstream representation to fall back on, and the final block carries the action-head / next-token head load. Avg-skip fingerprints in the wide M=4 logs confirm dyn-skip eagerly gates these high-w-recent blocks at low q.

## Implications for the paper

1. **No M sweep in the appendix retrofit_pareto table is fine** — narrow P caps observed skips below M=2 anyway, so the existing q-only sweep at M=2 captures the entire reachable Pareto.
2. **Don't introduce a wide-P row** — it would not contribute a usable operating point.
3. **The 1.029× compile-no-skip claim is still the best ≤-base point**. Eager+skip narrow ≈ 1.05× base decode is iso-cost-ish, not strictly faster.
4. **What needs to happen for the main table to claim "speedup at zero benchmark drop"**: pin the operating point (narrow + skip + q=0.99), then run the full benchmark suite at that point — currently the paper shows benchmarks NO-SKIP and speed WITH-SKIP, an internal inconsistency. See `paper_ablations_validation.md` §M, §G for what's already covered; HellaSwag / VLM tasks at the SKIP point are not yet measured.
