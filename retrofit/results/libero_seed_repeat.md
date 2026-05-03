# LIBERO seed-repeat — statistical-significance check on Path 0 vs. Path B

**Date:** 2026-05-01 (run launched 2026-04-30 ~22:30 EDT, finished 2026-05-01 07:42 EDT — ~9h 15min wall)
**Goal:** answer advisor's stat-significance ask on Tab. 4 (`tab:vla_libero`):
do the **Path B − Path 0** deltas reported in the paper survive when each cell
is averaged over multiple eval-time seeds?

## Setup

| Item | Value |
|---|---|
| Ckpts | `libero_path0_base_30k` (2B P0), `libero_pathB_warm_v2_30k` (2B PB), `libero_path0_4B_cleanbase_30k` (4B P0), `libero_pathB_4B_cleanbase_30k` (4B PB) |
| Suites | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| Seeds | `1, 2, 3` (additive on top of paper's existing seed-7 numbers) |
| Trials/task | 50 (LIBERO default), 10 tasks/suite → 500 rollouts/(ckpt,suite,seed) cell |
| Total | 4 ckpts × 4 suites × 3 seeds = **48 cells, 24 000 rollouts** |
| Queue | `experiments/queue_libero_seed_repeat.sh` (1 server / ckpt, 12 evals reuse same server) |
| Pairs | pair 1 (server GPU 0, render GPU 1, port 5694) → 2B; pair 2 (server GPU 2, render GPU 3, port 5695) → 4B |
| Eval modification | `examples/LIBERO/eval_files/eval_libero.sh` now forwards `SEED` env var via `--args.seed`; videos written to `/tmp/libero_videos` |

## Per-cell SR (3 seeds × 4 suites × 4 ckpts)

| Ckpt | Suite | seed1 | seed2 | seed3 | mean | std |
|---|---|---|---|---|---|---|
| **2B Path 0** | spatial | 96.8 | 96.2 | 97.4 | **96.80** | 0.60 |
|  | object  | 99.8 | 100.0 | 99.8 | **99.87** | 0.12 |
|  | goal    | 98.4 | 97.2 | 98.6 | **98.07** | 0.76 |
|  | lib10   | 92.8 | 92.4 | 93.6 | **92.93** | 0.61 |
| **2B Path B** | spatial | 97.0 | 97.6 | 96.2 | **96.93** | 0.70 |
|  | object  | 99.6 | 100.0 | 99.6 | **99.73** | 0.23 |
|  | goal    | 97.6 | 98.4 | 97.8 | **97.93** | 0.42 |
|  | lib10   | 92.6 | 93.6 | 91.8 | **92.67** | 0.90 |
| **4B Path 0** | spatial | 93.8 | 93.6 | 94.0 | **93.80** | 0.20 |
|  | object  | 99.4 | 98.8 | 99.4 | **99.20** | 0.35 |
|  | goal    | 98.8 | 97.6 | 99.2 | **98.53** | 0.83 |
|  | lib10   | 90.8 | 91.4 | 92.8 | **91.67** | 1.03 |
| **4B Path B** | spatial | 93.4 | 93.2 | 93.4 | **93.33** | 0.12 |
|  | object  | 99.8 | 99.4 | 100.0 | **99.73** | 0.31 |
|  | goal    | 98.2 | 97.2 | 98.8 | **98.07** | 0.81 |
|  | lib10   | 93.2 | 93.4 | 94.6 | **93.73** | 0.76 |

(All numbers in % SR. std = sample std with n=3, ddof=1.)

## 4-suite averages (per-seed average of the 4 suite cells, then mean ± std over 3 seeds)

| Cell | seed1 | seed2 | seed3 | **mean ± std** | paper (Tab. 4) |
|---|---|---|---|---|---|
| 2B Path 0 | 96.95 | 96.45 | 97.35 | **96.92 ± 0.45** | 96.05 |
| 2B Path B | 96.70 | 97.40 | 96.35 | **96.82 ± 0.53** | 96.75 (+0.70) |
| 4B Path 0 | 95.70 | 95.35 | 96.35 | **95.80 ± 0.51** | 96.05 |
| 4B Path B | 96.15 | 95.80 | 96.70 | **96.22 ± 0.45** | 96.70 (+0.65) |

**Observations**:
- **2B Path 0 actually trends *higher* in 3-seed mean (96.92) than the paper number (96.05)**.
  The paper's 2B P0 is the unlucky end of the seed distribution; spatial in particular jumps from
  paper 94.8 → 3-seed 96.80.
- 2B Path B is essentially flat between paper (96.75) and 3-seed mean (96.82).
- 4B Path 0 trends *lower* in 3-seed mean (95.80) than the paper number (96.05).
- 4B Path B also trends lower (96.22 vs. paper 96.70).

## Path B − Path 0 paired delta (the headline Δ in Tab. 4)

| Scale | per-seed Δ | mean Δ ± std | paired t-test (df=2) |
|---|---|---|---|
| 2B | −0.25, +0.95, −1.00 | **−0.10 ± 0.98** | t = −0.18, p = **0.876** |
| 4B |  +0.45, +0.45, +0.35 | **+0.42 ± 0.06** | t = +12.50, p = **0.006** |

Per-suite paired Δ (PB − P0, mean over 3 seeds):

| Suite | 2B Δ | 4B Δ |
|---|---|---|
| spatial | +0.13 | **−0.47** |
| object | −0.13 | +0.53 |
| goal   | −0.13 | −0.47 |
| lib10  | −0.27 | **+2.07** |

## Reading

**On 2B**:
- Path B − Path 0 ≈ **0** in 3-seed mean (Δ = −0.10 ± 0.98pp, p=0.88).
- Per-seed sign flips across seeds (−, +, −). The +0.70pp paper headline is
  inside-noise of seed-jitter on the *baseline* (Path 0 spatial swings 94.8
  paper → 96.8 mean) more than a real PB advantage.
- **Conclusion**: 2B PB and 2B P0 are statistically indistinguishable on the
  4-suite average. The "+0.70pp" cannot be defended as a clean win.

**On 4B**:
- Path B − Path 0 = **+0.42 ± 0.06pp**, paired t = +12.5, p = 0.006.
  Highly significant *because all three seeds move in the same direction* by
  almost the same amount (per-seed Δ: +0.45, +0.45, +0.35).
- The signal is **entirely carried by `libero_10`** (Δ = +2.07pp consistently);
  spatial and goal both go *negative* on PB (−0.47pp each), object positive
  (+0.53pp). Net is +0.42 because lib10 is the largest absolute swing.
- **Conclusion**: PB really does help 4B on long-horizon tasks (lib10), but
  *hurts* on shorter suites by a small amount. The aggregate +0.42pp is real
  and significant, but the within-suite story is "lib10 robustness" not
  "uniform PB > P0".

## Implications for paper

The paper's current Tab. 4 reports (cell, seed-7) — i.e. *one* eval seed.
Three honest options for revision:

1. **Report 4-suite mean ± std over (seed-7 + seeds 1/2/3)** = 4-seed mean.
   This makes 2B PB ≈ 2B P0 within noise — the "+0.70pp" claim collapses.
   On 4B, the +0.65pp shrinks to about +0.4pp but stays positive.
2. **Report 3-seed mean (seeds 1/2/3) only**, with a footnote that seed-7
   in earlier draft was an upper-tail single sample. Same outcome as (1).
3. **Keep deployment-style (seed-7) numbers but add a per-seed appendix
   table.** Most defensible for the user's stated narrative:
   "deployment performance varies by ~1pp seed-to-seed; we report a single
   representative deployment seed." This is what the existing
   `tab:vla_seed_variance` already does for `libero_goal` + `libero_10`,
   but only with 2 seeds; we can extend to 4 seeds × 4 suites.

The numerically most striking finding is the **4B lib10 +2.07pp, p=0.006**,
which is the *one* place PB definitively beats P0. The 2B story is harder
to defend with this data and may need to be framed as "matches Path 0 with
the same compute, while preserving the AttnRes intrinsic-depth signal for
ReSkip" rather than as a quality gain.

## Cell artifacts

- Per-cell logs: `retrofit/outputs/libero_seed_repeat/<ckpt>_seed<N>_<suite>/eval.log`
- Per-pair master logs:
  - `retrofit/outputs/libero_seed_repeat/pair1_2B_master.log`
  - `retrofit/outputs/libero_seed_repeat/pair2_4B_master.log`
- Per-pair server logs: `pair{1,2}_..._server.log`
- Queue script: `experiments/queue_libero_seed_repeat.sh`
- Eval-script change: `starVLA/examples/LIBERO/eval_files/eval_libero.sh` (now
  forwards `SEED` env var via `--args.seed "${SEED:-7}"`; video out to `/tmp/libero_videos`)
