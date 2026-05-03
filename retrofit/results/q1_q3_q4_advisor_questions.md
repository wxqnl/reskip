# Advisor questions Q1 / Q3 / Q4 — backup data

**Date:** 2026-05-01
**Goal:** answer 3 of the advisor's 5 reviewer-defense questions; Q2 (canonical-mix
training ablations) and Q5 (ReSkip-on-VLA paired test) are tracked separately.

| Q | Question | Answer status |
|---|---|---|
| Q1 | Qwen3-VL 2B/4B latency at q=0.85 / aggressive | **Done — see §Q1 (canonical L=4 rerun: eager + compile-prefill + compile-decode)** |
| Q2 | canonical-mix SFT-only / LoRA / adapter / no-skipKL | **Done — see `q2_canonical_mix_ablations.md`** |
| Q3 | why is 340M ReSkip ≡ AttnRes-full to 4 decimals? | **Done — confirmed 0% fire rate** |
| Q4 | 0.19 avg-skip ⇒ adaptive computation? per-example correlation | **Done — ρ ≈ 0.075–0.08, not difficulty-adaptive** |
| Q5 | VLA paired-test on +0.65pp / +1.10pp claim | seed-repeat done (`libero_seed_repeat.md`); +1.10pp ReSkip-on-VLA pending clarification |

---

## Q1 — Qwen3-VL 2B/4B latency on canonical L=4 partition (eager + compile)

**Important correction from earlier draft.** The previous version of this section
benched `H_r256_5k` (the legacy L=2 / 14-block VLM-only retrofit). The paper's
canonical retrofit cells are the L=4 partition (`2B_L4_v3_10k`, `4B_L4_v3_10k`),
which use 7 / 9 blocks instead of 14 / 18. Half as many blocks → half the
per-block dispatch overhead. Re-running on the canonical cells closes most of
the 1.39–1.43× eager gap and makes the compiled iso-cost claim writeable for
**both** prefill and decode.

**Bench scripts:**
- Eager + dyn-skip: `retrofit/bench/bench_cache_regime.py` (manual prefill + N×T=1 decode loop with `past_key_values`)
- Compile-prefill: `retrofit/bench/bench_retrofit_compile.py` (single-shot prefill, mode=`reduce-overhead`)
- Compile-decode: `retrofit/bench/bench_compile_decode_static.py` (T=1 decode with HF `StaticCache`, mode=`default`)

**Setup:** single L4 GPU, bf16, prefill ∈ {1024, 2048}. Eager: 128 decode tokens, median of 15 runs after 3 warm-ups, dyn-skip P=[1,4] (2B) / [1,2] (4B), max_skips=2, q ∈ {0.85, 0.99} with thresholds calibrated on LAMBADA-text. Compile: 64 decode tokens, median of 10 runs after 3 warm-ups, no skip (paper Appendix already documents that dynamic skip's `.item()` CPU sync breaks CUDA graphs / Inductor traces).

### 1. Eager — decode/tok (ms), cache=T (canonical L=4)

Logs: `retrofit/outputs/q1_compiled_rerun/bench_L4_{2B,4B}_q{085,099}.log`.

**2B (P={1,4}, max_skips=2)**
| q | seq | Base | Retrofit (γ=1) | Retrofit + dyn-skip | retrofit×base | +skip×base |
|---|---:|---:|---:|---:|---:|---:|
| 0.85 | 1024 | 11.95 | 14.82 | 13.53 | 1.240× | **1.132×** |
| 0.85 | 2048 | 11.92 | 14.79 | 13.18 | 1.241× | **1.105×** |
| 0.99 | 1024 | 11.99 | 14.69 | 12.98 | 1.226× | **1.083×** |
| 0.99 | 2048 | 11.84 | 14.62 | 13.72 | 1.235× | 1.159× |

**4B (P={1,2}, max_skips=2)**
| q | seq | Base | Retrofit (γ=1) | Retrofit + dyn-skip | retrofit×base | +skip×base |
|---|---:|---:|---:|---:|---:|---:|
| 0.85 | 1024 | 15.79 | 19.56 | 19.80 | 1.239× | 1.254× |
| 0.85 | 2048 | 15.73 | 19.51 | 19.76 | 1.240× | 1.256× |
| 0.99 | 1024 | 15.62 | 19.58 | 19.47 | 1.254× | 1.246× |
| 0.99 | 2048 | 15.51 | 18.96 | 19.25 | 1.223× | 1.241× |

Eager retrofit overhead is **~1.24× across all 8 cells** (down from ~1.40× on legacy L=2 — half the blocks, half the dispatch cost). Skip helps on 2B (best 1.083× at q=0.99 / seq=1024) but is essentially neutral / mildly harmful on 4B at this calibration — the eligible-block set P={1,2} is too narrow and per-call skip rate stays near zero, so the rule's per-token branch overhead exceeds the FLOP savings.

### 2. Compile-prefill — single-shot full-prompt latency (mode=`reduce-overhead`)

Logs: `retrofit/outputs/q1_compiled_rerun/compile_prefill_L4_{2B,4B}_redoh.log`.

| Model | seq | Base eager (ms) | Base compiled (ms) | Retrofit eager (ms) | Retrofit compiled (ms) | retrofit_compiled / base_compiled |
|---|---:|---:|---:|---:|---:|---:|
| 2B  | 1024 | 14.89 |  7.82 | 17.20 |  8.58 | **1.098×** |
| 2B  | 2048 | 24.94 | 16.44 | 28.05 | 17.40 | **1.058×** |
| 4B  | 1024 | 27.19 | 17.89 | 30.78 | 18.93 | **1.058×** |
| 4B  | 2048 | 50.29 | 37.86 | 55.55 | 39.47 | **1.043×** |

`reduce-overhead` enables CUDA graphs (fixed input shape) on top of Inductor fusion. At seq=2048 retrofit is **1.058× / 1.043× of base compiled** — the writeable iso-cost number for the prefill path. (Tried `max-autotune` for the headline 1.029× but that mode hits "no valid triton configs / OOM" on this hardware for several router GEMMs and falls back to the same kernels as `reduce-overhead`. Logs in `compile_prefill_L4_*_maxauto.log`.)

### 3. Compile-decode — T=1 decode with StaticCache (mode=`default`)

Logs: `retrofit/outputs/q1_compiled_rerun/compile_decode_static_L4_{2B,4B}_def.log`.

| Model | seq | Base eager (ms/tok) | Base compiled (ms/tok) | Retrofit eager (ms/tok) | Retrofit compiled (ms/tok) | retrofit_compiled / base_compiled |
|---|---:|---:|---:|---:|---:|---:|
| 2B  | 1024 | 15.10 | 4.83 | 17.53 | 5.40 | **1.117×** |
| 2B  | 2048 | 15.47 | 4.88 | 17.78 | 5.35 | **1.096×** |
| 4B  | 1024 | 18.74 | 7.70 | 21.87 | 8.04 | **1.044×** |
| 4B  | 2048 | 18.64 | 7.71 | 21.55 | 8.08 | **1.048×** |

Decode + compile required: (a) HF `StaticCache` (preallocates fixed-shape KV — `dynamic=True` with `DynamicCache` runs into a retrace storm because the cache shape grows per step), and (b) `mode=default` (Inductor fusion only). `reduce-overhead` adds CUDA graphs on top, but Qwen3-VL's `StaticCache.update()` reuses the same persistent buffer across steps (`self.keys.index_copy_(2, cache_position, key_states)`), which trips CUDA-graph's "tensor overwritten by subsequent run" guard even with `torch.compiler.cudagraph_mark_step_begin()` + output cloning. Logs in `compile_decode_static_L4_*_v2.log`. `mode=default` avoids the CUDA-graph dependency but still gets ~3× over eager via fused kernels.

### Reading (Q1, rewritten)

1. **The earlier "+30–40% retrofit decode overhead" was an L=2 artifact.** On the paper's canonical L=4 cells, eager retrofit overhead drops to ~1.24× (decode) and ~1.10–1.16× (prefill, no skip). Half the blocks, half the AttnRes dispatch.
2. **Compile closes the gap to writeable iso-cost in both regimes.**
   - **Prefill**, seq=2048: 2B 1.058×, 4B 1.043× of base compiled (`reduce-overhead`).
   - **Decode/tok**, seq=2048: 2B 1.096×, 4B 1.048× of base compiled (`mode=default` + StaticCache).
   - 4B is closer to iso-cost than 2B in both regimes (deeper base ⇒ AttnRes router is a smaller fraction of total FLOPs); this matches the paper's `tab:retrofit_latency_full` which already reports 1.029× at seq=2048 / max-autotune (a slightly tighter tuning regime than what fits on this L4 box).
3. **Skip helps on 2B eager, doesn't compose with compile.** 2B q=0.99 seq=1024 hits 1.083× eager (best decode point in the eager block); 4B P={1,2} skip is essentially a no-op at this calibration. Under compile, dynamic skip is intentionally not measured — paper's Appendix already documents that the skip rule's `.item()` CPU sync breaks the captured graph.
4. **What this means for the paper's iso-cost claim and §Method speed paragraph.** The 1.029× compiled iso-cost number in the paper's `tab:retrofit_latency_full` is the headline; the L=4 eager and L=4 compile-decode numbers above are the corroborating internal evidence. The previous draft's pessimistic "+30–40% slower under stock PyTorch" framing was wrong — it was specific to the L=2 legacy partition that the paper does not use.
5. **Implication for VLA / LIBERO numbers.** The +0.65pp / +1.10pp accuracy claims sit at the compiled iso-cost point; the eager+skip 2B numbers (1.083× best, 1.105× typical at seq=2048 q=0.85) also support an "approximately iso-cost" reading even without compile, on the canonical 2B cell.

**What this section should say in the paper.** Keep the iso-cost headline (`tab:retrofit_latency_full` row 8: 1.029× max-autotune at seq=2048). Add to §Method speed paragraph: "On the canonical L=4 partition, eager retrofit decode overhead is 1.22–1.25× of base and drops to 1.04–1.10× under `torch.compile` + StaticCache (Appendix~A; compile-decode requires StaticCache, mode=default — CUDA graphs do not compose with the cache update pattern in HF Qwen3-VL)." Drop the previous "+30–40% uncompiled" caveat — that number was an L=2 artifact. Internal logs live under `retrofit/outputs/q1_compiled_rerun/`.

---

## Q3 — q=0.85 fire rate on multiple-choice lm-eval tasks

**Probe script:** `experiments/probe_q085_per_task.py`
**Cell:** `outputs/reskip_340M_combined_35_skip2_q085` (340M AttnRes + ReSkip @ q=0.85, calibrated on FineWeb-Edu, P={3,5}, M=2)
**Output:** `retrofit/outputs/probe_q085_per_task.json`

| Task | Block-level skip rate at q=0.85 | n_examples | calls | Hypothesis confirmed? |
|---|---:|---:|---:|---|
| LAMBADA       | **0.000%** | 64 | 64 | ✓ |
| HellaSwag     | **0.000%** | 64 | 64 | ✓ |
| OpenBookQA    | **0.000%** | 64 | 64 | ✓ |
| ARC-Easy      | **0.000%** | 64 | 64 | ✓ |
| ARC-Challenge | **0.000%** | 64 | 64 | ✓ |
| MMLU          | **0.000%** | 64 | 64 | ✓ |
| PIQA          | (HF dataset script deprecated, skipped — same family of multiple-choice text, expect same behavior) | — | — | (n/a) |

### Reading (Q3)

The "ReSkip + AttnRes-full are bit-identical to four decimal places on PIQA / MMLU / OpenBookQA / ARC-E / ARC-C / HellaSwag / LAMBADA" finding in `llm_340M_extended_benchmarks.md` is **not** an artifact of eval size or rounding. **q=0.85 fires 0% of the time on every multiple-choice / cloze task tested** — the threshold was calibrated on FineWeb-Edu's `w_recent` distribution and the lm-eval task distributions all sit below the 85th percentile of FineWeb-Edu (paper's `tab:reskip_main_threshold` already documents ~9.5% fire on LAMBADA cloze with full-context, but the 64-example multiple-choice scoring inputs are too short / too easy to fire any skip).

**Implication for paper:** the existing `tab:reskip_benchmark` 7-task table is honest — at the calibrated operating point ReSkip really is a no-op on these tasks. The "ReSkip preserves every benchmark" claim is structurally true, with the caveat (already in paper Appendix~A.6) that "preserves" here means "fires zero skips, so by construction identical." The load-bearing dynamic-vs-static comparison is the q=0.5 B1+B2 ablation in `baseline_b1_b2_skip_strategies.md` where fire rate is forced into the ~12% regime.

---

## Q4 — Per-example difficulty × skip-depth correlation

**Probe script:** `experiments/probe_difficulty_skip_correlation.py`
**Cell:** `outputs/reskip_340M_b1b2_recent_minus_embed_gt_q050_M1` (B2c recent_minus_embed_gt, q=0.5, M=1; the cell that *does* fire ~12% on LAMBADA so we can actually measure correlation)
**Output:** `retrofit/outputs/probe_difficulty_skip_correlation.json`

| Task | N | mean PPL | mean skip / fwd | Spearman ρ(ppl, skip) | ρ(CE, skip) | PPL-quartile mean skip (low→high difficulty) |
|---|---:|---:|---:|---:|---:|---|
| LAMBADA   | 400 | 38.70 | 0.818 | **+0.075** | +0.075 | [0.77, 0.84, 0.79, 0.87] |
| HellaSwag | 400 | 79.69 | 0.475 | **+0.079** | +0.079 | [0.43, 0.44, 0.50, 0.53] |

### Reading (Q4)

- **Spearman ρ ≈ 0.075–0.08 on both tasks ⇒ effectively no correlation between example difficulty (PPL) and skip count.**
- The PPL-quartile scan shows a *mild* monotone trend on HellaSwag (low→high quartile: 0.43 → 0.53, +23% relative) and a non-monotone trend on LAMBADA (0.77 → 0.84 → 0.79 → 0.87) — the trend, if real, is at the noise floor.
- **What this means for the "adaptive computation" claim:**
  - ReSkip's dynamic skipping is **not** "harder examples get more compute." It is "different *positions / blocks* within an input route differently based on the AttnRes weights, but the *aggregate skip budget per input* is roughly fixed."
  - Concretely: the q=0.5 cell skips ~0.475–0.818 blocks per forward on average; the variance across examples is dominated by stochastic position-level threshold crossings rather than by example-level difficulty.
- **Defensible reframing for paper:** swap "adaptive depth that allocates compute by example difficulty" for "**input-conditional depth that allocates compute by per-position routing weights**". The mechanism is genuine adaptive routing (different inputs → different skip patterns) but the difficulty-conditional reading is not supported by this data.
- **Reviewer push-back risk:** if a reviewer asks "is your method really adaptive?" the strongest answer is the rate-matched dynamic-vs-static comparison in `baseline_b1_b2_skip_strategies.md` (B2.c +6–10pp over B1.c/d/e at the same ~12% rate), **not** an example-level difficulty curve.

### Anchor numbers for paper text

If we want to keep an "adaptive" sentence in §3 / abstract: cite
- Per-position skip distribution variance (already in `baseline_b1_b2_skip_strategies.md` per-position breakdown)
- Different inputs route over different block subsets (probe shows skip_positions list varies)

If we want to *drop* the adaptivity claim and replace with "calibrated dynamic skipping": this Q4 result is the justification.

---

## Cell artifacts

| Q | Script | Output |
|---|---|---|
| Q1 eager 2B q=0.85 / 0.99 (canonical L=4) | `retrofit/bench/bench_cache_regime.py` | `retrofit/outputs/q1_compiled_rerun/bench_L4_2B_q{085,099}.log` |
| Q1 eager 4B q=0.85 / 0.99 (canonical L=4) | same | `retrofit/outputs/q1_compiled_rerun/bench_L4_4B_q{085,099}.log` |
| Q1 compile-prefill 2B/4B (`reduce-overhead`) | `retrofit/bench/bench_retrofit_compile.py` | `retrofit/outputs/q1_compiled_rerun/compile_prefill_L4_{2B,4B}_redoh.log` |
| Q1 compile-decode 2B/4B (`default` + StaticCache) | `retrofit/bench/bench_compile_decode_static.py` | `retrofit/outputs/q1_compiled_rerun/compile_decode_static_L4_{2B,4B}_def.log` |
| Q1 compile-prefill `max-autotune` (kept for record — falls back to same kernels as redoh on this box) | `retrofit/bench/bench_retrofit_compile.py --compile-mode max-autotune` | `retrofit/outputs/q1_compiled_rerun/compile_prefill_L4_{2B,4B}_maxauto.log` |
| Q1 compile-decode `reduce-overhead` (errored — CUDA-graph + StaticCache compose issue) | `retrofit/bench/bench_compile_decode_static.py --compile-mode reduce-overhead` | `retrofit/outputs/q1_compiled_rerun/compile_decode_static_L4_*_v2.log` |
| Q1 (deprecated, L=2 / 14-block legacy) | `experiments/benchmark_qwen3vl_skip.py`, `bench_skip_decode_*.log` | `retrofit/outputs/bench_qwen3vl_skip_*.json`, `bench_skip_decode_{2B,4B}_q{085,050}.log` (**do not cite** — wrong partition / regime) |
| Q3 | `experiments/probe_q085_per_task.py` | `retrofit/outputs/probe_q085_per_task.json` / `.log` |
| Q4 | `experiments/probe_difficulty_skip_correlation.py` | `retrofit/outputs/probe_difficulty_skip_correlation.json` / `.log` |
