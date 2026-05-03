# NeurIPS-prep extra benchmark run — 2026-04-29

Tracking doc for the experiments launched 2026-04-29 to expand paper benchmark
coverage. All results feed into `paper/main.tex` updates (no auxiliary cells).

## Phase 1 — VLM benchmark expansion (2B / 4B base + retrofit)

**Goal**: extend `tab:retrofit_main` from current 6 VLM benchmarks (MMBench /
MMMU / MMStar / AI2D / OCRBench / RWQA) to 9 by adding MathVista_testmini /
ChartQA / POPE.

**Cells (4-way, parallel on GPUs 0-3)**:

| GPU | Model | State | Output dir |
|---|---|---|---|
| 0 | Qwen3-VL-2B base | — | `retrofit/outputs/lmms_eval_extra/2B_base/` |
| 1 | Qwen3-VL-2B retrofit (L=4 v3 10k) | `outputs/H_2B_r256_10k_L4_v3/retrofit_attnres_state.pt` | `retrofit/outputs/lmms_eval_extra/2B_retrofit/` |
| 2 | Qwen3-VL-4B base | — | `retrofit/outputs/lmms_eval_extra/4B_base/` |
| 3 | Qwen3-VL-4B retrofit (L=4 v3 10k) | `outputs/H_4B_r256_10k_L4_v3/retrofit_attnres_state.pt` | `retrofit/outputs/lmms_eval_extra/4B_retrofit/` |

**Tasks**: `mathvista_testmini,chartqa,pope` via `lmms_eval` (model
`qwen3_vl_retrofit` for retrofit cells, `qwen3_vl` for base). Wrapper:
`retrofit/eval/run_vlm_extra.sh`.

**Launch**: `bash retrofit/eval/run_vlm_extra.sh <cell> <gpu>` per cell. Each
launched at ~11:34 2026-04-29 via the 4 per-GPU pipeline scripts in
`retrofit/outputs/pipeline_logs/{2B_base,2B_retrofit,4B_base,4B_retrofit}.log`.

## Phase 2 — 340M LLM benchmark expansion

**Goal**: extend the 340M from-scratch existence-proof Table 2 (currently
`tab:reskip_benchmark` with LAMBADA/HellaSwag/ARC-E/ARC-C, 4 tasks) to 7 tasks
by adding PIQA / MMLU / OpenBookQA, and verify the lossless-skip claim still
holds on the new tasks.

**Cells (3-way, parallel on GPUs 0-2 after Phase 1 finishes; watchdogs auto-fire)**:

| GPU | Config | Model dir | Output dir |
|---|---|---|---|
| 0 | vanilla 340M (no AttnRes baseline) | `flame/saves/transformer-340M` | `retrofit/outputs/lm_eval_340M_extra/vanilla/` |
| 1 | AttnRes 340M full-depth | `flame/saves/reskip_transformer-340M` | `retrofit/outputs/lm_eval_340M_extra/attnres_full/` |
| 2 | AttnRes + ReSkip ({3,5}/M=2/q=0.85) | `outputs/reskip_340M_combined_35_skip2_q085` | `retrofit/outputs/lm_eval_340M_extra/attnres_reskip_35_q085/` |

GPU 3 explicitly idle in Phase 2 — paper-scope-minimalism rule (the
({2,3}/M=2/q=0.93) variant I had queued is appendix-only at PPL-ratio level,
not part of `tab:reskip_benchmark`).

**Tasks**: `piqa,mmlu,openbookqa,arc_easy,arc_challenge,lambada_openai,hellaswag`
via `experiments/flame_lm_eval.py`. ARC + LAMBADA + HellaSwag are re-run for
sanity (and to be on the same harness call as the new tasks).

**Watchdogs** (poll the corresponding VLM bash wrapper PID, fire 340M lm-eval
when VLM done):

- GPU 0: PID 3021368 → waits on 2950065 (2B_base VLM)
- GPU 1: PID 3021760 → waits on 2952610 (2B_retrofit VLM)
- GPU 2: PID 3022168 → waits on 2957062 (4B_base VLM)

Logs: `retrofit/outputs/pipeline_logs/{gpu0_340M_vanilla,gpu1_340M_attnres_full,gpu2_340M_reskip35}.log`.

The ({5}/M=1/q=0.95) row that's currently in `tab:reskip_benchmark` of main.tex
shows numbers identical to Full-depth, but no prepared model dir exists for
that exact config and there is no calibration record I could find. Either
the row is decorative / a placeholder, or the calibration data lives outside
this output tree. **Decision (per user)**: skip recalibrating that row;
when updating the table, either drop the row or keep it with a "—" / asterisk
in the new columns.

## Phase 3 — Static-pruning baseline (depth-axis comparison)

**Goal**: replace the qualitative `app:method_comparison` table with numerical
head-to-head comparison vs. ReSkip on the same Qwen3-VL-2B backbone.

**Cells (2-way, parallel on GPUs 0/1 after Phase 2 finishes)**:

| GPU | drop count | Layers picked (lowest Gromov influence first) | Output dir |
|---|---|---|---|
| 3 | (ranking only) | run `gromov_baseline.py`, save JSON | `retrofit/outputs/static_pruning/gromov_ranking_2B.json` |
| 0 | drop 4 | top-4 of Gromov ranking | `retrofit/outputs/static_pruning/drop_4/` |
| 1 | drop 8 | top-8 of Gromov ranking | `retrofit/outputs/static_pruning/drop_8/` |

drop-4 ≈ ReSkip average compute (avg ~1.06 skips/forward × 4 layers/block).
drop-8 ≈ ReSkip M_max worst case.

**Tasks**: each cell runs the full 9 lmms-eval VLM benchmarks
(MMBench/MMMU/MMStar/AI2D/OCRBench/RWQA + MathVista/ChartQA/POPE) plus
LAMBADA + HellaSwag — same column set as the retrofit cells in `tab:retrofit_main`,
so the comparison is apples-to-apples.

**Watchdogs**:
- GPU 3 (PID 3078882): waits for 4B-retrofit VLM PID 2963964 → Gromov ranking.
- GPU 0 (PID 3078948): waits for the 340M-vanilla bash wrapper (PID 3021368) and
  the Gromov JSON → fires drop-4 cell.
- GPU 1 (PID 3079246): waits for the 340M-AttnRes-full bash wrapper (PID 3021760)
  and the Gromov JSON → fires drop-8 cell.

Logs: `retrofit/outputs/pipeline_logs/{gpu3_gromov,gpu0_prune_drop4,gpu1_prune_drop8}.log`.

## Phase 4 — VLA landscape table

**Goal**: extend `tab:vla_landscape` (currently 5 cited rows + ours) by 5 more
cited rows.

**No experiments**. Citations from `retrofit/results/vla_landscape_research_2026-04-29.md`:
OpenVLA-OFT (kim2025openvlaoft), CogACT (li2024cogact), π0-FAST (pertsch2025fast),
NORA-Long (hung2025nora), GR00T N1 (nvidia2025groot).

## Phase 5 — paper update (no GPU)

After Phases 1 + 2 + 3 finish, update `paper/main.tex`:

1. Extend `tab:retrofit_main` with 3 new VLM columns (MathVista / ChartQA / POPE).
2. Extend `tab:reskip_benchmark` with 3 new task columns (PIQA / MMLU / OpenBookQA);
   handle the ({5}/M=1/q=0.95) row per the decision above.
3. Replace the qualitative `app:method_comparison` table with numerical
   static-pruning vs. ReSkip rows (drop-4 + drop-8).
4. Extend `tab:vla_landscape` with 5 new cited rows + bib entries.
5. Update abstract / Section 3 prose / discussion if narrative shifts.

## What we are NOT doing this session

- `({5}/M=1/q=0.95)` recalibration on 340M.
- `({2,3}/M=2/q=0.93)` cross-cell in Table 2 (appendix-only).
- 110M cross-scale extension to the new tasks (existing appendix mention is
  enough).
- 2B/4B LLM-side benchmarks (per user clarification: 2B+ is VLM/VLA scope only).
- ShortGPT-style consecutive-block pruning (we use Gromov non-consecutive selection only — paper just needs ONE depth-axis baseline, not two).
