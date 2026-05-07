# Qwen3-VL-2B Related Skip Baselines (2026-05-04)

## Scope

- Backbone: `Qwen3-VL-2B`.
- Retrofit state: `retrofit/outputs/H_2B_r256_10k_L4_v3/retrofit_attnres_state.pt`.
- LLM representative eval: `lambada_openai,hellaswag`, `--limit 2000`.
- VLM representative eval: `ai2d,mmmu_val,mmstar`, full split.
- GPU range used: local CUDA devices 0-3.

## Paper-final compact table (`Reskip_5_7_3.pdf` Table 3)

下面这张表是定稿 paper-facing 版本。后面的小节保留原始探索日志和更细粒度 artifact 路径；如果数值口径冲突，以本表和 `paper/Reskip_5_7_3_FINAL_DATA.md` 为准。

| Method | Lamb.-500 | HSwag-500 | AI2D | MMMU | MMStar | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Static block skip | 37.3 | 53.8 | 58.0 | 33.8 | 33.6 | 73.1 |
| Gromov-4 | 6.9 | 44.4 | 57.3 | 35.6 | 49.1 | 78.3 |
| Gromov-8 | 2.0 | 36.5 | 6.8 | 23.9 | 1.9 | 94.6 |
| MoD | 47.4 | 56.1 | 34.6 | 33.4 | 22.0 | 71.2 |
| LayerSkip | 14.6 | 46.0 | 37.9 | 31.1 | 44.6 | 80.4 |
| Base (LoRA) | 53.3 | 55.2 | 73.6 | 41.4 | 53.6 | 69.9 |
| AR-Retrofit full (ours) | 56.4 | 58.7 | 75.8 | 43.2 | 53.6 | 57.2 |
| ReSkip (ours) | 56.2 | 57.9 | 75.9 | 43.0 | 54.9 | 67.5 |

## Existing ReSkip VLM Results

The project did already have ReSkip VLM results, but they were not full
image-benchmark `lmms-eval` runs:

- VLM-only LAMBADA-500: `retrofit/analysis/paper_main_experiments.md`, Table 5.
  `dynskip q=0.85` gives acc 0.560 vs no-skip 0.570, ppl 5.258 vs 4.526,
  avg skips 0.19 / max 2.
- VLA/LIBERO ReSkip Pareto: `retrofit/analysis/paper_main_experiments.md`,
  Table 4.

The missing cell was dynamic ReSkip on the same image VLM benchmarks used by
the main base/full table. That cell is now run below.

## P and q Selection

For Qwen3-VL-2B L4, earlier analysis used P={1,4}. On the representative LLM
eval, P={1,4} with q in {0.85,0.95,0.99} over-skips HellaSwag. The conservative
safe point is P={4}, M=1.

For image VLM, I added a target-distribution routing dump and calibrated
thresholds on a held-out subset of AI2D/MMMU/MMStar:

- Calibration command output:
  `retrofit/outputs/related_qwen2b/vlm_target_calib/routing_ai2d_mmmu_mmstar_limit64.jsonl`
- Calibration size: 64 examples per task, 384 forwards.
- Block 4 thresholds: q50 tau=0.382, q95 tau=0.477.
- Full eval selection: P={4}, M=1, target-calibrated q=0.50.

The earlier LAMBADA-calibrated q=0.95 sensitivity point used tau=0.344 for
block 4, which is more aggressive on image-VLM inputs. It lands at nearly the
same operating point as target-calibrated q=0.50.

## LLM Results

| Method | HellaSwag acc_norm | LAMBADA acc | LAMBADA ppl | skips / forward | Notes |
|---|---:|---:|---:|---:|---|
| Base Qwen3-VL-2B | 0.5515 | 0.5325 | 9.259 | - | `lm_base` |
| Retrofit full | 0.5870 | 0.5640 | 6.976 | 0.000 | `lm_retrofit_full` |
| ReSkip, P={4}, q=0.95, M=1 | 0.5870 | 0.5640 | 7.037 | 0.0416 | dynamic, near-lossless |
| Random block skip, P={1,4}, p=0.0208 | 0.5825 | 0.5590 | 7.239 | 0.0328 | rate-near ReSkip |
| Static skip block 4 | 0.5375 | 0.3730 | 23.548 | 1.0000 | same block, always skip |
| Static skip block 1 | 0.4885 | 0.3880 | 21.039 | 1.0000 | early block stress |
| MoD proxy, layers 12-15, keep=0.8 | 0.5605 | 0.4740 | 11.821 | - | untrained token-bypass proxy |
| LayerSkip/CALM proxy, drop layers 24-27 | 0.4595 | 0.1460 | 7954.696 | - | untrained early-exit proxy |
| Gromov pruning, drop 4 layers | 0.4440 | 0.0690 | 1713.714 | - | static layer-pruning baseline |

LLM artifacts live under `retrofit/outputs/related_qwen2b/lm_*/summary.json`.

## VLM Results

| Method | AI2D | MMMU | MMStar | skips / forward | Notes |
|---|---:|---:|---:|---:|---|
| Base Qwen3-VL-2B | 0.7364 | 0.4144 | 0.5363 | - | main base table |
| Retrofit full | 0.7584 | 0.4322 | 0.5357 | 0.000 | canonical L4 v3 |
| ReSkip target-calib q=0.50, P={4}, M=1 | 0.7584 | 0.4322 | 0.5486 | 0.5355 | target image-VLM calibration |
| ReSkip LAMBADA-calib q=0.95, P={4}, M=1 | 0.7584 | 0.4333 | 0.5486 | 0.5529 | sensitivity point |
| Random block skip, block 4, p=0.553 | 0.6671 | 0.3878 | 0.4156 | 0.5517 | rate-matched to ReSkip |
| Static skip block 4 | 0.5800 | 0.3378 | 0.3359 | 1.0000 | same block, always skip |
| MoD proxy, layers 12-15, keep=0.8 | 0.3462 | 0.3344 | 0.2197 | - | untrained token-bypass proxy |
| Gromov pruning, drop 4 layers | 0.5732 | 0.3556 | 0.4906 | - | static pruning |
| Gromov pruning, drop 8 layers | 0.0683 | 0.2389 | 0.0190 | - | static pruning |
| LayerSkip/CALM proxy, drop layers 24-27 | 0.3789 | 0.3111 | 0.4455 | - | untrained early-exit proxy |

VLM artifacts:

- Target ReSkip:
  `retrofit/outputs/related_qwen2b/vlm_reskip_target_q050_p4/`
- LAMBADA-calib ReSkip sensitivity:
  `retrofit/outputs/related_qwen2b/vlm_reskip_q095_p4/`
- Random rate-match:
  `retrofit/outputs/related_qwen2b/vlm_random_p0553/`
- Static skip:
  `retrofit/outputs/related_qwen2b/vlm_static_b4/`
- MoD proxy:
  `retrofit/outputs/related_qwen2b/vlm_mod_proxy/`
- Gromov pruning:
  `retrofit/outputs/static_pruning/drop_4/vlm/`,
  `retrofit/outputs/static_pruning/drop_8/vlm/`

## Official-Style Trained Depth Baselines

I then filled the missing trained-baseline comparison for LayerSkip, CALM, and
MoD on the same Qwen3-VL-2B backbone and the same v3 retrofit data mix.

Training setup:

- Script: `retrofit/train/train_qwen3vl_depth_baseline.py`.
- Data: same `v3` mix as retrofit training.
- Steps: 10k, max sequence length 2048.
- Backbone: frozen Qwen3-VL-2B.
- Trainable carrier: LoRA rank 64 / alpha 128 on `q_proj,v_proj`, 12.85M
  trainable parameters.
- LayerSkip: early-exit supervision at layers 16/20/24 plus stochastic layer
  dropout over decoder layers 8-27, p=0.15.
- CALM: early-exit supervision at layers 16/20/24; evaluated at the same
  skip-enabled exit-24 operating point for the speed/accuracy table.
- MoD: trained token routers on decoder layers 12-15, keep ratio 0.8.

Training artifacts:

- LayerSkip: `retrofit/outputs/related_qwen2b/official_depth/layerskip_v3_10k/`
- CALM: `retrofit/outputs/related_qwen2b/official_depth/calm_v3_10k/`
- MoD: `retrofit/outputs/related_qwen2b/official_depth/mod_v3_10k/`
- Aggregate: `retrofit/outputs/related_qwen2b/official_depth/official_depth_summary.md`

Official-style LLM representative results:

| Method | HellaSwag acc_norm | LAMBADA acc | LAMBADA ppl | Notes |
|---|---:|---:|---:|---|
| LayerSkip trained, exit 24 | 0.4290 | 0.1870 | 460.243 | LoRA + skip layers 24-27 |
| CALM trained, exit 24 | 0.4315 | 0.1970 | 295.893 | LoRA + skip layers 24-27 |
| MoD trained, layers 12-15 keep=0.8 | 0.4980 | 0.3945 | 23.503 | trained token routers |

Official-style VLM representative results:

| Method | AI2D | MMMU | MMStar | Notes |
|---|---:|---:|---:|---|
| LayerSkip trained, exit 24 | 0.2587 | 0.2411 | 0.2284 | LoRA + skip layers 24-27 |
| CALM trained, exit 24 | 0.2600 | 0.2700 | 0.3057 | LoRA + skip layers 24-27 |
| MoD trained, layers 12-15 keep=0.8 | 0.2591 | 0.2567 | 0.2976 | trained token routers |

Official-style 512 -> 4096 decode speed, same run group:

| Method | ms/token | tok/s | prefill ms | Skip / route stats |
|---|---:|---:|---:|---|
| Base Qwen3-VL-2B | 21.048 | 47.51 | 20.3 | - |
| LayerSkip trained, exit 24 | 20.725 | 48.25 | 28.8 | skip layers 24-27 |
| CALM trained, exit 24 | 21.190 | 47.19 | 23.3 | skip layers 24-27 |
| MoD trained, layers 12-15 keep=0.8 | 24.784 | 40.35 | 37.0 | 816 token bypasses over 20608 routed tokens |

MoD training stats: 4,910,748 token bypasses over 24,633,352 routed tokens
(`keep_ratio=0.8`, actual bypass fraction 0.199).

Outcome: these trained depth baselines do not close the gap to base or ReSkip
on the selected representative benchmarks. In particular, LayerSkip/CALM lose
most VLM accuracy at the exit-24 operating point, while trained MoD preserves
more LLM accuracy but is still below base/ReSkip and is slower in eager decode
because the per-token router/top-k overhead exceeds the saved layer work.

## Decode Speed at 512 -> 4096

Setup:

- Text-side decode-speed benchmark on Qwen3-VL-2B.
- Prompt length: 512 tokens.
- Decode length: 4096 one-token steps.
- KV cache: `StaticCache`, preallocated to 4616 tokens.
- Mode: eager bf16, single timed full decode run after a 32-token warmup.
- Script: `retrofit/bench/bench_decode_512_4096_variants.py`.
- Outputs: `retrofit/outputs/related_qwen2b/decode_512_4096/json/`.

| Method | tok/s | Paper-facing note |
|---|---:|---|
| Base (LoRA) | 69.9 | Qwen3-VL-2B base speed |
| AR-Retrofit full | 57.2 | full retrofit, no skip |
| ReSkip | 67.5 | calibrated dynamic skip |
| Static block skip | 73.1 | faster but quality collapses |
| Gromov-4 | 78.3 | static pruning |
| Gromov-8 | 94.6 | static pruning, catastrophic quality |
| MoD | 71.2 | trained/proxy row reported in Table 3 |
| LayerSkip | 80.4 | exit/skip row reported in Table 3 |

Interpretation:

1. Static/pruned baselines are faster because they remove compute
   unconditionally, but their VLM accuracy collapses.
2. Eager ReSkip does not produce a speedup in this pure-text 512/4096
   microbenchmark. At the LAMBADA-calibrated operating point it fires at the
   intended rate, but the Python threshold gate offsets the saved block work.
3. Target-calibrated image-VLM ReSkip is the correct accuracy cell for
   AI2D/MMMU/MMStar, but on this pure-text speed prompt it fires zero times;
   this is another threshold-transfer datapoint.

## Interpretation

1. ReSkip is not a no-op on image VLM: target-calibrated q=0.50 fires 6631
   skips over 12382 forwards, 0.5355 skips/forward.
2. The same skip budget with random block-4 skipping drops all three VLM
   metrics, especially MMStar (0.5486 -> 0.4156).
3. Static block-4 skipping is much worse than dynamic ReSkip, despite using
   the same block.
4. The earlier CALM/LayerSkip and MoD rows are zero-training proxies. The
   trained official-style rows above now fill the missing method-specific
   comparison on the same base and data mix, and they remain below ReSkip.
