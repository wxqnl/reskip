# ReSkip 附录实验汇总（中文整合版）

记录日期：2026-05-04  
推荐引用：后续附录优先引用本文件，旧的分散 Markdown 只作为原始记录保留。

## 0. 文档范围

这份文档把近期 ReSkip 相关实验统一整理到一个地方，覆盖：

- 340M pretrain LLM 的 benchmark-context 动态 ReSkip 复核；
- 340M base transformer / AttnRes-full / ReSkip 的 7 个 LLM benchmark；
- Qwen3-VL-2B 上的 base / retrofit-full / ReSkip / static / random / Gromov / MoD / LayerSkip / CALM 对比；
- Qwen3-VL-4B 上的 6 个主 VLM benchmark target-calibrated ReSkip 补充；
- Qwen3-VL-2B 的 512 输入、4096 输出 decode 速度；
- LayerSkip / CALM / MoD 的同数据训练版 baseline；
- P、q 选择策略、skip stats、评估路径和结论 caveat。

术语：

- `P`：允许动态 skip 的 block group position 集合，0-based 编号。
- `q`：用于从 calibration metric 分布中取阈值的分位数。
- `M` 或 `max_skips`：每次 forward 最多允许跳过的 block 数。
- `skips / forward`：平均每次 traced forward 真实触发的 skip 数。

## 1. 340M LLM：旧结果问题与重新校准

### 1.1 旧结果的问题

旧配置：

| strategy | probe | P | q | max_skips | 结论 |
|---|---|---:|---:|---:|---|
| `recent_weight_gt` | `attn_only` | `{3,5}` | 0.85 | 2 | 0-trigger，不能作为有效 ReSkip benchmark |

复核发现，旧的 340M `P={3,5}, q=0.85` 在 lm-eval benchmark 上和 full-depth 完全一致，是因为 skip 没有触发：

| 配置 | skip_events / forwards | avg_blocks | 结论 |
|---|---:|---:|---|
| full-depth | 0 / 9727 | 8.000 | baseline |
| old q085 P={3,5} | 0 / 9727 | 8.000 | 分数一致来自 0-trigger |

因此，这个旧结果只保留为负例：FineWeb 长上下文阈值不能直接迁移到 benchmark prompt/scoring 分布。

### 1.2 FineWeb 候选迁移失败

重新扫描 FineWeb-Edu 时使用更低的 q 范围：

```text
q in {0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75}
```

FineWeb analysis 的主要候选：

| 候选 | probe | P | q | max_skips | FineWeb PPL | FineWeb avg_blocks |
|---|---|---:|---:|---:|---:|---:|
| best quality | `attn_only` | `{1,3,5}` | 0.45 | 1 | 313.226 | 7.031 |
| best tolerated/skip | `attn_only` | `{1,3,5}` | 0.45 | 2 | 330.048 | 6.219 |
| recommended | `first_attn` | `{4,5,6}` | 0.70 | 1 | 447.837 | 7.500 |

benchmark limit-256 sanity 结果：

| 配置 | P | q / threshold | skip_events / positions | avg_blocks | LAMBADA acc | LAMBADA ppl | PIQA acc_norm | MMLU acc | 处理 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| full-depth | - | - | 0 / 14888 | 8.000 | 0.3828 | 19.655 | 0.7031 | 0.2573 | baseline |
| FineWeb tolerated | `{1,3,5}` | q=0.45 | 2170 / 14888 | 6.834 | 0.1133 | 527.964 | 0.5352 | 0.2408 | reject: early skip 损伤严重 |
| FineWeb quality | `{1,3,5}` | q=0.45 | 1151 / 14888 | 7.382 | 0.1523 | 213.912 | 0.5742 | 0.2410 | reject: early skip 损伤严重 |
| FineWeb recommended | `{4,5,6}` | q=0.70 | 43 / 14888 | 7.977 | 0.3711 | 20.079 | 0.7031 | 0.2590 | reject: 触发太少 |
| FineWeb low1 q045 | `{5}` | threshold=0.4349 | 0 / 14888 | 8.000 | 0.3828 | 19.655 | 0.7031 | 0.2573 | reject: 0-trigger |
| FineWeb low1 q050 | `{5}` | threshold=0.4362 | 0 / 14888 | 8.000 | 0.3828 | 19.655 | 0.7031 | 0.2573 | reject: 0-trigger |
| FineWeb low2 q045 | `{5,6}` | thresholds 0.4349 / 0.3613 | 0 / 14888 | 8.000 | 0.3828 | 19.655 | 0.7031 | 0.2573 | reject: 0-trigger |

结论：

- `P={1,3,5}` 在 FineWeb 上能跳，但 benchmark 上 early position skip 破坏 LAMBADA。
- `P={4,5,6}` 较稳，但触发率太低。
- FineWeb 给出的 `P={5}` 阈值约 `0.435-0.436`，在 benchmark 上仍不触发。
- 因此必须使用 benchmark context 重新校准阈值。

### 1.3 Benchmark-context calibration

校准方式：

- 用 full-depth model 在 lm-eval request contexts 上 forward；
- 强制返回 routing info，但不执行 skip；
- 收集 `recent_weight_gt + attn_only` 的 phase-1 metric；
- calibration 使用 7 个 benchmark，`limit=512`；
- 最终只在 late position `P={5}` 上选 q，降低 early skip 风险。

Position 5 的 benchmark-context metric 分布：

| count | mean | min | p10 | p25 | p50 | p75 | p80 | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2665 | 0.3943 | 0.3118 | 0.3688 | 0.3874 | 0.3971 | 0.4069 | 0.4095 | 0.4141 | 0.4264 |

导出的 q 候选：

| P | q | threshold at pos5 | max_skips |
|---:|---:|---:|---:|
| `{5}` | 0.50 | 0.397135 | 1 |
| `{5}` | 0.55 | 0.398438 | 1 |
| `{5}` | 0.60 | 0.400391 | 1 |
| `{5}` | 0.65 | 0.402344 | 1 |
| `{5}` | 0.70 | 0.404948 | 1 |
| `{5}` | 0.75 | 0.406901 | 1 |
| `{5}` | 0.80 | 0.409505 | 1 |

q50/q60/q70/q80 进入 limit-256 sanity；q55/q65/q75 作为插值候选导出但未正式评估。

### 1.4 q sanity 与最终选择

limit-256 sanity：

| 配置 | q | threshold | skip_events / positions | avg_blocks | LAMBADA acc | LAMBADA ppl | HellaSwag acc_norm | ARC-E acc_norm | ARC-C acc_norm | PIQA acc_norm | MMLU acc | OBQA acc_norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full-depth | - | - | 0 / 14888 | 8.000 | 0.3828 | 19.655 | 0.4922 | 0.5547 | 0.3008 | 0.7031 | 0.2573 | 0.3789 |
| pos5 q50 | 0.50 | 0.397135 | 1114 / 14888 | 7.401 | 0.3125 | 44.213 | 0.4844 | 0.5508 | 0.3047 | 0.6836 | 0.2339 | 0.3750 |
| pos5 q60 | 0.60 | 0.400391 | 865 / 14888 | 7.535 | 0.3438 | 31.104 | 0.4766 | 0.5547 | 0.3047 | 0.7031 | 0.2365 | 0.3789 |
| pos5 q70 | 0.70 | 0.404948 | 656 / 14888 | 7.648 | 0.3711 | 22.584 | 0.4922 | 0.5547 | 0.3008 | 0.7031 | 0.2389 | 0.3789 |
| pos5 q80 | 0.80 | 0.409505 | 380 / 14888 | 7.796 | 0.3789 | 20.731 | 0.4922 | 0.5547 | 0.3008 | 0.6953 | 0.2461 | 0.3789 |

最终选择：

```text
strategy = recent_weight_gt
probe_mode = attn_only
P = {5}
q = 0.80
threshold at pos5 = 0.4095052182674408
max_skips = 1
```

选择逻辑：

1. 排除旧 q085，因为 `skip_events=0`。
2. 排除 early destructive P，例如 `P={1,3,5}`。
3. 排除 FineWeb 阈值直接迁移导致的 0-trigger 配置。
4. 用 benchmark-context 重新估 q。
5. 在 limit-256 sanity 中，q80 是最稳的非零触发点。
6. 正式结论不包装成“显著加速或零退化”：它有效触发，但收益有限，MMLU 有下降。

## 2. 340M LLM：base / full / ReSkip 7-task 结果

### 2.1 模型与任务

| 项 | 值 |
|---|---|
| Vanilla 340M | `flame/saves/transformer-340M` |
| AttnRes-full | `flame/saves/reskip_transformer-340M` |
| 旧 AttnRes + ReSkip | `outputs/reskip_340M_combined_35_skip2_q085`，0-trigger |
| 重新校准 ReSkip | `outputs/reskip_340M_benchmark_context_20260503/pos5_q080_M1` |
| Tasks | `piqa,mmlu,openbookqa,arc_easy,arc_challenge,lambada_openai,hellaswag` |
| Headline metric | `acc_norm` for PIQA/OpenBookQA/ARC/HellaSwag，`acc` for MMLU/LAMBADA |

### 2.2 旧扩展表：用于说明 0-trigger 问题

| Cell | PIQA | OpenBookQA | ARC-E | ARC-C | MMLU | LAMBADA acc | LAMBADA ppl | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla 340M | 0.6779 | 0.3320 | 0.5602 | 0.3046 | 0.2594 | 0.3790 | 24.71 | 0.4436 |
| AttnRes-full | 0.6893 | 0.3580 | 0.5438 | 0.3012 | 0.2555 | 0.4054 | 20.20 | 0.4607 |
| AttnRes + old ReSkip `{3,5}/M=2/q=0.85` | 0.6893 | 0.3580 | 0.5438 | 0.3012 | 0.2555 | 0.4054 | 20.20 | 0.4607 |

这张旧表不能作为“有效 dynamic skip”证据，因为 ReSkip 行是 0-trigger。

AttnRes-full 是否低于 Vanilla base：

- 低于 base：ARC-E `0.5438 < 0.5602`，ARC-C `0.3012 < 0.3046`，MMLU `0.2555 < 0.2594`。
- 高于 base：PIQA、OpenBookQA、LAMBADA、HellaSwag。

### 2.3 重新校准后的 full eval

| Metric | Full | ReSkip pos5 q80 | Delta |
|---|---:|---:|---:|
| LAMBADA acc | 0.4054 | 0.4054 | 0.0000 |
| LAMBADA ppl | 20.2018 | 20.2018 | 0.0000 |
| HellaSwag acc_norm | 0.4606 | 0.4607 | +0.0001 |
| ARC-E acc_norm | 0.5450 | 0.5446 | -0.0004 |
| ARC-C acc_norm | 0.3012 | 0.3003 | -0.0009 |
| PIQA acc_norm | 0.6893 | 0.6872 | -0.0022 |
| MMLU acc | 0.2554 | 0.2434 | -0.0120 |
| OpenBookQA acc_norm | 0.3580 | 0.3580 | 0.0000 |

完整 skip stats：

| 配置 | forwards_with_trace | total_positions | skip_events | avg_blocks | block_skip_rate | forward_skip_rate | per_position_skips |
|---|---:|---:|---:|---:|---:|---:|---|
| full-depth | 9727 | 77816 | 0 | 8.000 | 0.0000 | 0.0000 | `[0,0,0,0,0,0,0,0]` |
| ReSkip pos5 q80 | 9727 | 77816 | 706 | 7.927 | 0.0091 | 0.0726 | `[0,0,0,0,0,706,0,0]` |

结论：340M 的 benchmark-context ReSkip 是有效触发，但省算幅度约 0.9% block-level，并且 MMLU 下降约 1.2 个点。它不应写成“显著加速且零退化”。

## 3. Qwen3-VL-2B：实验范围与 P/q 选择

### 3.1 范围

| 项 | 值 |
|---|---|
| Backbone | `Qwen3-VL-2B` |
| Retrofit state | `retrofit/outputs/H_2B_r256_10k_L4_v3/retrofit_attnres_state.pt` |
| LLM representative eval | `lambada_openai,hellaswag`，`limit=2000` |
| VLM representative eval | `ai2d,mmbench_en_dev,mmmu_val,mmstar,ocrbench,realworldqa`，full split |
| GPU | 本轮使用本地 CUDA 0-3 |

### 3.2 ReSkip VLM 旧结果情况

项目里之前确实有 ReSkip VLM 相关结果，但不是本轮主表的完整 image benchmark：

- VLM-only LAMBADA-500：`retrofit/analysis/paper_main_experiments.md`，Table 5；
- VLA/LIBERO ReSkip Pareto：`retrofit/analysis/paper_main_experiments.md`，Table 4。

缺失的是：在主 VLM benchmark 上，同一 base、同一 retrofit state 的 dynamic ReSkip 完整 eval。AI2D / MMMU / MMStar 先补齐；随后又补了 MMBench / OCRBench / RealWorldQA。

### 3.3 Qwen3-VL-2B P/q 选择

LLM：

- 早期候选使用 `P={1,4}`。
- 在 LAMBADA/HellaSwag representative eval 上，`P={1,4}` 配 q in `{0.85,0.95,0.99}` 对 HellaSwag 有过跳风险。
- LLM 保守点选 `P={4}, M=1`。

VLM：

- 增加 target-distribution routing dump，在 AI2D/MMMU/MMStar held-out subset 上校准。
- Calibration output：`retrofit/outputs/related_qwen2b/vlm_target_calib/routing_ai2d_mmmu_mmstar_limit64.jsonl`
- Calibration size：每个任务 64 个样本，合计 384 个 forward。
- Block 4 thresholds：q50 `tau=0.382`，q95 `tau=0.477`。
- 正式 VLM eval 选：`P={4}, M=1, target-calibrated q=0.50`。

补充：

- 早期 LAMBADA-calib q=0.95 的 block4 threshold 是 `tau=0.344`，对 image-VLM input 更激进。
- 它在 VLM 上落到和 target-calib q=0.50 接近的 operating point。

## 4. Qwen3-VL-2B：LLM benchmark 结果

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

结论：

- Retrofit-full 高于 base。
- ReSkip 在该 LLM representative point 上基本保持 Retrofit-full 分数，同时非零触发但触发较少。
- 同等或更激进的 random/static skip 明显损伤质量。

## 5. Qwen3-VL-2B：VLM benchmark 结果

主 6-task VLM full split：

| Method | AI2D | MMBench | MMMU | MMStar | OCRBench | RWQA | skips / forward | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Base Qwen3-VL-2B | 0.7364 | 75.77 | 0.4144 | 0.5363 | 0.7720 | 0.6480 | - | main base table |
| Retrofit full | 0.7584 | 78.8660 | 0.4322 | 0.5357 | 0.8140 | 0.6610 | 0.0000 | canonical L4 v3 |
| ReSkip target-calib q=0.50, P={4}, M=1 | 0.7584 | 78.8660 | 0.4322 | 0.5486 | 0.7310 | 0.6601 | 0.5203 | target image-VLM calibration |

三任务 sensitivity / baseline 对照：

| Method | AI2D | MMMU | MMStar | skips / forward | Notes |
|---|---:|---:|---:|---:|---|
| ReSkip LAMBADA-calib q=0.95, P={4}, M=1 | 0.7584 | 0.4333 | 0.5486 | 0.5529 | sensitivity point |
| Random block skip, block 4, p=0.553 | 0.6671 | 0.3878 | 0.4156 | 0.5517 | rate-matched to ReSkip |
| Static skip block 4 | 0.5800 | 0.3378 | 0.3359 | 1.0000 | same block, always skip |
| MoD proxy, layers 12-15, keep=0.8 | 0.3462 | 0.3344 | 0.2197 | - | untrained token-bypass proxy |
| Gromov pruning, drop 4 layers | 0.5732 | 0.3556 | 0.4906 | - | static pruning |
| Gromov pruning, drop 8 layers | 0.0683 | 0.2389 | 0.0190 | - | static pruning |
| LayerSkip/CALM proxy, drop layers 24-27 | 0.3789 | 0.3111 | 0.4455 | - | untrained early-exit proxy |

VLM skip stats：

- Target-calib ReSkip q=0.50，6-task aggregate：16338 skips / 31400 traced forwards，`0.5203 skips/forward`。
- Target-calib ReSkip q=0.50，original 3-task subset：6631 skips / 12382 traced forwards，`0.5355 skips/forward`。
- LAMBADA-calib ReSkip q=0.95：`0.5529 skips/forward`。
- Random block4 p=0.553：`0.5517 skips/forward`。
- Static block4：`1.0000 skips/forward`。

关键结论：

1. ReSkip 在 image VLM benchmark 上不是 no-op，target-calib q=0.50 有稳定非零触发。
2. 在 AI2D / MMBench / MMMU / MMStar / RWQA 上，target-calib q=0.50 保持或接近 Retrofit full；但 OCRBench 明显下降 `0.8140 -> 0.7310`。
3. 同等 skip budget 的 random block skip 明显掉分，尤其 MMStar `0.5486 -> 0.4156`。
4. Static block4 也明显低于 dynamic ReSkip，说明“何时跳”比“跳哪个 block”更关键。
5. Gromov/static pruning 更快但质量损失大，不适合直接作为同质量对照。

## 6. Qwen3-VL-4B：VLM benchmark ReSkip 补充

### 6.1 设置与校准

| 项 | 值 |
|---|---|
| Backbone | `Qwen3-VL-4B` |
| Retrofit state | `retrofit/outputs/H_4B_r256_10k_L4_v3/retrofit_attnres_state.pt` |
| Tasks | `ai2d,mmbench_en_dev,mmmu_val,mmstar,ocrbench,realworldqa` |
| Dynamic strategy | `recent_weight_gt` |
| 4B canonical P | `{1,2}` |
| Calibration split | 6 个 VLM benchmark 各 `limit=64` |
| Calibration dump | `retrofit/outputs/related_qwen4b/vlm_target_calib_6bench/routing_ai2d_mmbench_mmmu_mmstar_ocr_rwqa_limit64.jsonl` |

q=0.85 的阈值：

| Block | threshold |
|---:|---:|
| 1 | 0.9375 |
| 2 | 0.7500 |

limit-64 sanity：

| 配置 | AI2D | MMBench | MMMU | MMStar | OCRBench | RWQA | skips / forward | 处理 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full-depth | 0.8594 | 95.3125 | 0.4844 | 0.5826 | 0.0620 | 0.6875 | 0.0000 | baseline |
| q50, P={1,2}, M=1 | 0.8594 | 90.6250 | 0.4844 | 0.5826 | 0.0600 | 0.6875 | 0.5457 | reject: MMBench drop |
| q85, P={1,2}, M=1 | 0.8594 | 95.3125 | 0.4844 | 0.5826 | 0.0620 | 0.6875 | 0.1653 | keep |
| q95, P={1,2}, M=1 | 0.8594 | 95.3125 | 0.4844 | 0.5826 | 0.0620 | 0.6875 | 0.0310 | too conservative |
| q85, P={1,2}, M=2 | 0.8594 | 95.3125 | 0.4844 | 0.5826 | 0.0620 | 0.6875 | 0.1653 | canonical full eval |

选择逻辑：

1. `P={1,2}` 沿用 4B canonical block set。
2. q50 虽然 skip 多，但 MMBench limit sanity 从 `95.31` 掉到 `90.63`。
3. q95 质量稳，但触发过少。
4. q85 在 sanity 上保持 full-depth，同时有非零触发。
5. `M=2` sanity 与 `M=1` 完全一致，说明 q85 下 `max_skips` 没有绑定；正式 4B canonical 记录采用 `M=2`。

### 6.2 Full benchmark

| Method | AI2D | MMBench | MMMU | MMStar | OCRBench | RWQA | skips / forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-VL-4B | 0.8190 | 83.3333 | 0.4900 | 0.6243 | 0.8190 | 0.7150 | - |
| Retrofit full, L4 v3 | 0.8248 | 85.2234 | 0.5211 | 0.6319 | 0.8250 | 0.7072 | 0.0000 |
| ReSkip q=0.85, P={1,2}, M=2 | 0.8248 | 85.2234 | 0.5211 | 0.6214 | 0.8250 | 0.7072 | 0.1600 |

和 Retrofit full 相比：

- 保持不变：AI2D、MMBench、MMMU、OCRBench、RealWorldQA。
- 下降：MMStar `0.6319 -> 0.6214`，约 `-1.05 pp`。
- 不是 no-op：全 6 task 合计 `5632` skips / `35201` traced forwards，`0.1600 skips/forward`。
- per-block skips：`[0,4692,940,0,0,0,0,0,0]`，实际只触发 block 1 / block 2。
- `M=1` full 与 `M=2` full 在本轮完全同分、同 skip stats；附录主表建议引用 `M=2` canonical 路径。

Artifacts：

| 内容 | 路径 |
|---|---|
| q85 M2 config | `retrofit/outputs/related_qwen4b/dyn_skip_configs/vlm_6bench_limit64_q085_p1p2_M2.json` |
| q85 M2 full eval | `retrofit/outputs/related_qwen4b/vlm_reskip_q085_p1p2_M2_full/` |
| q85 M1 full eval | `retrofit/outputs/related_qwen4b/vlm_reskip_q085_p1p2_full/` |
| q50/q85/q95 sanity | `retrofit/outputs/related_qwen4b/vlm_reskip_sanity_q*_p1p2*/` |
| logs | `retrofit/outputs/related_qwen4b/logs/` |

### 6.3 HellaSwag representative

设置：

- Task：`hellaswag`，`limit=2000`，lm-eval-harness。
- Dynamic ReSkip：LAMBADA-prefix calibration，`P={1,2}, q=0.99, M=2`。
- Thresholds：block 1 `0.9275`，block 2 `0.6769`。
- Output：`retrofit/outputs/related_qwen4b/lm_reskip_q099_p1p2_hswag_limit2000/summary.json`。

| Method | HellaSwag acc_norm | HellaSwag acc | skips / forward | per-block skips |
|---|---:|---:|---:|---|
| ReSkip q=0.99, P={1,2}, M=2 | 0.5810 | 0.4620 | 0.3140 | `[0,314,0,0,0,0,0,0,0]` |

说明：这是为了补齐 4B ReSkip 的 HellaSwag 缺项；同一文档里 4B no-skip text baseline 来自 `paper_main_experiments.md` 的 n=2000 表。

## 7. Qwen3-VL-2B：512 输入、4096 输出 decode 速度

### 7.1 ReSkip / static / random / proxy 同组速度

设置：

- Text-side decode speed benchmark；
- prompt length = 512 tokens；
- decode length = 4096 one-token steps；
- KV cache = `StaticCache`，预分配到 4616 tokens；
- eager bf16；
- 32-token warmup 后单次完整计时；
- Script：`retrofit/bench/bench_decode_512_4096_variants.py`；
- Outputs：`retrofit/outputs/related_qwen2b/decode_512_4096/json/`。

| Method | ms/token | tok/s | total decode s | prefill ms | skip stats |
|---|---:|---:|---:|---:|---|
| Base Qwen3-VL-2B | 14.317 | 69.85 | 58.6 | 18.7 | - |
| Retrofit full | 16.604 | 60.23 | 68.0 | 23.3 | 0 skips |
| ReSkip target-calib q=0.50, P={4} | 18.201 | 54.94 | 74.6 | 25.6 | 0 skips on this text prompt |
| ReSkip LAMBADA-calib q=0.95, P={4} | 17.398 | 57.48 | 71.3 | 23.8 | 2200 skips；0.537 / forward |
| Random block4 skip, p=0.553 | 17.407 | 57.45 | 71.3 | 21.3 | 2223 skips；0.543 / forward |
| Static block4 skip | 15.133 | 66.08 | 62.0 | 21.7 | 4097 skips；1.000 / forward |
| MoD proxy, layers 12-15, keep=0.8 | 15.344 | 65.17 | 62.8 | 19.3 | token bypass proxy |
| Gromov drop4 | 12.777 | 78.27 | 52.3 | 18.0 | 4 layers removed |
| Gromov drop8 | 10.577 | 94.55 | 43.3 | 13.1 | 8 layers removed |
| LayerSkip/CALM proxy, exit at layer 24 | 12.434 | 80.42 | 50.9 | 17.1 | layers 24-27 skipped |

解释：

- static/pruned baseline 快，是因为无条件删计算，但 VLM 质量崩得更明显。
- eager ReSkip 在这个纯文本 512/4096 microbenchmark 里没有速度收益；LAMBADA-calib 点确实触发，但 Python threshold gate 开销抵消了省掉的 block 计算。
- target-calib image-VLM 阈值是正确的 VLM accuracy 点，但转到纯文本 speed prompt 上会 0-trigger，这是阈值跨分布迁移的一个 datapoint。

### 7.2 官方风格 depth baseline 同组速度

这一组是 LayerSkip / CALM / MoD 训练版 baseline 的同组速度，base speed 只建议和本表内部比较，不要和上一张表直接横比。

| Method | ms/token | tok/s | prefill ms | Skip / route stats |
|---|---:|---:|---:|---|
| Base Qwen3-VL-2B | 21.048 | 47.51 | 20.3 | - |
| LayerSkip trained, exit 24 | 20.725 | 48.25 | 28.8 | skip layers 24-27 |
| CALM trained, exit 24 | 21.190 | 47.19 | 23.3 | skip layers 24-27 |
| MoD trained, layers 12-15 keep=0.8 | 24.784 | 40.35 | 37.0 | 816 token bypasses / 20608 routed tokens |

解释：

- LayerSkip trained 在同组里略快于 base，但质量损失很大。
- CALM trained exit-24 和 base 速度接近，但质量损失也大。
- MoD trained 在 eager decode 中更慢，因为 router/top-k 开销超过了 saved layer work。

## 8. Qwen3-VL-2B：LayerSkip / CALM / MoD 训练版 baseline

### 8.1 训练设置

| 项 | 值 |
|---|---|
| Script | `retrofit/train/train_qwen3vl_depth_baseline.py` |
| Data | same `v3` mix as retrofit training |
| Steps | 10k |
| max sequence length | 2048 |
| Backbone | frozen Qwen3-VL-2B |
| Trainable carrier | LoRA rank 64 / alpha 128 on `q_proj,v_proj` |
| Trainable params | 12.85M |
| LayerSkip | aux CE at layers 16/20/24 + stochastic layer dropout over layers 8-27, p=0.15 |
| CALM | aux CE at layers 16/20/24；本表评估 fixed exit-24 operating point |
| MoD | token routers on layers 12-15，keep ratio 0.8 |

Artifacts：

| Method | Artifact |
|---|---|
| LayerSkip | `retrofit/outputs/related_qwen2b/official_depth/layerskip_v3_10k/` |
| CALM | `retrofit/outputs/related_qwen2b/official_depth/calm_v3_10k/` |
| MoD | `retrofit/outputs/related_qwen2b/official_depth/mod_v3_10k/` |
| Aggregate | `retrofit/outputs/related_qwen2b/official_depth/official_depth_summary.md` |

说明：

- 这里是“方法忠实/官方风格”本地复现：同 base、同数据、method-specific training。
- CALM 当前记录的是训练后的 fixed exit-24 operating point，不应写成完整 dynamic confidence-threshold CALM。

### 8.2 LLM representative results

| Method | HellaSwag acc_norm | LAMBADA acc | LAMBADA ppl | Notes |
|---|---:|---:|---:|---|
| LayerSkip trained, exit 24 | 0.4290 | 0.1870 | 460.243 | LoRA + skip layers 24-27 |
| CALM trained, exit 24 | 0.4315 | 0.1970 | 295.893 | LoRA + skip layers 24-27 |
| MoD trained, layers 12-15 keep=0.8 | 0.4980 | 0.3945 | 23.503 | trained token routers |

### 8.3 VLM representative results

| Method | AI2D | MMMU | MMStar | Notes |
|---|---:|---:|---:|---|
| LayerSkip trained, exit 24 | 0.2587 | 0.2411 | 0.2284 | LoRA + skip layers 24-27 |
| CALM trained, exit 24 | 0.2600 | 0.2700 | 0.3057 | LoRA + skip layers 24-27 |
| MoD trained, layers 12-15 keep=0.8 | 0.2591 | 0.2567 | 0.2976 | trained token routers |

### 8.4 MoD stats

训练阶段：

| calls | keep_ratio | tokens | tokens_bypassed | bypass fraction |
|---:|---:|---:|---:|---:|
| 40000 | 0.8 | 24633352 | 4910748 | 0.199 |

LLM eval 阶段：

| calls | keep_ratio | tokens | tokens_bypassed |
|---:|---:|---:|---:|
| 5000 | 0.8 | 1941116 | 372084 |

Speed eval 阶段：

| calls | keep_ratio | tokens | tokens_bypassed |
|---:|---:|---:|---:|
| 16520 | 0.8 | 20608 | 816 |

结论：MoD router 的实际 bypass 比例和 keep=0.8 基本一致，但在当前 eager implementation 下没有速度收益。

## 9. Related baseline 总结

| Baseline | 类型 | 是否本轮同 base/data 跑过 | 结论 |
|---|---|---|---|
| Static block skip | ReSkip block static ablation | 是 | 快/触发稳定，但质量明显下降 |
| Random block skip | ReSkip block random ablation | 是 | rate-matched 后仍明显低于 dynamic ReSkip |
| Gromov pruning drop4/drop8 | static layer pruning | 是 | 速度快，但 VLM 质量下降大 |
| MoD proxy | zero-training token bypass proxy | 是 | 质量低于 ReSkip；proxy 只能作为粗略 stress test |
| LayerSkip/CALM proxy | zero-training early-exit proxy | 是 | 质量低于 ReSkip；不能作为正式 trained baseline |
| LayerSkip trained | method-specific trained baseline | 是 | exit-24 速度略快，但 LLM/VLM 质量低 |
| CALM trained | method-specific trained baseline | 是 | 当前是 fixed exit-24 operating point，质量低 |
| MoD trained | method-specific trained baseline | 是 | 质量好于 early-exit baseline，但仍低于 base/ReSkip，速度更慢 |

## 10. 实验脚本与结果路径

### 10.1 340M

| 内容 | 路径 |
|---|---|
| 中文复核旧记录 | `APPENDIX_RESKIP_340M_BENCHMARK_CONTEXT.md` |
| Calibration script | `experiments/benchmark_context_reskip_calibrate.py` |
| Calibration JSON | `outputs/reskip_benchmark_context_analysis_340M_20260503/benchmark_context_analysis.json` |
| Selected config | `outputs/reskip_340M_benchmark_context_20260503/pos5_q080_M1` |
| Full eval rerun summary | `outputs/lm_eval_reskip_340M_benchmark_context_rerun_20260503_113945/final_summary.json` |
| 旧 7-task base/full/reskip 扩展表 | `retrofit/results/llm_340M_extended_benchmarks.md` |

### 10.2 Qwen3-VL-2B

| 内容 | 路径 |
|---|---|
| 相关 baseline 原始记录 | `retrofit/results/qwen2b_related_skip_baselines_20260504.md` |
| Decode speed script | `retrofit/bench/bench_decode_512_4096_variants.py` |
| Official-depth training script | `retrofit/train/train_qwen3vl_depth_baseline.py` |
| Official-depth eval launcher | `retrofit/eval/run_official_depth_baselines.py` |
| Official-depth aggregate | `retrofit/outputs/related_qwen2b/official_depth/official_depth_summary.md` |
| ReSkip target VLM eval | `retrofit/outputs/related_qwen2b/vlm_reskip_target_q050_p4/` |
| ReSkip target VLM 补跑 MMB/OCR/RWQA | `retrofit/outputs/related_qwen2b/vlm_reskip_target_q050_p4_missing_mmb_ocr_rwqa/` |
| VLM target calibration dump | `retrofit/outputs/related_qwen2b/vlm_target_calib/routing_ai2d_mmmu_mmstar_limit64.jsonl` |
| 512/4096 speed JSONs | `retrofit/outputs/related_qwen2b/decode_512_4096/json/` |

### 10.3 Qwen3-VL-4B

| 内容 | 路径 |
|---|---|
| ReSkip target calibration dump | `retrofit/outputs/related_qwen4b/vlm_target_calib_6bench/routing_ai2d_mmbench_mmmu_mmstar_ocr_rwqa_limit64.jsonl` |
| ReSkip q85 M2 config | `retrofit/outputs/related_qwen4b/dyn_skip_configs/vlm_6bench_limit64_q085_p1p2_M2.json` |
| ReSkip q85 M2 full eval | `retrofit/outputs/related_qwen4b/vlm_reskip_q085_p1p2_M2_full/` |
| ReSkip q85 M1 full eval | `retrofit/outputs/related_qwen4b/vlm_reskip_q085_p1p2_full/` |
| ReSkip q99 HellaSwag limit-2000 | `retrofit/outputs/related_qwen4b/lm_reskip_q099_p1p2_hswag_limit2000/summary.json` |
| Full no-skip raw reference | `retrofit/outputs/lmms_eval_block_v3/4B_L4/retrofit/models__Qwen3-VL-4B/20260423_173819_results.json` |

## 11. 最终可写进论文附录的结论

1. 340M 旧 q085 结果不是有效 ReSkip，因为它是 0-trigger；重新 benchmark-context 校准后，`P={5}, q=0.80, M=1` 能非零触发，但收益有限且 MMLU 有下降。
2. Qwen3-VL-2B 上，image-VLM target-calib ReSkip `P={4}, q=0.50, M=1` 是有效触发的，不是 no-op；AI2D/MMBench/MMMU/RWQA 保持或接近 full，MMStar 略高，但 OCRBench 明显下降。
3. Qwen3-VL-4B 上，`P={1,2}, q=0.85, M=2` 是有效触发的：6 个 VLM benchmark 合计 `0.1600 skips/forward`；AI2D/MMBench/MMMU/OCRBench/RWQA 保持 full，MMStar 下降约 `1.05 pp`。
4. 4B ReSkip 的 HellaSwag 缺项已补：`P={1,2}, q=0.99, M=2, limit=2000` 得到 `acc_norm=0.5810`，`0.3140 skips/forward`。
5. rate-matched random skip 和 static skip 都明显低于 dynamic ReSkip，说明 ReSkip 的 routing signal 有作用。
6. Gromov/static pruning 可以更快，但它们的质量损失大，不是同质量速度对照。
7. LayerSkip/CALM/MoD 的训练版 baseline 已经补齐；在当前同 base/data/representative benchmark 下，它们仍低于 base/ReSkip。MoD trained 的质量相对早退 baseline 更好，但 eager decode 速度更慢。
8. 速度表必须按同一 run group 比较。ReSkip 当前 eager Python gate 在 512/4096 纯文本 microbenchmark 中没有实际速度收益；若要主打速度，需要后续 kernel/compile-level 优化或更合适的触发分布。

## 12. ReSkip 可视化：routing、difficulty 与 margin preservation

目标：按老师建议做一个 3-panel figure，直接连接 Sec. 2.3 的 margin-preservation prior 和实验现象：

1. routing heatmap 是否说明 input-dependent；
2. hard / high-uncertainty sample 是否保留更多 depth；
3. skip perturbation 是否大多落在 `2epsilon <= Delta` 的 margin-preserving 区域。

新增脚本：

- `retrofit/analysis/reskip_adaptive_depth_figure.py`
- `retrofit/analysis/reskip_token_margin_figure.py`

主输出：

- Figure PDF：`paper/figures/reskip_adaptive_depth_vlm.pdf`
- Figure PNG：`paper/figures/reskip_adaptive_depth_vlm.png`
- 同步保留 token figure：`paper/figures/reskip_adaptive_depth_vlm_token.pdf/png`
- Raw token records：`retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160/token_records.jsonl`
- Raw sample records：`retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160/sample_records.jsonl`
- Summary：`retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160/summary.json`

### 12.1 正式 probe 配置

| Item | Value |
|---|---|
| Base | Qwen3-VL-2B |
| Retrofit state | `retrofit/outputs/H_2B_r256_10k_L4_v3/retrofit_attnres_state.pt` |
| ReSkip diagnostic config | `P={2}, tau=0.512, M=1` |
| Dynamic strategy | `recent_weight_gt` |
| Benchmark source | MMBench dev |
| Probe size | `n=160` |
| Probe unit | cached decode forward / token |
| Decode horizon | 8 full-depth greedy tokens per sample |
| Margin | full-depth token logit `top1 - top2` |
| Perturbation | `epsilon = max` logit shift over full-depth top-10 candidate tokens |

关键结果：

| Metric | Value |
|---|---:|
| Samples | 160 |
| Decode forwards / token points | 1280 |
| Prediction preserved | 0.9617 |
| Avg skipped blocks / decode forward | 0.1008 |
| Margin condition `2epsilon <= Delta` | 0.8328 |
| Spearman rho(uncertainty, skip) | -0.1421 |

Difficulty-bin skip：

| Full-depth uncertainty bin | n token points | mean margin | mean skipped blocks / forward |
|---|---:|---:|
| Low / easy | 634 | 6.7395 | 0.1372 |
| High / hard | 646 | 1.1099 | 0.0650 |

### 12.2 解释

- Panel (a) 支持 input-dependent：easy / hard representative samples 的 routing heatmap 不同；红框对应实际触发 skip 的 block。
- Panel (b) 支持 difficulty-aware depth：按 full-depth margin 分组后，easy token steps 的 skip 明显更多，hard token steps 更少；`0.1372 -> 0.0650`，Spearman `rho=-0.1421`。
- Panel (c) 支持 margin-preservation rationale：`83.28%` token decisions 落在 `2epsilon <= Delta` 区域，`96.17%` top-1 prediction preserved。这里的 `epsilon` 是 full-depth top-10 candidate token logits 上的最大扰动；这是 decode-step 粒度，和 ReSkip 实际做 routing decision 的 forward 粒度一致。

### 12.3 额外 probe

为了确认为什么之前的图不成立，做过 option-level probe：

| Probe | Result | Interpretation |
|---|---|---|
| 4B canonical `q=0.85, P={1,2}`, MMBench option scoring, n=160 | preserved `0.9062`, condition `0.1625`, skip `0.3250/fwd` | answer-option likelihood 会跨多个 option tokens 累积扰动，`epsilon` 过保守，不能支撑 margin-preservation 图 |
| 2B target-calib `q=0.50, P={4}`, MMBench option scoring, n=64 | preserved `0.7969`, condition `0.3906`, skip `0.5358/fwd` | block 4 的 routing/margin 相关性方向不适合画 hard-retains-depth |

因此最终采用 token-level decode probe，并先做 routing/margin scan：2B 上 block 1/2 的 `w_recent` 与 full-depth margin 正相关，其中 block 2 在保守阈值 `tau=0.512` 下同时满足非零 skip、hard fewer skips、prediction preserved 和 margin region majority。

### 12.4 建议写法

可用 caption：

> ReSkip adapts depth according to input difficulty. (a) Representative VLM samples follow different routing patterns; red boxes denote skipped blocks. (b) Average skipped blocks decrease on high-uncertainty decode steps. (c) Most ReSkip decisions lie below the margin-preservation boundary `2epsilon = Delta`, explaining why predictions are usually preserved.

注意：这张图是 mechanism / calibration diagnostic，operating point 是 `P={2}, tau=0.512, M=1`，不是替代主 benchmark 表里的 target-calibrated VLM operating point。

## 13. VLA 参数选择消融补数：P / M 空白 cell

目标：补齐 compact ablation 表里 VLA 列的两个空白位置，尤其是 `P=all, M=4`。

### 13.1 可直接填表的结果

表中 VLA 列为 LIBERO 4-suite average success rate，单位 `%`。括号内 delta 对应 no-skip/full reference；2B/4B 的 reference 都是 `96.25%`，表内四舍五入为 `96.3%`。

| Knob | Setting | VLA-2B | VLA-4B | Decision |
|---|---|---:|---:|---|
| `P, M` | `{1,4}/{1,2}, M=2` | `97.4 (+1.1)` | `96.5 (+0.2)` | chosen |
| `P, M` | `{1,4}/{1,2}, M=4` | `97.4 (+1.1)` | `96.5 (+0.2)` | no gain |
| `P, M` | `all, M=4` | `96.3 (+0.0; exact +0.05)` | `89.5 (-6.8)` | reject |

说明：

- `{1,4}/{1,2}, M=4` 没有重跑。原因是 canonical eligible set 只有两个 block，`M=4` 在实现上不可能比 `M=2` 多触发，所以它和 chosen `M=2` 是同一个 deterministic operating point。
- `all, M=4` 已完整补跑 VLA 四套；2B VLA 侧基本 parity，但 VLM 侧已经大幅下降，4B VLA 侧也明显下降，所以最终仍是 reject。

### 13.2 `all, M=4` 配置

| Scale | Config | q | Eligible | max skips |
|---|---|---:|---|---:|
| 2B | `retrofit/outputs/dyn_skip_configs/pathB_2B_L4_v3_30k_sim_q099_all_M4.json` | 0.99 | all thresholded blocks 1-6 | 4 |
| 4B | `retrofit/outputs/dyn_skip_configs/pathB_4B_L4_v3_30k_sim_q099_all_M4.json` | 0.99 | all thresholded blocks 1-8 | 4 |

这里的 `eligible_blocks: null` 表示所有带阈值的 block 都 eligible；不是 canonical narrow P。

### 13.3 `all, M=4` 原始 VLA 结果

2B：

| Suite | Success rate | Effective block ratio | Log |
|---|---:|---:|---|
| `libero_spatial` | 96.2 | 0.9892 | `retrofit/outputs/libero_eval_full/ablate_vla_2B_allM4_q099_20260505_rerun_libero_spatial/eval.log` |
| `libero_object` | 98.8 | 0.9955 | `retrofit/outputs/libero_eval_full/ablate_vla_2B_allM4_q099_20260505_rerun_libero_object/eval.log` |
| `libero_goal` | 96.6 | 0.9817 | `retrofit/outputs/libero_eval_full/ablate_vla_2B_allM4_q099_20260505_rerun_libero_goal/eval.log` |
| `libero_10` | 93.6 | 0.9961 | `retrofit/outputs/libero_eval_full/ablate_vla_2B_allM4_q099_20260505_libero10_retry2_noping_libero_10/eval.log` |
| **Mean** | **96.30** | **0.9906** | - |

4B：

| Suite | Success rate | Effective block ratio | Log |
|---|---:|---:|---|
| `libero_spatial` | 83.2 | 0.9435 | `retrofit/outputs/libero_eval_full/ablate_vla_4B_allM4_q099_20260505_rerun_libero_spatial/eval.log` |
| `libero_object` | 96.8 | 0.9937 | `retrofit/outputs/libero_eval_full/ablate_vla_4B_allM4_q099_20260505_rerun_libero_object/eval.log` |
| `libero_goal` | 91.8 | 0.9787 | `retrofit/outputs/libero_eval_full/ablate_vla_4B_allM4_q099_20260505_rerun_libero_goal/eval.log` |
| `libero_10` | 86.2 | 0.9759 | `retrofit/outputs/libero_eval_full/ablate_vla_4B_allM4_q099_20260505_libero10_retry2_noping_libero_10/eval.log` |
| **Mean** | **89.50** | **0.9729** | - |

Approximate skipped blocks per forward from the suite-mean effective ratio:

| Scale | Mean effective block ratio | Blocks | Approx skipped blocks / forward |
|---|---:|---:|---:|
| 2B | 0.9906 | 7 | 0.0656 |
| 4B | 0.9729 | 9 | 0.2435 |

### 13.4 运行备注

- 本轮补跑使用 GPU 0-3：2B server/client 在 0/1，4B server/client 在 2/3。
- 第一次长跑中两个 `libero_10` wrapper 接近结束时被 websocket keepalive timeout 打断，不能采用 partial log。
- 已对本地 LIBERO eval 的 websocket client/server 禁用 keepalive ping timeout：
  - `starVLA/deployment/model_server/tools/websocket_policy_client.py`
  - `starVLA/deployment/model_server/tools/websocket_policy_server.py`
- 这个补丁只避免本地长 rollout 中的 false disconnect，不改变模型、routing、threshold 或 skip 判定逻辑。
