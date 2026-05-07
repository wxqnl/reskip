# Appendix: 340M Dynamic ReSkip Benchmark-Context Calibration

记录日期：2026-05-03

本附录记录 340M pretrain LLM 的 dynamic ReSkip 复核流程，包括 P / q 选择、候选评估、skip stats、最终选择策略和正式 benchmark 结果。这里的 `P` 表示允许动态跳过的 block-group position 集合，position 使用 0-based 编号；`P={5}` 表示第 6 个 block group。

## A. 实验目标

旧的 `recent_weight_gt + attn_only + P={3,5} + q=0.85 + max_skips=2` 在 `lm-eval` benchmark 上与 full-depth 分数完全一致。复核后发现该结果是 `0-trigger`：skip 没有真实发生，因此不能作为有效 ReSkip benchmark。

本轮重新走完整流程：

1. 用带 `skip_stats` 的真实 `lm-eval` forward 验证旧结果。
2. 用 FineWeb-Edu analysis 重新扫描较低 q 范围，检查候选 P / q 的迁移性。
3. 在 benchmark context 上做 full-depth trace calibration，重新估计 q 阈值。
4. 对候选 q 做 limit-256 sanity。
5. 选定最终策略后跑完整 7-task eval，并复跑一次确认结果一致。

相关文件：

- Calibration script: [benchmark_context_reskip_calibrate.py](/home/user01/Minko/reskip2/reskip/experiments/benchmark_context_reskip_calibrate.py)
- Calibration JSON: [benchmark_context_analysis.json](/home/user01/Minko/reskip2/reskip/outputs/reskip_benchmark_context_analysis_340M_20260503/benchmark_context_analysis.json)
- Selected model: [pos5_q080_M1](/home/user01/Minko/reskip2/reskip/outputs/reskip_340M_benchmark_context_20260503/pos5_q080_M1)
- Full eval rerun summary: [final_summary.json](/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_benchmark_context_rerun_20260503_113945/final_summary.json)

## B. Evaluation Setup

模型：

- Full-depth baseline: `/home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer-340M`
- Dynamic ReSkip candidate root: `/home/user01/Minko/reskip2/reskip/outputs/reskip_340M_benchmark_context_20260503`

Tasks:

- `lambada_openai`
- `hellaswag`
- `arc_easy`
- `arc_challenge`
- `piqa`
- `mmlu`
- `openbookqa`

lm-eval 设置：

- `batch_size=8`
- `dtype=bfloat16`
- `--collect_skip_stats` enabled
- calibration: `limit=512`
- sanity: `limit=256`
- final eval: no `limit`
- GPUs: final rerun split across GPU 0-3

Skip stats 定义：

- `skip_events`: 实际发生的 block skip 次数。
- `total_positions`: `forwards_with_trace * num_block_groups`，本模型为 8 个 block groups。
- `avg_blocks`: `executed_blocks_sum / forwards_with_trace`。
- `block_skip_rate`: `skip_events / total_positions`。
- `forward_skip_rate`: `skip_events / forwards_with_trace`；当 `max_skips=1` 时也等价于有 skip 的 forward 占比。

## C. 旧 q=0.85 结果复核

旧配置：

| strategy | probe | P | q | max_skips | 结论 |
|---|---|---:|---:|---:|---|
| `recent_weight_gt` | `attn_only` | `{3,5}` | 0.85 | 2 | 0-trigger，benchmark 分数不能作为有效 ReSkip 结果 |

7-task 复核：

| 配置 | skip_events / forwards | avg_blocks | 结论 |
|---|---:|---:|---|
| full-depth | 0 / 9727 | 8.000 | baseline |
| old q085 P={3,5} | 0 / 9727 | 8.000 | 与 full-depth 完全一致是因为没有触发 skip |

因此，旧 q085 结果只保留为负例：FineWeb 长上下文阈值不能直接迁移到 benchmark prompt/scoring 分布。

## D. FineWeb-Edu Analysis 候选与迁移失败

重新跑 FineWeb-Edu analysis 时，将 q 范围从旧的高分位收窄到：

`q ∈ {0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75}`

FineWeb analysis 的主要候选：

| 候选 | probe | position mode | P | q | max_skips | FineWeb PPL | FineWeb avg_blocks |
|---|---|---|---:|---:|---:|---:|---:|
| best quality | `attn_only` | `ablation3` | `{1,3,5}` | 0.45 | 1 | 313.226 | 7.031 |
| best tolerated/skip | `attn_only` | `ablation3` | `{1,3,5}` | 0.45 | 2 | 330.048 | 6.219 |
| recommended | `first_attn` | `recommended` | `{4,5,6}` | 0.70 | 1 | 447.837 | 7.500 |

将这些候选导出后，在 benchmark limit-256 sanity 上验证：

| 配置 | P | q / threshold | skip_events / positions | avg_blocks | LAMBADA acc | LAMBADA ppl | PIQA acc_norm | MMLU acc | 处理 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| full-depth | - | - | 0 / 14888 | 8.000 | 0.3828 | 19.655 | 0.7031 | 0.2573 | baseline |
| FineWeb tolerated | `{1,3,5}` | q=0.45 | 2170 / 14888 | 6.834 | 0.1133 | 527.964 | 0.5352 | 0.2408 | reject: early skip 破坏严重 |
| FineWeb quality | `{1,3,5}` | q=0.45 | 1151 / 14888 | 7.382 | 0.1523 | 213.912 | 0.5742 | 0.2410 | reject: early skip 破坏严重 |
| FineWeb recommended | `{4,5,6}` | q=0.70 | 43 / 14888 | 7.977 | 0.3711 | 20.079 | 0.7031 | 0.2590 | reject: trigger 太少 |
| FineWeb low1 q045 | `{5}` | threshold=0.4349 | 0 / 14888 | 8.000 | 0.3828 | 19.655 | 0.7031 | 0.2573 | reject: 0-trigger |
| FineWeb low1 q050 | `{5}` | threshold=0.4362 | 0 / 14888 | 8.000 | 0.3828 | 19.655 | 0.7031 | 0.2573 | reject: 0-trigger |
| FineWeb low2 q045 | `{5,6}` | thresholds 0.4349 / 0.3613 | 0 / 14888 | 8.000 | 0.3828 | 19.655 | 0.7031 | 0.2573 | reject: 0-trigger |

结论：

- `P={1,3,5}` 虽然在 FineWeb analysis 上有较大 skip，但 benchmark 上的 early position skip 明显破坏 LAMBADA。
- `P={4,5,6}` 较稳但触发率太低。
- FineWeb 给出的 `P={5}` 阈值约 `0.435-0.436`，在 benchmark 上仍为 0-trigger。
- 因此必须在 benchmark context 上重新校准 q，而不是直接复用 FineWeb 阈值。

## E. Benchmark-Context Calibration

校准方式：

- 使用 full-depth model 在 `lm-eval` request contexts 上 forward。
- 强制返回 `routing_info`，但不执行 skip。
- 收集 `recent_weight_gt + attn_only` 的 phase-1 metric。
- 使用 `limit=512` 覆盖 7 个 benchmark。
- 从 late candidate 中选 `P={5}` 做 q 校准，原因是：
  - early `P={1,3}` 在 sanity 中明显损伤质量；
  - FineWeb 的 low-position 经验表明 position 5 有一定可跳空间；
  - 单点 P={5} 能把 skip 风险局部化，便于解释和复核；
  - position 5 在 benchmark-context q80 下可以非零触发，同时 limit-256 质量损失最小。

Position 5 的 benchmark-context metric 分布：

| count | mean | min | p10 | p25 | p50 | p75 | p80 | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2665 | 0.3943 | 0.3118 | 0.3688 | 0.3874 | 0.3971 | 0.4069 | 0.4095 | 0.4141 | 0.4264 |

导出的 q 候选：

| P | q | threshold at pos5 | max_skips | 导出目录 |
|---:|---:|---:|---:|---|
| `{5}` | 0.50 | 0.397135 | 1 | `pos5_q050_M1` |
| `{5}` | 0.55 | 0.398438 | 1 | `pos5_q055_M1` |
| `{5}` | 0.60 | 0.400391 | 1 | `pos5_q060_M1` |
| `{5}` | 0.65 | 0.402344 | 1 | `pos5_q065_M1` |
| `{5}` | 0.70 | 0.404948 | 1 | `pos5_q070_M1` |
| `{5}` | 0.75 | 0.406901 | 1 | `pos5_q075_M1` |
| `{5}` | 0.80 | 0.409505 | 1 | `pos5_q080_M1` |

注意：q50/q60/q70/q80 做了 limit-256 sanity；q55/q65/q75 作为插值候选导出但未进入正式 sanity 表。

## F. q Candidate Sanity Eval

Sanity 设置：

- 7 tasks
- `limit=256`
- `batch_size=8`
- `max_skips=1`
- all skips restricted to `P={5}`

| 配置 | q | threshold | skip_events / positions | avg_blocks | LAMBADA acc | LAMBADA ppl | HellaSwag acc_norm | ARC-E acc_norm | ARC-C acc_norm | PIQA acc_norm | MMLU acc | OBQA acc_norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full-depth | - | - | 0 / 14888 | 8.000 | 0.3828 | 19.655 | 0.4922 | 0.5547 | 0.3008 | 0.7031 | 0.2573 | 0.3789 |
| pos5 q50 | 0.50 | 0.397135 | 1114 / 14888 | 7.401 | 0.3125 | 44.213 | 0.4844 | 0.5508 | 0.3047 | 0.6836 | 0.2339 | 0.3750 |
| pos5 q60 | 0.60 | 0.400391 | 865 / 14888 | 7.535 | 0.3438 | 31.104 | 0.4766 | 0.5547 | 0.3047 | 0.7031 | 0.2365 | 0.3789 |
| pos5 q70 | 0.70 | 0.404948 | 656 / 14888 | 7.648 | 0.3711 | 22.584 | 0.4922 | 0.5547 | 0.3008 | 0.7031 | 0.2389 | 0.3789 |
| pos5 q80 | 0.80 | 0.409505 | 380 / 14888 | 7.796 | 0.3789 | 20.731 | 0.4922 | 0.5547 | 0.3008 | 0.6953 | 0.2461 | 0.3789 |

Candidate decision:

- q50/q60 触发率较高，但 LAMBADA 和 MMLU 损失明显。
- q70 在 LAMBADA 上接近 baseline，但 MMLU 仍低。
- q80 是 sanity 中最稳的非零触发点，保留少量 skip，同时把 LAMBADA/ARC/HellaSwag/OpenBookQA 基本保持住。
- 因此正式 full eval 选择 `P={5}, q=0.80, threshold=0.409505, max_skips=1`。

## G. Final Full Eval

最终策略：

```text
strategy = recent_weight_gt
probe_mode = attn_only
P = {5}
q = 0.80
thresholds = [1e9, 1e9, 1e9, 1e9, 1e9, 0.4095052182674408, 1e9, 1e9]
max_skips = 1
```

完整 7-task eval 结果：

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

完整 7-task skip stats：

| 配置 | forwards_with_trace | total_positions | skip_events | avg_blocks | block_skip_rate | forward_skip_rate | per_position_skips |
|---|---:|---:|---:|---:|---:|---:|---|
| full-depth | 9727 | 77816 | 0 | 8.000 | 0.0000 | 0.0000 | `[0,0,0,0,0,0,0,0]` |
| ReSkip pos5 q80 | 9727 | 77816 | 706 | 7.927 | 0.0091 | 0.0726 | `[0,0,0,0,0,706,0,0]` |

复跑验证：

- First full eval summary: `/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_benchmark_context_20260503/final_summary.json`
- Rerun summary: `/home/user01/Minko/reskip2/reskip/outputs/lm_eval_reskip_340M_benchmark_context_rerun_20260503_113945/final_summary.json`
- 两次正式 7-task eval 的指标和 skip stats 一致。

## H. Selection Strategy for Appendix

最终选择逻辑可以概括为：

1. **必须验证非零触发**：旧 q085 被排除，因为 `skip_events=0`。
2. **先排除 early destructive P**：`P={1,3,5}` 在 benchmark sanity 中导致 LAMBADA 大幅下降，说明 FineWeb 上的 PPL proxy 不能直接保证 benchmark 质量。
3. **再排除 0-trigger P/q**：FineWeb low1/low2 的 `P={5}` 或 `P={5,6}` 阈值过高，在 benchmark 上不触发。
4. **用 benchmark-context 重新估 q**：在真实 lm-eval request context 上收集 metric，得到 pos5 阈值从 FineWeb 的约 `0.435` 降到 benchmark-context 的 `0.397-0.410`。
5. **用 limit-256 sanity 选 q**：q50/q60/q70 触发更多但损失更大；q80 是最稳的非零触发点。
6. **正式结果不包装成显著加速或零退化**：q80 有效触发，但 `avg_blocks 8.000 -> 7.927`，省算幅度约 0.9% block-level；MMLU 下降约 1.2 个点。

最终结论：

`P={5}, q=0.80, max_skips=1` 是当前 340M benchmark-context 复核中最稳的有效动态 ReSkip 配置，但它是“有效触发、收益有限、有 MMLU 损失”的结果，不应作为“显著加速且零退化”的主结论。
