# 论文数字唯一口径

更新时间：2026-09-12。除非完成新的正式实验并同步修改本文件，否则不要从其他旧 Markdown 抄数。

## 1. VLM 主表：六任务完整评测

多 seed 行为算术均值；`±` 为训练 seed 间样本标准差。Base 只有一个固定模型，不报告伪方差。

| Model | System | Seeds | AI2D | MMBench | MMMU | MMStar | OCRBench | RealWorldQA | Macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B | Base | 1 | 75.648 | 76.203 | 42.333 | 53.754 | 80.700 | 65.229 | 65.644 |
| Qwen3-VL-2B | Full AttnRes | 3 | 76.231 | 77.950 | 45.000 | 54.397 | 84.267 | 64.880 | 67.121 ± 0.394 |
| Qwen3-VL-2B | Full RSM-AttnRes | 3 | 76.738 | 77.749 | 45.444 | 55.773 | 84.000 | 66.187 | **67.649 ± 0.466** |
| Qwen3-VL-2B | RSM-ReSkip | 1 (seed 0) | 76.911 | 78.694 | 45.889 | 56.049 | 82.200 | 66.536 | 67.713 |
| Qwen3-VL-4B | Base | 1 | 81.801 | 83.333 | 50.444 | 61.790 | 82.600 | 71.634 | 71.934 |
| Qwen3-VL-4B | Full AttnRes | 3 | 82.103 | 85.137 | 52.222 | 61.079 | 84.867 | 71.634 | 72.840 ± 0.354 |
| Qwen3-VL-4B | Full RSM-AttnRes | 3 | 82.599 | 84.107 | 51.481 | 62.316 | 85.467 | 72.505 | **73.079 ± 0.194** |
| Qwen3-VL-4B | RSM-ReSkip | 1 (seed 2) | 82.804 | 84.021 | 51.222 | 62.370 | 84.800 | 72.810 | 73.005 |
| SmolVLM2-2.2B | Base | 1 | 66.354 | 68.041 | 35.778 | 44.548 | 60.200 | 58.431 | 55.559 |
| SmolVLM2-2.2B | Full AttnRes | 3 | 66.192 | 67.669 | 35.407 | 43.353 | 61.233 | 57.386 | 55.207 ± 0.340 |
| SmolVLM2-2.2B | Full RSM-AttnRes | 3 | 66.267 | 68.013 | 36.000 | 43.763 | 61.067 | 58.388 | **55.583 ± 0.236** |
| InternVL3.5-2B | Base | 1 | 75.907 | 77.148 | 49.333 | 59.327 | 83.100 | 60.654 | 67.578 |
| InternVL3.5-2B | Full AttnRes | 1 | 75.518 | 76.460 | 49.667 | 59.140 | 83.300 | 61.569 | 67.609 |
| InternVL3.5-2B | Full RSM-AttnRes | 1 | 75.518 | 76.460 | 49.778 | 59.230 | 82.700 | 60.915 | 67.434 |

Matched-seed 的 `RSM − Full AttnRes`：Qwen3-VL-2B `+0.528 ± 0.082 pp`；Qwen3-VL-4B `+0.239 ± 0.159 pp`；SmolVLM2 `+0.376 ± 0.116 pp`；InternVL3.5 单 seed `−0.176 pp`。

ReSkip 行是冻结 operating point 的单 seed 结果，不应与三 seed Full 均值直接计算质量损失；对应的 matched-seed 对比见下一节。

## 2. ReSkip 正式质量点

ReSkip 为单个冻结 confirmatory seed，必须与对应 seed 的 Full RSM 比，而不是与三 seed 均值比。

| Model | Matched seed | Full RSM | RSM-ReSkip | Δ vs Full | Removed block-eq/token | Gate |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-VL-2B | 0 | 67.970 | 67.713 | −0.257 pp | 0.959 | 通过 `Δ ≥ −0.30 pp` |
| Qwen3-VL-4B | 2 | 73.255 | 73.005 | −0.251 pp | 0.767 | 通过 `Δ ≥ −0.30 pp` |

下一档均失败：2B `1.019 block-eq/token, −0.309 pp`；4B `1.089 block-eq/token, −0.512 pp`。不要把 2B 写成“安全削减 1.02 个 block”。

## 3. H100 固定 64-token 速度

真实六任务图片和 prompt，batch size 1，每任务 12 个请求、三重复；速度比为逐请求配对几何均值，区间为 10,000 次 bootstrap。它是 decode-dominant 端到端测速，不是 accuracy benchmark。

| Model | Safe target | Block-eq/token | ReSkip/Base | 95% CI | ReSkip/Full RSM | 95% CI | Matched n |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B | 0.45 | 0.959 | **1.0438x** | [1.0357, 1.0532] | **1.1752x** | [1.1628, 1.1897] | 72 |
| Qwen3-VL-4B | 0.25 | 0.767 | **1.0286x** | [1.0227, 1.0363] | **1.0715x** | [1.0670, 1.0762] | 72 |

## 4. Natural-generation 速度

仅比较输出 token 数一致的请求。该表必须与固定 64-token 表并列或放附录，不能只挑更好看的固定长度结果。

| Model | ReSkip/Base | 95% CI | Matched n | ReSkip/Full RSM | 95% CI | Matched n |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B | 1.0055x | [1.0027, 1.0084] | 205 | 1.0927x | [1.0884, 1.0970] | 315 |
| Qwen3-VL-4B | 0.9950x | [0.9906, 0.9994] | 38 | 1.0372x | [1.0328, 1.0417] | 51 |

因此准确结论是：safe ReSkip 在固定 64-token decode 上超过 Base；自然生成时 2B 仅小幅超过 Base，4B 略低于 Base，但两者都明显快于 Full RSM。

## 5. LIBERO VLA：已完成但方向混合

每个 suite 为 500 episodes；Macro 是四个 suite 的非加权平均。

| Scale | System | Spatial | Object | Goal | LIBERO-10 | Macro | Δ vs Base |
|---|---|---:|---:|---:|---:|---:|---:|
| 2B | Base | 93.4 | 96.6 | 93.4 | 67.0 | 87.60 | — |
| 2B | AttnRes | 94.8 | 97.8 | 90.8 | 68.8 | 88.05 | +0.45 pp |
| 2B | RSM v2 | 96.4 | 98.0 | 93.8 | 56.0 | 86.05 | −1.55 pp |
| 4B | Base | 95.4 | 97.8 | 98.0 | 65.8 | 89.25 | — |
| 4B | AttnRes | 95.2 | 95.0 | 95.6 | 58.4 | 86.05 | −3.20 pp |
| 4B | RSM v2 | 97.2 | 99.4 | 92.8 | 71.2 | **90.15** | +0.90 pp |
| 2B | Path-C RSM (auxiliary) | 95.0 | 100.0 | 95.4 | 74.2 | 91.15 | +3.55 pp |

Path-C 在 protocol 中标记为 auxiliary，不能无说明替换 2B primary 行。VLA 当前只支持“4B primary RSM 有提升、2B 结果依赖训练路径”的边界结论，不支持跨规模稳定提升。

## 6. 七项纯文本 LM benchmark

2026-09-11 完成，0-shot；除 SmolVLM2 的 AttnRes/RSM 为三训练 seed 外，其余行为单 checkpoint。Average 为表中七项 accuracy 的算术平均；该组结果是补充能力验证，不替代 VLM 主表。

| Family | Method | n | LAMBADA | HellaSwag | PIQA | ARC-E | ARC-C | OpenBookQA | WinoGrande | Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B | Base | 1 | 52.49 | 61.40 | 73.94 | 72.26 | 47.18 | 38.40 | 62.27 | 58.28 |
| Qwen3-VL-2B | Full AttnRes | 1 | 59.79 | 62.71 | 75.08 | 78.37 | 49.57 | 38.20 | 62.35 | 60.87 |
| Qwen3-VL-2B | Full RSM-AttnRes | 1 | 57.73 | 63.16 | 74.97 | 79.00 | 50.85 | 38.20 | 63.54 | 61.07 |
| Qwen3-VL-4B | Base | 1 | 60.94 | 69.85 | 77.09 | 78.62 | 55.72 | 40.40 | 66.77 | 64.20 |
| Qwen3-VL-4B | Full AttnRes | 1 | 64.00 | 71.22 | 77.69 | 82.91 | 58.96 | 40.20 | 67.88 | 66.12 |
| Qwen3-VL-4B | Full RSM-AttnRes | 1 | 62.80 | 71.44 | 78.18 | 83.38 | 58.96 | 41.60 | 66.93 | 66.18 |
| InternVL3.5-2B | Base | 1 | 57.09 | 65.32 | 73.88 | 75.25 | 48.72 | 37.00 | 60.62 | 59.70 |
| InternVL3.5-2B | Full AttnRes | 1 | 56.12 | 64.97 | 73.88 | 77.74 | 49.57 | 37.80 | 60.69 | 60.11 |
| InternVL3.5-2B | Full RSM-AttnRes | 1 | 56.06 | 64.91 | 73.39 | 77.23 | 49.49 | 37.40 | 60.85 | 59.91 |
| SmolVLM2-2.2B | Base | 1 | 65.67 | 67.68 | 74.27 | 69.57 | 44.11 | 40.80 | 65.43 | 61.08 |
| SmolVLM2-2.2B | Full AttnRes | 3 | 65.50 | 67.49 | 74.43 | 69.53 | 43.97 | 41.07 | 65.77 | 61.11 ± 0.16 |
| SmolVLM2-2.2B | Full RSM-AttnRes | 3 | 65.46 | 67.60 | 74.39 | 69.65 | 44.08 | 40.67 | 65.59 | 61.06 ± 0.27 |

## 7. 写作边界

- 可写：RSM 改善 Qwen 2B/4B 和 SmolVLM2 的 matched Full AttnRes；ReSkip 是冻结后的 token-level 推理期附加收益。
- 可写：safe policy 在固定 64-token 测速上相对 Base 获得 `4.38%`（2B）和 `2.86%`（4B）配对几何均值加速。
- 不可写：RSM 在所有 VLM family 上都提升；RSM 普遍提高 skip-risk classifier；自然生成在所有规模上都超过 Base；VLA 跨规模一致提升。
- 不可复用：早期并发测速、单任务 12-request 的 2B `1.0707x`、失败的 aggressive 4B policy、任何超出注册质量带的 Pareto 点。
