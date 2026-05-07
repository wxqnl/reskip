# Reskip_5_7_3.pdf 定稿数据对齐记录

本文件是项目内对 `Reskip_5_7_3.pdf` 的数据真值表。后续 README、paper source、appendix 和中文汇总都以这里为准；旧实验日志只作为原始记录，不再作为 paper-facing 数值来源。

## 1. 题目与核心 claims

- 定稿题目：**AR-Retrofit: Retrofitting Pretrained Decoders with Attention Residuals**
- Qwen3-VL-2B：新增/训练参数约 **7.4M**，小于 **0.4%**。
- Qwen3-VL-4B：新增/训练参数约 **11.8M**，小于 **0.3%**。
- 2B / 4B VLM 六项平均提升分别为 **+2.1 / +1.2 pp**。
- LAMBADA 提升分别为 **+3.3 / +8.7 pp**。
- LIBERO VLA full retrofit 相对 matched OFT baseline 提升分别为 **+1.3 / +0.7 pp**。

## 2. Table 1：340M from-scratch

| Model | LAMBADA acc | LAMBADA ppl | HellaSwag | PIQA | OpenBookQA | Latency vs base |
|---|---:|---:|---:|---:|---:|---:|
| Base transformer | 37.9 | 24.7 | 44.4 | 67.8 | 33.2 | 1.0x |
| AttnRes full | 40.5 | 20.2 | 46.1 | 68.9 | 35.8 | 1.8x |
| AttnRes + ReSkip | 40.5 | 20.2 | 46.1 | 68.9 | 35.8 | 1.1x |

## 3. Table 2：Qwen3-VL 主结果

| Scale / Method | Lamb. | HSwag | MMB | MMMU | MMStar | AI2D | OCR | RWQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B Base (LoRA) | 53.2 | 50.6 | 75.8 | 41.4 | 53.6 | 73.6 | 77.2 | 64.8 |
| 2B + AR-Retrofit | 56.5 | 50.0 | 78.9 | 43.2 | 53.6 | 75.8 | 81.4 | 66.1 |
| 2B Δ | +3.3 | -0.6 | +3.1 | +1.8 | 0.0 | +2.2 | +4.2 | +1.3 |
| 2B + ReSkip | 56.1 | 49.2 | 78.9 | 43.0 | 54.9 | 75.9 | 80.1 | 66.0 |
| 4B Base (LoRA) | 57.6 | 56.2 | 83.3 | 49.0 | 62.4 | 81.9 | 81.9 | 71.5 |
| 4B + AR-Retrofit | 66.3 | 55.2 | 85.2 | 52.1 | 63.2 | 82.5 | 82.4 | 71.8 |
| 4B Δ | +8.7 | -1.0 | +1.9 | +3.1 | +0.8 | +0.6 | +0.5 | +0.3 |
| 4B + ReSkip | 63.3 | 54.9 | 85.0 | 51.4 | 62.1 | 82.5 | 82.5 | 70.7 |

## 4. Table 3：skip baseline 对比，Qwen3-VL-2B

Decode 为 512 输入 / 4096 输出、StaticCache、tokens/s。

| Method | Lamb.-500 | HSwag-500 | AI2D | MMMU | MMStar | Decode |
|---|---:|---:|---:|---:|---:|---:|
| Static block skip | 37.3 | 53.8 | 58.0 | 33.8 | 33.6 | 73.1 |
| Gromov-4 | 6.9 | 44.4 | 57.3 | 35.6 | 49.1 | 78.3 |
| Gromov-8 | 2.0 | 36.5 | 6.8 | 23.9 | 1.9 | 94.6 |
| MoD | 47.4 | 56.1 | 34.6 | 33.4 | 22.0 | 71.2 |
| LayerSkip | 14.6 | 46.0 | 37.9 | 31.1 | 44.6 | 80.4 |
| Base (LoRA) | 53.3 | 55.2 | 73.6 | 41.4 | 53.6 | 69.9 |
| AR-Retrofit full | 56.4 | 58.7 | 75.8 | 43.2 | 53.6 | 57.2 |
| ReSkip | 56.2 | 57.9 | 75.9 | 43.0 | 54.9 | 67.5 |

## 5. Table 4：LIBERO 4-suite

| Scale | Policy | Spatial | Object | Goal | Long-10 | Avg | Δ vs Base |
|---|---|---:|---:|---:|---:|---:|---:|
| 2B | Base | 94.8 | 99.8 | 97.4 | 91.4 | 95.9 | -- |
| 2B | Full | 97.8 | 99.6 | 98.6 | 92.6 | 97.2 | +1.3 |
| 2B | ReSkip | 97.6 | 99.2 | 99.0 | 93.6 | 97.4 | +1.5 |
| 4B | Base | 95.0 | 99.2 | 97.8 | 92.2 | 96.1 | -- |
| 4B | Full | 94.6 | 99.8 | 98.2 | 94.2 | 96.7 | +0.7 |
| 4B | ReSkip | 96.4 | 98.2 | 98.4 | 92.8 | 96.5 | +0.4 |

## 6. Table 5：ReSkip 参数选择消融

VLM 列为 LAMBADA-500 accuracy，VLA 列为 LIBERO 4-suite avg success rate，单位均为 `%`。括号内 delta 对应 no-skip/full reference：VLM 2B/4B 为 56.4/66.3，VLA 2B/4B 为 97.2/96.7。

| Knob | Setting | VLM-2B | VLM-4B | VLA-2B | VLA-4B | Decision |
|---|---|---:|---:|---:|---:|---|
| P,M | main per-domain setting | 56.2 (-0.2) | 61.6 (-4.7) | 97.4 (+0.2) | 96.5 (-0.2) | chosen |
| P,M | main cap or wider cap | 56.0 (-0.5) | 61.6 (-4.7) | 97.2 (+0.0) | 96.3 (-0.4) | no gain |
| P,M | all, M=4 | 45.4 (-11.1) | 54.2 (-12.0) | 96.3 (-0.9) | 89.5 (-7.2) | reject |
| q | M=0 | 57.0 | 66.3 | 97.2 | 96.7 | reference |
| q | main per-domain q | 56.2 (-1.0) | 61.6 (-4.7) | 97.4 (+0.2) | 96.5 (-0.2) | chosen |
| q | aggressive stress test | 41.2 (-15.8) | 49.2 (-17.1) | 83.1 (-14.1) | 94.3 (-2.4) | reject |

## 7. Figure 2 calibration 可视化结果

- routing heatmaps 展示 easy/hard 样本 skip pattern 不同；
- average skipped blocks 随 full-depth uncertainty 增大而下降；
- margin-preservation panel 中 **83.3%** decisions 满足 `2 epsilon <= Delta`，token-level prediction preserved 为 **96.2%**。

## 8. Appendix 表格关键数据

### Table 6：recipe-selection ablation

| Name | gamma | rank | steps | VLM% | ramp | LAMBADA | HSwag | MMB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-VL-2B | -- | -- | -- | -- | -- | 0.532 | 0.506 | 0.727 |
| gamma-free (A) | gamma≈0 trainable | 128 | 10k | 50 | -- | 0.576 | 0.512 | 0.720 |
| Selection-stage winner | 0->1 fast | 256 | 5k | 50 | 0.3 | 0.576 | 0.522 | 0.710 |
| + low rank | 0->1 fast | 64 | 10k | 50 | 0.3 | 0.570 | 0.530 | 0.697 |
| + very low rank | 0->1 fast | 32 | 10k | 50 | 0.3 | 0.568 | 0.526 | 0.683 |
| + low rank, VLM-heavy | 0->1 fast | 64 | 10k | 80 | 0.3 | 0.560 | 0.524 | 0.660 |
| + mid rank, VLM-heavy | 0->1 fast | 128 | 10k | 80 | 0.3 | 0.564 | 0.520 | 0.653 |
| + VLM-heavy | 0->1 fast | 256 | 10k | 80 | 0.3 | 0.560 | 0.520 | 0.617 |
| + longer,10k | 0->1 fast | 256 | 10k | 50 | 0.3 | 0.586 | -- | 0.587 |
| + slow ramp | 0->1 slow | 256 | 10k | 50 | 0.7 | 0.568 | 0.520 | 0.517 |
| + longer,20k | 0->1 fast | 256 | 20k | 50 | 0.3 | 0.568 | 0.520 | 0.513 |

### Table 7：data-mix screening

| Scale / Mix | AI2D | MMBench | MMMU | MMStar | OCRBench | RealWorldQA |
|---|---:|---:|---:|---:|---:|---:|
| 2B base | 0.736 | 75.77 | 0.414 | 0.536 | 0.772 | 0.648 |
| 2B v1 5k | 0.684 | 72.85 | 0.406 | 0.386 | 0.795 | 0.447 |
| 2B v1 10k | 0.677 | 73.71 | 0.404 | 0.471 | 0.801 | 0.642 |
| 2B v2 5k | 0.283 | 73.80 | 0.421 | 0.422 | 0.806 | 0.512 |
| 4B base | 0.819 | 83.33 | 0.490 | 0.624 | 0.819 | 0.715 |
| 4B v1 5k | 0.810 | 83.76 | 0.510 | 0.579 | 0.812 | 0.707 |
| 4B v1 10k | 0.580 | 81.79 | 0.432 | 0.437 | 0.812 | 0.689 |
| 4B v2 5k | 0.603 | 81.87 | 0.477 | 0.333 | 0.808 | 0.686 |

### Table 8：LAMBADA-500 ReSkip Pareto

| q | LAMBADA acc | ppl | Δ acc | avg skips |
|---|---:|---:|---:|---:|
| M=0 | 0.5700 | 4.526 | -- | 0.00/≤7 |
| 0.85 | 0.5600 | 5.258 | -1.0 | 0.19/2 |
| 0.50 | 0.4120 | 12.550 | -15.8 | 1.06/2 |
| 0.30 | 0.3900 | 14.005 | -18.0 | 1.17/2 |

### VLA ReSkip calibration

- 2B action-drift MSE sorted `(x10^-3)`：block1 0.4，block4 34.0，block0 58.1，block5 87.0，剩余 >100；`P={1,4}`。
- 4B action-drift MSE sorted `(x10^-3)`：block1 0.6，block2 11.0，block0 34.0，block3 63.0，剩余 >120；`P={1,2}`。
- threshold records：2B 为 31,286，4B 为 31,257。
- cross-scale P transfer：2B `P={1,4}` 直接迁移到 4B，在 `q=0.85` 下 LIBERO-Spatial 下降到 12.4%。
- LAMBADA-calibrated `q=0.85` 直接用于 LIBERO-Spatial，会从 99.5% 降到 64%。
