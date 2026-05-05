# ReSkip Adaptive-Depth Figure 打包说明

这份说明用于把 Figure X 交给别人继续做视觉优化。重点是：图可以美化，但数据口径和结论边界不要改。

## 1. 图的目标

老师建议的目标是做一个 3-panel figure，把 ReSkip 的 routing、input difficulty 和 margin-preservation rationale 连起来：

1. **Input-dependent routing**：不同 VLM 输入有不同 routing / skip pattern。
2. **Hard steps keep more depth**：full-depth 不确定性更高的 decode step 更少触发 skip。
3. **Margin-preservation**：多数 ReSkip decision 落在 `2epsilon <= Delta` 区域，因此 prediction 通常保持不变。

当前可用 caption：

> ReSkip adapts depth according to input difficulty. (a) Representative VLM samples follow different routing patterns; red boxes denote skipped blocks. (b) Average skipped blocks decrease on high-uncertainty decode steps. (c) Most ReSkip decisions lie below the margin-preservation boundary `2epsilon = Delta`, explaining why predictions are usually preserved.

## 2. 当前正式图

主图文件：

- `paper/figures/reskip_adaptive_depth_vlm.pdf`
- `paper/figures/reskip_adaptive_depth_vlm.png`

同步保留的 token-level 版本：

- `paper/figures/reskip_adaptive_depth_vlm_token.pdf`
- `paper/figures/reskip_adaptive_depth_vlm_token.png`

当前两组文件内容相同，主图 `reskip_adaptive_depth_vlm.*` 已经复制为 token-level diagnostic 的最终版本。

## 3. 数据与实验口径

这张图不是从旧 benchmark 表拼出来的，而是重新跑的 token-level decode probe。

| Item | Value |
|---|---|
| Base model | Qwen3-VL-2B |
| Model path | `/home/user01/Minko/models/Qwen3-VL-2B` |
| Retrofit state | `retrofit/outputs/H_2B_r256_10k_L4_v3/retrofit_attnres_state.pt` |
| Dataset | MMBench dev |
| Probe size | 160 samples |
| Decode horizon | 8 full-depth greedy decode tokens / sample |
| Total token points | 1280 cached decode forwards |
| ReSkip diagnostic config | `P={2}, tau=0.512, M=1` |
| Dynamic strategy | `recent_weight_gt` |
| Margin | full-depth token logit `top1 - top2` |
| Perturbation | `epsilon = max` logit shift over full-depth top-10 candidate tokens |

注意：这个 operating point 是 **mechanism / calibration diagnostic point**，不是替代主 benchmark 表的 target-calibrated VLM operating point。

## 4. 关键数值

来自：

- `retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160/summary.json`

| Metric | Value |
|---|---:|
| Samples | 160 |
| Token points | 1280 |
| Prediction preserved | 0.9617 |
| Avg skipped blocks / decode forward | 0.1008 |
| Margin condition `2epsilon <= Delta` | 0.8328 |
| Spearman rho(uncertainty, skip) | -0.1421 |

Difficulty-bin skip：

| Full-depth uncertainty bin | n token points | mean margin | mean skipped blocks / forward |
|---|---:|---:|---:|
| Low / easy | 634 | 6.7395 | 0.1372 |
| High / hard | 646 | 1.1099 | 0.0650 |

解释：

- Easy / low-uncertainty token steps skip 更多。
- Hard / high-uncertainty token steps skip 更少。
- `83.28%` token decisions 满足 margin sufficient condition。
- `96.17%` token predictions preserved。

## 5. 三个 panel 的数据定义

### Panel (a): Routing Heatmap

数据来源：

- `sample_records.jsonl`

字段：

- `w_recent_mean`：该 sample 的各 block 平均 recent-source routing weight。
- `skipped_blocks_any`：该 sample 的 decode horizon 内实际触发过 skip 的 block。
- `mean_margin_delta`：该 sample 的平均 full-depth token margin。
- `mean_skipped_blocks`：该 sample 的平均 skip count。

画法：

- 行：代表性 easy / hard samples。
- 列：blocks `B0` 到 `B6`。
- 颜色：`w_recent_mean`。
- 红框：`skipped_blocks_any`。

当前图里红框主要在 `B2`，因为 diagnostic config 是 `P={2}`。

### Panel (b): Difficulty vs Skip

数据来源：

- `token_records.jsonl`

字段：

- `margin_delta`
- `skip_count`

定义：

- full-depth uncertainty 用 margin 反向表示。
- easy = high margin。
- hard = low margin。
- 当前采用二分组，阈值是 median margin `3.0`。

结果：

- Easy bin skip：`0.1372`
- Hard bin skip：`0.0650`

这支持 “harder steps keep more depth”。

### Panel (c): Margin-Preservation Region

数据来源：

- `token_records.jsonl`

字段：

- `margin_delta`
- `two_epsilon`
- `prediction_preserved`
- `margin_preservation_condition`

定义：

- 横轴：`Delta(x) = full top1 logit - full top2 logit`
- 纵轴：`2epsilon(x)`
- `epsilon` 是 full-depth top-10 candidate token logits 上的最大 ReSkip perturbation。
- 虚线：`2epsilon = Delta`
- 颜色 / marker：prediction preserved vs changed。

结果：

- `83.28%` below boundary。
- `96.17%` prediction preserved。

## 6. 复现实验命令

在 repo root `/home/user01/Minko/reskip2/reskip` 下运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/user01/Minko/reskip2/reskip/retrofit \
/home/user01/Minko/reskip2/.venv/bin/python \
retrofit/analysis/reskip_token_margin_figure.py \
  --n 160 \
  --progress-every 80 \
  --threshold 0.512 \
  --eligible-block 2 \
  --max-skips 1 \
  --decode-tokens 8 \
  --top-k 10 \
  --difficulty-bins 2 \
  --out-dir retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160
```

脚本会输出：

- `paper/figures/reskip_adaptive_depth_vlm_token.pdf`
- `paper/figures/reskip_adaptive_depth_vlm_token.png`
- `retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160/token_records.jsonl`
- `retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160/sample_records.jsonl`
- `retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160/summary.json`

如果要覆盖主图文件：

```bash
cp paper/figures/reskip_adaptive_depth_vlm_token.pdf paper/figures/reskip_adaptive_depth_vlm.pdf
cp paper/figures/reskip_adaptive_depth_vlm_token.png paper/figures/reskip_adaptive_depth_vlm.png
```

## 7. 为什么不用之前那版 option-level 图

之前跑过 option-level scoring probe：

| Probe | Result | Why rejected |
|---|---|---|
| 4B canonical `q=0.85, P={1,2}`, MMBench option scoring, n=160 | preserved `0.9062`, condition `0.1625`, skip `0.3250/fwd` | answer-option likelihood 跨多个 tokens 累积扰动，`epsilon` 过保守，不能支撑 margin-preservation 图 |
| 2B target-calib `q=0.50, P={4}`, MMBench option scoring, n=64 | preserved `0.7969`, condition `0.3906`, skip `0.5358/fwd` | block 4 的 routing/margin 相关性方向不适合画 hard-retains-depth |

最终改成 token-level decode probe，是因为 ReSkip 的 skip decision 本来就在 cached forward / token 粒度发生；这个粒度和理论条件更一致。

## 8. 给视觉优化者的建议

可以优化：

- 字体大小、字号一致性、坐标轴间距。
- Panel (a) 的 row label 可以缩短，避免拥挤。
- Panel (b) 可以把 easy / hard 的柱子颜色做得更清楚。
- Panel (c) 可以降低点透明度，或者用 density / hexbin 减少重叠。
- 可以把 `83.3% below boundary` 和 `96.2% preserved` 做成 panel 内的小 annotation。
- 可以把红框说明放进 caption，而不是图内额外文字。

不要改：

- `Delta` 的定义。
- `epsilon` 的定义。
- `P={2}, tau=0.512, M=1` 的说明。
- “diagnostic operating point，不替代主 benchmark operating point” 这句话。
- Panel (b) 的语义：easy/high-margin skip 更多，hard/low-margin skip 更少。

## 9. 打包清单

建议一起发给对方：

- `paper/figures/reskip_adaptive_depth_vlm.pdf`
- `paper/figures/reskip_adaptive_depth_vlm.png`
- `retrofit/results/reskip_adaptive_depth_figure_pack_cn.md`
- `retrofit/analysis/reskip_token_margin_figure.py`
- `retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160/summary.json`
- `retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160/token_records.jsonl`
- `retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_p2_tau0512_token_n160/sample_records.jsonl`

