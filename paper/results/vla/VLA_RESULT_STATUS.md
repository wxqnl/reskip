# LIBERO VLA 当前结果与使用边界

正式 full-path 队列已完成。每个 policy 在 LIBERO Spatial/Object/Goal/10 各评测 500 episodes。

| Scale | System | Spatial | Object | Goal | LIBERO-10 | Macro |
|---|---|---:|---:|---:|---:|---:|
| 2B | Base | 93.4 | 96.6 | 93.4 | 67.0 | 87.60 |
| 2B | AttnRes | 94.8 | 97.8 | 90.8 | 68.8 | 88.05 |
| 2B | RSM v2 | 96.4 | 98.0 | 93.8 | 56.0 | 86.05 |
| 4B | Base | 95.4 | 97.8 | 98.0 | 65.8 | 89.25 |
| 4B | AttnRes | 95.2 | 95.0 | 95.6 | 58.4 | 86.05 |
| 4B | RSM v2 | 97.2 | 99.4 | 92.8 | 71.2 | 90.15 |
| 2B | Path-C RSM (auxiliary) | 95.0 | 100.0 | 95.4 | 74.2 | 91.15 |

解释：4B primary RSM 相对 Base `+0.90 pp`、相对 AttnRes `+4.10 pp`；2B primary RSM 相对 Base `−1.55 pp`、相对 AttnRes `−2.00 pp`，下降集中在 LIBERO-10。Path-C 为 auxiliary training path，虽然达到 91.15，但不能在未重新定义实验角色的情况下替代 primary。

论文建议：当前 VLA 不作为“跨规模稳定提升”的核心证据。若正文篇幅紧张，移至附录；若保留正文，必须完整报告方向不一致，并把结论限定为“RSM 可迁移到 embodied policy，但结果对 scale/training path 敏感”。

## 2026-09-10 matched paper-recipe rerun

新的 2B Base/AttnRes matched recipe 已完成 LIBERO-10 与 Spatial，各 500 episodes；两条队列均在 Object 评测中断，Goal 尚未开始，当前也没有活动进程。其已完成结果和部分 rollout 均保留在 `experiments/attnres_vla_paper_recipe_2b_20260907_node42_eval/`，供续跑使用。

该 rerun 在四个 suite 全部完成并形成统一汇总前，不替换上表，也不进入 `PAPER_NUMBERS.md`。
