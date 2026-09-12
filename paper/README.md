# ReSkip / RSM 论文工作区

这是论文写作、定稿数字与复现索引的唯一入口。正文数字以 [PAPER_NUMBERS.md](PAPER_NUMBERS.md) 为准；实验目录保存原始产物，不直接承担论文材料导航。

## 当前入口

1. `manuscript/main.tex`：唯一可编辑 LaTeX 正文；`main.pdf` 是它的当前构建结果。
2. `manuscript/REVISION_PLAN.md`：当前改稿提纲和方法边界。
3. `PAPER_NUMBERS.md`：已经确认、可写入论文的数字及使用限制。
4. `method/METHOD_AND_EVIDENCE.md`：AR-Retrofit、RSM 和 ReSkip 的方法证据。
5. `results/` 与 `analysis/`：整理后的结果表、敏感性分析和统计边界。
6. `reproducibility/SOURCE_PATHS.md`：checkpoint、原始结果、policy、runtime receipt 和新增评测的服务器路径。

## 当前证据状态

| 模块 | 状态 | 论文位置 |
|---|---|---|
| Qwen3-VL-2B/4B Full RSM-AttnRes | 可用，三训练 seed | 主表与主结论 |
| SmolVLM2-2.2B Full RSM-AttnRes | 可用，三训练 seed；绝对 Base margin 很小 | 跨架构补充 |
| InternVL3.5-2B | 未通过正向迁移门 | 诊断/局限性，不写成成功 |
| Qwen3-VL-2B/4B token-level ReSkip | 可用，冻结推理策略 | 主表或首个附录表 |
| Fixed-64 H100 runtime | 可用，六任务各 12 请求、三重复 | 主要速度证据 |
| Natural-generation runtime | 可用，但 4B 略慢于 Base | 完整披露/附录 |
| LIBERO VLA | 已完成；2B/4B 方向不一致 | 支持性或附录，不作普适主张 |
| 七项纯文本 LM benchmark | 2026-09-11 已完成；尚未写入正文 | 补充能力验证候选 |
| 新 2B matched VLA paper-recipe rerun | 部分完成后中断；不得作为正式结果 | 待续跑 |

## 唯一方法口径

- RSM 是 AR-Retrofit/AttnRes Full-path 适应的一部分，不是外挂 skip controller。
- RSM 在普通 Full-path 训练中联合学习；训练不执行 skip forward，不使用 skip label、compute target 或第二阶段训练。
- ReSkip 在模型冻结后复用 AttnRes 原生路由与 RSM 状态，逐 token 做推理期动态深度决策。
- 训练新增参数属于 AttnRes 表达路径；没有专门为跳层训练的额外 gate/head。

## 数字使用规则

- Full 模型的主表使用三 seed 均值与样本标准差。
- ReSkip 的质量损失必须与同一 seed 的 Full RSM 比较：2B seed 0，4B seed 2。
- Fixed-64 主速度使用统一的六任务 `n=72` 口径：2B `1.0438x`、4B `1.0286x` 相对 Base。
- 早期 2B 单任务、12 请求的 `1.0707x` 只保留在审计源中，不作为论文主数字。
- 不把六个 benchmark 当作 iid 重复，不把跨任务范围写成置信区间。
- 新实验只有在同步更新 `PAPER_NUMBERS.md` 后，才进入正文口径。

## 目录

```text
manuscript/        正文、当前 PDF、改稿计划和一份审阅参考稿
method/            当前方法说明；实现代码只从 canonical 实验目录引用
results/           VLM、LM、VLA 与 runtime 的整理结果
analysis/          敏感性、统计验证和失败边界
figures/           主文与附录 canonical 图；manuscript 不再保留重复副本
reproducibility/   protocol、policy、runtime receipt 与源路径索引
```

历史计划、淘汰路线和旧稿不放在活动目录；可恢复归档位于 `/data/Minko/archive/reskip_rsm_cleanup_20260912/`，更早材料位于 `/data/Minko/archive/reskip_pre_rsm_20260907/`。

飞书改稿计划：<https://ccn8j7d4anqr.feishu.cn/wiki/Gm2qw4uBoiUJxakqz2vcKcUjnbh>
