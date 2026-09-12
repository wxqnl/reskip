# 整理与清理记录

最近更新：2026-09-12

## 2026-09-12

活动目录整理为 `manuscript / method / results / analysis / figures / reproducibility` 六类；旧计划和淘汰材料不再与当前材料并列。

### 删除的重复或临时内容

- 删除 `method/code/` 的五个代码副本；均已逐文件确认与 canonical 实验目录一致。
- 删除 `manuscript/figures/` 中 18 个与顶层 `figures/` 完全一致的副本。
- 删除六个未被正文引用、且已在旧仓库或归档保存的历史图和生成脚本。
- 删除活动目录中的旧中文稿、原投稿副本和 pre-revision render；均已确认在旧仓库或中央归档中有相同副本。
- 删除根目录临时 PDF 目录 `tmp_intro_revision_20260911_vDuC1v/`；其中 PDF 与保留的审阅参考稿一致。
- 删除已完成 LLM 队列的 Python cache、空 launcher log 和失效 PID 文件；删除已中断 VLA 队列的失效 PID 文件。

### 移动和归位

- 当前审阅参考稿移动到 `manuscript/reference/`。
- 当前改稿提纲移动到 `manuscript/REVISION_PLAN.md`。
- 旧 checklist 与原 `legacy/` 移到 `/data/Minko/archive/reskip_rsm_cleanup_20260912/`，可恢复。
- node43 VLA 阶段脚本归入对应 node42 eval 的 `protocol/node43_training_stage/`。
- 2026-09-11 的纯文本 LM 汇总表和协议纳入 `results/llm/` 与 `reproducibility/llm/`。
- LaTeX 图路径改为直接读取顶层 canonical 图；编译脚本在退出时清理中间文件。

### 明确保留

- checkpoint、训练日志、完整 raw eval、runtime raw rows、policy 和 protocol。
- 新 2B matched VLA paper-recipe 的已完成与部分评测，供续跑；未把它误写成正式结果。
- `experiments/reskip_paper_vlm_completion_20260824` 与 `experiments/attnres_functional_identity_2b_20260831`，因为最终 provenance 仍引用它们。

## 2026-09-07

- 当前主证据保留在 `experiments/attnres_residual_strength_gate_2b_20260831`、`experiments/attnres_rsm_vlm_scale_generalization_20260831`、`experiments/attnres_rsm_vla_paper_refresh_20260902` 与 `experiments/attnres_rsm_vla_transfer_fix_20260903`。
- 已终止的 2026-08-25 至 2026-08-31 ReSkip/AttnRes 候选归档到 `experiments/_archive/reskip_superseded_20260907/`。
- 旧根目录审计、评审与计划材料归档到 `archive/reskip_pre_rsm_20260907/root_materials/`。
- 删除当前 VLM 实验代码下的 Python cache、旧 PDF 页面渲染缓存和临时文本抽取目录。

两轮整理均未删除 checkpoint、正式 benchmark 输出、runtime raw rows、protocol、policy 或训练日志；已有 Git 工作树中的 tracked 历史文件未改动。
