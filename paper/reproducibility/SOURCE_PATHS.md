# Canonical source paths

所有路径均在 `New-H100-2:/data/Minko`。本工作区只复制论文所需的小型表格、协议与图；checkpoint、完整 raw eval 和大体积训练产物保留在原实验目录。

## `RSM` 分支内的实现快照

- VLM/RSM 训练、评测、阈值标定与表格构建：`experiments/rsm/code/`
- 冻结的 AR-Retrofit 依赖：`experiments/rsm/code/base_snapshot/`
- device-resident ReSkip runtime：`experiments/rsm/runtime_device/`
- LM benchmark runner：`experiments/rsm/code/run_rsm_lm_eval.py` 与 `run_generic_lm_eval.py`
- VLA 集成、trainer 与 2B/4B 配置：`experiments/rsm/vla/`

分支不提交 checkpoint、数据集、运行日志和本机编译出的 CUDA `.so`；下列服务器路径仍是完整审计源。

## VLM

- 2B seed-0 RSM 方法与运行：`experiments/attnres_residual_strength_gate_2b_20260831`
- 多 seed / 4B / SmolVLM2 / InternVL：`experiments/attnres_rsm_vlm_scale_generalization_20260831`
- Base 与部分 Full AttnRes canonical dependency：`experiments/reskip_paper_vlm_completion_20260824`
- 2B Full AttnRes control dependency：`experiments/attnres_functional_identity_2b_20260831`
- 质量表：`experiments/attnres_rsm_vlm_scale_generalization_20260831/reports/VLM_FINAL_QUALITY_TABLES.json`
- 质量逐 seed：`experiments/attnres_rsm_vlm_scale_generalization_20260831/reports/VLM_FINAL_QUALITY_ROWS.csv`
- 2B safe policy：`experiments/attnres_residual_strength_gate_2b_20260831/policy_screen_seed259123/conservative_gate_policy_device.json`
- 4B safe policy：`experiments/attnres_rsm_vlm_scale_generalization_20260831/reskip/qwen3vl_4b_seed2_capped_policy_seed259123/capped_native_policy.json`
- Pareto：`experiments/attnres_rsm_vlm_scale_generalization_20260831/sensitivity/reports/PARETO_SWEEP.csv`
- Fixed-64 runtime：`experiments/attnres_rsm_vlm_scale_generalization_20260831/sensitivity/reports/RUNTIME_SENSITIVITY.csv`
- 2B natural runtime：`experiments/attnres_residual_strength_gate_2b_20260831/FINAL_RUNTIME_SUMMARY.json`
- 4B natural runtime：`experiments/attnres_rsm_vlm_scale_generalization_20260831/runtime/qwen3vl_4b_seed2_capped/natural12_per_task_r3/BENCHMARK_NATURAL_RUNTIME.json`
- 4B host/device equivalence：`experiments/attnres_rsm_vlm_scale_generalization_20260831/runtime/qwen3vl_4b_seed2_capped/host_device_equivalence_1_per_task_r3/HOST_DEVICE_EQUIVALENCE.json`

## Language-model benchmark

- 完整评测目录：`experiments/reskip_llm_benchmark_20260911`
- 汇总表：`experiments/reskip_llm_benchmark_20260911/tables/llm_benchmark_extended.md`
- seed 级表：`experiments/reskip_llm_benchmark_20260911/tables/llm_benchmark_extended_seedwise.csv`
- 论文工作区副本：`paper/Reskip_RSM/results/llm/`
- 协议副本：`paper/Reskip_RSM/reproducibility/llm/protocol/`

## VLA

- 初始 full-path training/checkpoint：`experiments/attnres_rsm_vla_paper_refresh_20260902`
- corrected transfer/eval：`experiments/attnres_rsm_vla_transfer_fix_20260903`
- 最终合并结果：`experiments/attnres_rsm_vla_transfer_fix_20260903/eval/resume_fullpath_gpu0_3_20260904/combined_fullpath_results.json`
- 队列状态：`experiments/attnres_rsm_vla_transfer_fix_20260903/eval/resume_fullpath_gpu0_3_20260904/queue_status.json`
- 新 2B matched paper-recipe checkpoint 副本：`experiments/attnres_vla_paper_recipe_2b_20260907_node42_eval_source`
- 新 2B matched paper-recipe 评测：`experiments/attnres_vla_paper_recipe_2b_20260907_node42_eval`
- 新评测状态：LIBERO-10 与 Spatial 完成，Object 中断，Goal 未开始；截至 2026-09-12 未进入论文正式数字。
- node43 训练阶段脚本归入：`experiments/attnres_vla_paper_recipe_2b_20260907_node42_eval/protocol/node43_training_stage/`

## Manuscript provenance

- 旧版可编辑稿来源：`starVLA_workspace/reskip/paper/main.tex`
- 原投稿参考 PDF：`starVLA_workspace/reskip/Reskip_5_7_3.pdf`
- 2026-09-07 前改进计划已归档：`archive/reskip_pre_rsm_20260907/root_materials/`
- 2026-09-12 活动工作区清理归档：`archive/reskip_rsm_cleanup_20260912/`
- 飞书改稿页：<https://ccn8j7d4anqr.feishu.cn/wiki/Gm2qw4uBoiUJxakqz2vcKcUjnbh>
