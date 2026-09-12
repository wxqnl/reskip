# RSM implementation snapshot

This directory collects the source files used by the current RSM experiments. It keeps the original run structure where that structure affects imports or checkpoint loading.

## Layout

| Path | Role |
|---|---|
| `code/` | RSM installation, Qwen and generic-VLM training, evaluation, policy calibration, result assembly, and LM evaluation |
| `code/base_snapshot/` | Frozen AR-Retrofit and data-loader dependency used by the recorded runs |
| `runtime_device/` | Device-resident action selection, K/V-only layer path, and CUDA conditional-graph source |
| `vla/` | RSM-aware starVLA integration, trainer, scale-specific configs, checkpoint initialization, and launcher |

The source under `code/` comes from the final VLM scale-generalization experiment. `run_rsm_lm_eval.py`, `run_generic_lm_eval.py`, and `build_llm_table.py` come from the completed 2026-09-11 language-model evaluation. The VLA integration was identical in the completed 2B and 4B transfer runs, so this directory keeps one source copy and two configs.

The prebuilt CUDA shared object is excluded. `runtime_device/device_conditional_reskip.py` compiles `runtime_device/csrc/device_conditional_graph_ext.cu` for the local CUDA environment when needed.

## Main entry points

- `code/train_residual_strength_gate.py`: train Qwen full-path RSM from the frozen base.
- `code/train_generic_vlm_attnres.py`: train the generic-VLM scale-transfer cells.
- `code/evaluate_residual_strength_gate.py`: forced-action and held-out evaluation.
- `code/calibrate_native_gate_policy.py`: calibrate frozen ReSkip thresholds from recorded native signals.
- `code/bench_rsm_natural_runtime.py`: matched natural-generation and fixed-length runtime harness.
- `code/run_rsm_lm_eval.py` and `code/run_generic_lm_eval.py`: zero-shot LM evaluation.
- `vla/train_starvla.py`: VLA fine-tuning with the RSM-aware integration.

Use the policies and protocols under [`../../paper/reproducibility/`](../../paper/reproducibility/). Server-side checkpoints and full raw outputs are indexed in [`../../paper/reproducibility/SOURCE_PATHS.md`](../../paper/reproducibility/SOURCE_PATHS.md).

The scripts require the original model, dataset, `transformers`, `lmms-eval`, `lm-evaluation-harness`, Triton, and starVLA environments used by the experiments. No small-model or synthetic fallback is included.
