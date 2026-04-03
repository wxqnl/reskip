# Flame LM Playbook

This document describes how to run the repo's main-line LM pretraining in the
local `flame/` framework.

## What Was Added

- a HuggingFace-style custom model type: `reskip_transformer`
- a baseline configuration
- an AttnRes configuration
- flame registration through `AutoConfig` and `AutoModelForCausalLM`
- tensor-parallel compatibility by mapping `reskip_transformer` to flame's transformer TP plan

Key files:

- `flame/custom_models/reskip_transformer/config_reskip_transformer.py`
- `flame/custom_models/reskip_transformer/modeling_reskip_transformer.py`
- `flame/configs/reskip_baseline_350M.json`
- `flame/configs/reskip_attnres_350M.json`

## Model Variants

`reskip_baseline_350M.json`

- standard decoder-only Transformer
- RoPE enabled
- no AttnRes routing

`reskip_attnres_350M.json`

- same backbone size
- block-level AttnRes enabled
- 8 AttnRes blocks over 24 decoder layers

Both configs target the repo's 350M-scale experiments:

- hidden size: 896
- layers: 24
- heads: 14
- FFN size: 3584
- seq len: 2048

## Tokenizer Requirement

Flame expects a HuggingFace tokenizer path.

You need a local tokenizer directory for offline training, for example a local
`gpt2` tokenizer export. Point `--model.tokenizer_path` at that directory.

## Dataset Requirement

Flame expects text datasets through its standard loader. For local SlimPajama
parquet shards, use a preprocessing path that exposes text examples to flame's
dataset layer, or point it at a local HuggingFace-compatible dataset/cache.

## Baseline Training

Run from the repo root:

```bash
cd flame
NNODE=1 NGPU=4 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder ../outputs/flame_reskip_baseline_350m \
  --model.config configs/reskip_baseline_350M.json \
  --model.tokenizer_path /path/to/local/gpt2_tokenizer \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 2000 \
  --lr_scheduler.decay_type cosine \
  --lr_scheduler.lr_min 0.1 \
  --training.batch_size 8 \
  --training.seq_len 2048 \
  --training.context_len 2048 \
  --training.gradient_accumulation_steps 8 \
  --training.steps 20000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset parquet \
  --training.data_files /path/to/SlimPajama-627B/data/train-*.parquet \
  --training.num_workers 8 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.data_parallel_shard_degree -1 \
  --training.tensor_parallel_degree 1 \
  --checkpoint.interval 1000 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 10
```

## AttnRes Training

```bash
cd flame
NNODE=1 NGPU=4 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder ../outputs/flame_reskip_attnres_350m \
  --model.config configs/reskip_attnres_350M.json \
  --model.tokenizer_path /path/to/local/gpt2_tokenizer \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 2000 \
  --lr_scheduler.decay_type cosine \
  --lr_scheduler.lr_min 0.1 \
  --training.batch_size 8 \
  --training.seq_len 2048 \
  --training.context_len 2048 \
  --training.gradient_accumulation_steps 8 \
  --training.steps 20000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset parquet \
  --training.data_files /path/to/SlimPajama-627B/data/train-*.parquet \
  --training.num_workers 8 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.data_parallel_shard_degree -1 \
  --training.tensor_parallel_degree 1 \
  --checkpoint.interval 1000 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 10
```

## Notes

- the flame model is registered at import time by `flame/custom_models/__init__.py`
- the custom type name is `reskip_transformer`
- the current flame integration is focused on main-line pretraining, not the repo's original skip-sweep evaluation path
