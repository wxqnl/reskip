# Flame LM Playbook

This document describes the `flame/` LM pretraining path that is compatible with
the repo's `StarVLA` Qwen backbones.

## What Changed

- added a new HuggingFace-style custom model type: `reskip_qwen3`
- added a config-generation script that extracts the text backbone config from a
  local `Qwen3-VL` / `StarVLA` base model directory
- updated `flame` to sync `vocab_size`, `bos/eos/pad` ids from the tokenizer at
  runtime
- added a `StarVLA` language-model init hook that can load a flame LM checkpoint
  into `qwen_vl_interface.model.language_model`

Key files:

- `flame/custom_models/reskip_qwen3/config_reskip_qwen3.py`
- `flame/custom_models/reskip_qwen3/modeling_reskip_qwen3.py`
- `flame/scripts/make_reskip_qwen3_config.py`
- `flame/flame/utils/convert_dcp_to_hf.py`
- `src/starvla_integration.py`
- `starVLA/starVLA/model/modules/vlm/QWen3.py`

## Compatibility Rule

If you want the LM checkpoint to warm-start VLA training, the LM pretraining
must use:

- the same tokenizer directory as `StarVLA` `base_vlm`
- the same text hidden size / layer count / attention layout as the target
  `Qwen3-VL` text backbone

Do not substitute another tokenizer for this transfer path.

## Step 1: Generate a Flame Config from the VLA Base Model

Assume your VLA backbone is:

- `/path/to/Qwen3-VL-4B-Instruct-Action`

Generate the flame LM config from that local model:

```bash
cd flame
python scripts/make_reskip_qwen3_config.py \
  --base-vlm-path /path/to/Qwen3-VL-4B-Instruct-Action \
  --output configs/reskip_qwen3_attnres_from_starvla.json \
  --use-attn-res \
  --attn-res-num-blocks 8 \
  --attn-res-temperature 1.0
```

For a baseline config, omit `--use-attn-res`:

```bash
cd flame
python scripts/make_reskip_qwen3_config.py \
  --base-vlm-path /path/to/Qwen3-VL-4B-Instruct-Action \
  --output configs/reskip_qwen3_baseline_from_starvla.json
```

## Step 2: Train in Flame

Use the same local tokenizer/model directory for `--model.tokenizer_path`.

AttnRes LM training:

```bash
cd flame
NNODE=1 NGPU=4 LOG_RANK=0 bash train.sh \
  --job.config_file flame/flame/models/fla.toml \
  --job.dump_folder ../outputs/flame_reskip_qwen3_attnres \
  --model.config configs/reskip_qwen3_attnres_from_starvla.json \
  --model.tokenizer_path /path/to/Qwen3-VL-4B-Instruct-Action \
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

Baseline LM training is identical except:

- `--model.config configs/reskip_qwen3_baseline_from_starvla.json`
- output dir changed

## Step 3: Convert Flame Checkpoint to HF Format

StarVLA loads a normal HuggingFace-style checkpoint or checkpoint directory. If
your flame run saved DCP checkpoints, convert one step to HF format:

```bash
cd flame
python flame/utils/convert_dcp_to_hf.py \
  --path ../outputs/flame_reskip_qwen3_attnres_hf \
  --step 20000 \
  --config configs/reskip_qwen3_attnres_from_starvla.json \
  --tokenizer /path/to/Qwen3-VL-4B-Instruct-Action
```

This creates a directory like:

- `../outputs/flame_reskip_qwen3_attnres_hf/config.json`
- `../outputs/flame_reskip_qwen3_attnres_hf/model.safetensors` or
  `pytorch_model.bin`
- tokenizer files

## Step 4: Warm-Start StarVLA from the Flame LM Checkpoint

`StarVLA` now supports:

- `framework.qwenvl.language_model_init_checkpoint`
- `framework.qwenvl.language_model_init_strict`

You can pass the HF checkpoint directory directly:

```bash
cd starVLA
language_model_init_checkpoint=/absolute/path/to/outputs/flame_reskip_qwen3_attnres_hf \
bash examples/LIBERO/train_files/run_libero_train_attnres.sh
```

Or call the training script directly:

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero_attnres.yaml \
  --framework.qwenvl.base_vlm /path/to/Qwen3-VL-4B-Instruct-Action \
  --framework.qwenvl.language_model_init_checkpoint /absolute/path/to/outputs/flame_reskip_qwen3_attnres_hf
```

## What Actually Transfers

The StarVLA loader imports the LM weights into:

- `qwen_vl_interface.model.language_model`

It ignores:

- `lm_head.*`
- non-language-model weights

This is the intended path:

1. flame text pretraining on SlimPajama
2. export HF checkpoint
3. StarVLA language-model warm start
4. VLA finetuning / rollout evaluation

## Tokenizer Notes

For this transfer-compatible route:

- use the same tokenizer path as `base_vlm`
- do not replace it with `gpt2` or another tokenizer
- flame now synchronizes `vocab_size`, `bos_token_id`, `eos_token_id`, and
  `pad_token_id` from the tokenizer at train startup

If you intentionally switch tokenizer families, you break direct LM-to-VLA
checkpoint reuse.
