# Flame LM 说明

这份文档只说明 `flame` 在 LM 主线里的用途，不再讨论和 `StarVLA` 的直接桥接。

## 1. 现在 flame 用来做什么

`flame` 的作用是：

- 给 LM 主线提供一个更高性能的训练框架
- 支持 `Qwen3` 风格的 decoder-only LM 实验
- 用于做不同参数规模下的 LM 可行性实验

它现在不再承担：

- 直接初始化 `StarVLA`
- 直接连接 VLA 训练

## 2. 当前实现

关键文件：

- `flame/custom_models/reskip_qwen3/config_reskip_qwen3.py`
- `flame/custom_models/reskip_qwen3/modeling_reskip_qwen3.py`
- `flame/scripts/make_reskip_qwen3_config.py`

## 3. tokenizer

LM 在 flame 里可以使用你自己指定的本地 HuggingFace tokenizer。

要求是：

- tokenizer 本地可加载
- 和当前 LM 实验目标一致

不需要和 `StarVLA` 一致。

## 4. 基本流程

### 4.1 生成配置

如果你要做 `Qwen3` 风格 LM，可先生成 config：

```bash
cd flame
python scripts/make_reskip_qwen3_config.py \
  --base-vlm-path /path/to/local/qwen3_or_qwen3_vl_dir \
  --output configs/reskip_qwen3_attnres.json \
  --use-attn-res \
  --attn-res-num-blocks 8 \
  --attn-res-temperature 1.0
```

如果要 baseline，去掉 `--use-attn-res`。

### 4.2 训练

```bash
cd flame
NNODE=1 NGPU=4 LOG_RANK=0 bash train.sh \
  --job.config_file flame/flame/models/fla.toml \
  --job.dump_folder ../outputs/flame_reskip_qwen3_attnres \
  --model.config configs/reskip_qwen3_attnres.json \
  --model.tokenizer_path /path/to/local_tokenizer \
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

## 5. 定位

`flame` 现在就是 LM 主线的一部分：

- 可以做 `Qwen3` 风格对比
- 可以做不同参数量实验
- 可以独立于 VLA 存在

如果你的目标是论文主干，优先把它理解成：

- `LM 的高性能训练后端`

而不是：

- `LM -> VLA 的中间桥梁`

## 6. 一键 350M 全流程

如果你要直接跑 `350M baseline + 350M AttnRes + skip sweep + lm_eval benchmark + 结果图`，现在已经有一键脚本：

```bash
bash flame/scripts/run_reskip_350m_full_pipeline.sh
```

运行前至少要设置两个环境变量：

```bash
export TOKENIZER_PATH=/path/to/local_hf_tokenizer
export DATA_GLOB=/path/to/SlimPajama-627B/data/train-*.parquet
```

常用可选项：

```bash
export NGPU=4
export DEVICE=cuda:0
export TASKS=lambada_openai,hellaswag,arc_easy,arc_challenge
export THRESHOLDS="0.01 0.02 0.05 0.10"
export RESULT_ROOT=../outputs/flame_reskip_350m_pipeline
```

这个脚本会依次完成：

1. 训练 `350M baseline`
2. 训练 `350M AttnRes`
3. 对 `baseline` 做 `lm_eval`
4. 对 `AttnRes full-depth` 做 `lm_eval`
5. 对多个 skip threshold 做 `lm_eval + block profile`
6. 生成：
   - `reskip_350m_benchmark.png`
   - `summary_table.json`
   - `selection_summary.json`
   - `selection_summary.md`

其中推荐的 skip threshold 会根据自动规则选出：

- 优先选择在 `attnres_full` 分数 3% 以内损失的前提下，执行 block 最少的设置
- 如果没有满足条件的设置，就退化为选择分数最高的 skip 设置
