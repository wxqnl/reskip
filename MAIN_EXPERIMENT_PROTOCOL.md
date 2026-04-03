# 主实验流程

这份文档只保留当前项目真正需要的主干实验，不再混入多余的 `LM -> VLA` 桥接设定。

现在的结论很明确：

1. `LM` 和 `VLA` 可以完全分开做。
2. `LM` 的目标是验证 `AttnRes / ReSkip / ReLoop` 在语言建模上的成立性。
3. `VLA` 的目标是验证 `AttnRes / ReSkip` 在真实机器人控制 backbone 上的有效性。
4. 不要求 LM 和 VLA 共用同一套骨架，也不要求 LM checkpoint 直接迁移到 VLA。

---

## 一、LM 主线

### 1. 目标

LM 部分要回答的是：

- `AttnRes` 在真实自然语言预训练里是否成立
- `ReSkip` 是否能形成质量-算力 tradeoff
- `ReLoop` 是否能在参数共享下保持效果
- 这些结论是否能随着模型规模变化而保持

### 2. 数据

主数据：

- `SlimPajama-627B`
- 本地 parquet

当前服务器路径：

```text
/inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data
```

### 3. tokenizer

LM 可以独立选择 tokenizer。

当前可行路线有两种：

1. 直接用 repo 主线：
   - `tiktoken/gpt2`
   - 配合 `experiments/train_lm.py`

2. 用 flame 高性能框架：
   - 本地 HuggingFace tokenizer
   - 配合 `flame/custom_models/reskip_qwen3`

这里不需要和 VLA 保持一致。

### 4. 核心实验

#### LM-1 Baseline

模型：

- `d_model=896`
- `n_heads=14`
- `n_layers=24`
- `n_blocks=8`
- `seq_len=2048`
- 约 `350M`

命令：

```bash
torchrun --nproc_per_node=1 experiments/train_lm.py \
  --mode baseline \
  --dataset slimpajama \
  --data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --val_data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --text_key text \
  --tokenizer_name gpt2 \
  --d_model 896 \
  --n_heads 14 \
  --n_layers 24 \
  --n_blocks 8 \
  --seq_len 2048 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --max_steps 20000 \
  --warmup_steps 2000 \
  --lr 3e-4 \
  --weight_decay 0.1 \
  --use_rope \
  --amp_dtype bf16 \
  --num_train 5120000 \
  --num_val 4096 \
  --num_workers 4 \
  --log_every 1 \
  --device cuda \
  --output_dir outputs/350m_baseline
```

#### LM-2 AttnRes

命令：

```bash
torchrun --nproc_per_node=1 experiments/train_lm.py \
  --mode standard \
  --dataset slimpajama \
  --data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --val_data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --text_key text \
  --tokenizer_name gpt2 \
  --d_model 896 \
  --n_heads 14 \
  --n_layers 24 \
  --n_blocks 8 \
  --seq_len 2048 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --max_steps 20000 \
  --warmup_steps 2000 \
  --lr 3e-4 \
  --weight_decay 0.1 \
  --use_rope \
  --amp_dtype bf16 \
  --num_train 5120000 \
  --num_val 4096 \
  --num_workers 4 \
  --log_every 1 \
  --device cuda \
  --output_dir outputs/350m_attnres
```

#### LM-3 Skip Sweep

命令：

```bash
python experiments/train_lm.py \
  --mode eval \
  --checkpoint outputs/350m_attnres/best.pt \
  --dataset slimpajama \
  --data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --val_data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --text_key text \
  --tokenizer_name gpt2 \
  --batch_size 2 \
  --num_val 4096 \
  --num_workers 4 \
  --device cuda \
  --output_dir outputs/350m_attnres_eval
```

#### LM-4 Routing Analysis

命令：

```bash
python experiments/analyze_routing.py \
  --checkpoint outputs/350m_attnres/best.pt \
  --sweep_results outputs/350m_attnres/final_results.json \
  --train_log outputs/350m_attnres/train_log.jsonl \
  --device cuda \
  --output_dir outputs/350m_attnres/analysis
```

#### LM-5 ReLoop

命令：

```bash
python experiments/train_lm.py \
  --mode looping \
  --dataset slimpajama \
  --data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --val_data_path /inspire/ssd/project/multimodal-brain-signal/liuzhenyang-240108540154/Simo/datasets/SlimPajama-627B/data \
  --text_key text \
  --tokenizer_name gpt2 \
  --d_model 896 \
  --n_heads 14 \
  --n_layers 24 \
  --n_blocks 8 \
  --seq_len 2048 \
  --n_unique_blocks 4 \
  --max_loops 3 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --max_steps 20000 \
  --warmup_steps 2000 \
  --lr 3e-4 \
  --weight_decay 0.1 \
  --amp_dtype bf16 \
  --device cuda \
  --output_dir outputs/350m_reloop_k4_m3
```

### 5. 扩展规模

LM 可以继续做：

- `700M`
- `1.3B`
- `2B`

这些实验是独立的 LM scaling 实验，不需要考虑和 VLA 一致。

---

## 二、VLA 主线

### 1. 目标

VLA 部分要回答的是：

- `AttnRes` 能不能接进真实 VLA backbone
- 训练后能不能学出有意义的 depth routing
- `uniform skip` 和 `modality-aware skip` 哪个更好
- 在 rollout 上能否提升效率且尽量不损失 success rate

### 2. 正确实验对象

VLA 主实验不是：

- `experiments/benchmark_vla.py`

那个只是 toy benchmark。

VLA 主实验真正要跑的是：

- `StarVLA`
- `Qwen3-VL-4B-Instruct-Action`
- `LIBERO`

### 3. 数据

训练数据：

- `libero_all`

评测：

- `libero_goal`
- 或者你本地可用的 `LIBERO-Long` 套件

### 4. tokenizer

VLA 这边直接跟随 `Qwen3-VL-4B-Instruct-Action`。

这里不需要理会 LM 用的 tokenizer。

### 5. 核心思路

VLA 不是从零训练一个新的 VLM。

而是：

1. 直接使用现成的 `Qwen3-VL-4B-Instruct-Action`
2. 在它上面接入 `AttnRes`
3. 用 `LIBERO` 微调
4. 再做 rollout 评测

所以这里本质上是：

- `Qwen3-VL` 原 backbone 继续复用
- `AttnRes` 作为新增结构重新训练

### 6. 核心实验

#### VLA-1 Full-depth 训练

目的：

- 验证 AttnRes 接入真实 VLA backbone 后可以稳定训练
- 先不启用 skip

命令：

```bash
bash starVLA/examples/LIBERO/train_files/run_libero_train_attnres.sh
```

这一步训练的是：

- backbone: `QwenOFT`
- base model: `Qwen3-VL-4B-Instruct-Action`
- dataset: `LIBERO`
- AttnRes: 开
- skip: 关

#### VLA-2 Full-depth rollout

先启动 server：

```bash
cd starVLA
bash examples/LIBERO/eval_files/run_policy_server_attnres.sh
```

再跑 rollout：

```bash
bash examples/LIBERO/eval_files/eval_libero_attnres.sh
```

这一步对应：

- `skip_mode=none`

#### VLA-3 Uniform Skip rollout

使用同一个 checkpoint，只改 skip 配置，跑：

- `skip_mode=uniform`

#### VLA-4 Modality-aware Skip rollout

还是同一个 checkpoint，再跑：

- `skip_mode=modality_aware`

### 7. 最终比较什么

VLA 主实验真正该比较的是三组：

1. `full depth`
2. `uniform skip`
3. `modality-aware skip`

主指标：

- `success rate`
- `latency`
- `effective block ratio`
- `skip 后性能下降多少`

---

## 三、两条主线的关系

现在主干实验里，`LM` 和 `VLA` 是分开的。

它们的关系不是参数直接共享，而是方法论共享：

1. `LM`
   证明 `AttnRes / ReSkip / ReLoop` 在自然语言预训练里成立

2. `VLA`
   证明同样的结构思想接到真实机器人 backbone 后也成立

所以当前实验主干里：

- 不需要 LM 和 VLA 骨架一致
- 不需要 tokenizer 一致
- 不需要 LM checkpoint 直接加载到 VLA

## 四、现在真正的主干核心

如果只保留最核心的东西，就是：

### LM

1. baseline
2. AttnRes
3. skip sweep
4. routing analysis
5. ReLoop

### VLA

1. StarVLA + AttnRes full-depth 训练
2. full-depth rollout
3. uniform skip rollout
4. modality-aware skip rollout

这就是当前最干净、最合理、最符合 paper 主线的实验结构。
