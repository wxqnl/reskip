#!/bin/bash
# Q2 canonical-mix ablations (4 cells × ~26 min each, parallel on GPUs 0-3).
# Anchored on H_r256_5k canonical: r=256, 5k, 50/50, γ-curr 0→1 at frac=0.3, KL=1.
#
# A_lora      (GPU 0): pure LoRA SFT, rank=64 q+v → ~14M trainable, no AttnRes
# B_no_adapter(GPU 1): canonical retrofit but --no-adapter (γ-blend, no MLP)
# C_no_skipkl (GPU 2): canonical retrofit but --kl-weight 0 (drop skip-KL)
# D_no_curric (GPU 3): canonical retrofit but γ→1 immediately (no curriculum)
set -u
PROJECT=/home/user01/Minko/reskip2/reskip
PY=/home/user01/Minko/reskip2/.venv/bin/python
MODEL=/home/user01/Minko/models/Qwen3-VL-2B
OUT_ROOT=$PROJECT/retrofit/outputs/q2_ablations
mkdir -p "$OUT_ROOT"
cd "$PROJECT"

run_canonical_variant() {
  local label="$1"; local gpu="$2"; shift 2
  local outdir="$OUT_ROOT/${label}"
  mkdir -p "$outdir"
  local log="$outdir/run.log"
  echo "[$(date)] $label gpu=$gpu -> $outdir" | tee -a "$OUT_ROOT/launch.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY retrofit/train/train_qwen3vl_attnres_retrofit.py \
    --model-path "$MODEL" \
    --num-blocks 14 \
    --adapter-rank 256 \
    --steps 5000 \
    --gamma-schedule --gamma-start 0 --gamma-end 1 --gamma-ramp-frac 0.3 \
    --p-multimodal 0.5 \
    --kl-weight 1.0 \
    --entropy-weight 0.02 \
    --output-dir "$outdir" \
    --gpu 0 \
    "$@" \
    > "$log" 2>&1 &
  echo "  pid=$!  log=$log" | tee -a "$OUT_ROOT/launch.log"
}

run_lora_sft() {
  local label="$1"; local gpu="$2"
  local outdir="$OUT_ROOT/${label}"
  mkdir -p "$outdir"
  local log="$outdir/run.log"
  echo "[$(date)] $label gpu=$gpu -> $outdir" | tee -a "$OUT_ROOT/launch.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY retrofit/train/train_qwen3vl_lora.py \
    --lora-r 64 --lora-alpha 128 \
    --target-modules q_proj,v_proj \
    --steps 5000 \
    --p-multimodal 0.5 \
    --output-dir "$outdir" \
    --gpu 0 \
    > "$log" 2>&1 &
  echo "  pid=$!  log=$log" | tee -a "$OUT_ROOT/launch.log"
}

# === Launch all 4 in parallel on GPUs 0-3 ===
run_lora_sft           "A_lora_r64_qv"       0
run_canonical_variant  "B_no_adapter"        1  --no-adapter
run_canonical_variant  "C_no_skipkl_kl0"     2  --kl-weight 0
run_canonical_variant  "D_no_gamma_curric"   3  --gamma-ramp-frac 0.001

wait
echo "[$(date)] ALL Q2 ABLATION CELLS DONE" | tee -a "$OUT_ROOT/launch.log"
