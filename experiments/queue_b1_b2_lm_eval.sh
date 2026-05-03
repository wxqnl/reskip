#!/bin/bash
# Queue runner for B1/B2 lm-eval cells on a single GPU.
# Usage:
#   bash queue_b1_b2_lm_eval.sh <gpu> <which_set>

set -e
GPU="${1:?usage: <gpu> <set>}"
SET="${2:?which set}"

PROJECT=/home/user01/Minko/reskip2/reskip
PYBIN=$PROJECT/../.venv/bin/python
LM_EVAL=$PROJECT/experiments/baseline_b1_b2_lm_eval.py
TASKS="piqa,openbookqa,arc_easy,arc_challenge,mmlu,lambada_openai,hellaswag"
LOG_ROOT=$PROJECT/retrofit/outputs/lm_eval_b1b2

mkdir -p $LOG_ROOT

run_dynamic() {
  local cell_dir=$1
  local label=$2
  local out_subdir=$3
  echo "[queue:$SET:gpu$GPU] starting $label at $(date)"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PROJECT/experiments:$PROJECT/flash-linear-attention \
    $PYBIN $LM_EVAL \
      --model_path $cell_dir \
      --device cuda --tasks "$TASKS" --batch_size 8 \
      --output $LOG_ROOT/$out_subdir \
      --label "$label" \
      2>&1 | tee $LOG_ROOT/${out_subdir}.log
  echo "[queue:$SET:gpu$GPU] finished $label at $(date)"
}

run_random() {
  local label="B1e_B2d_random_P3orP5"
  echo "[queue:$SET:gpu$GPU] starting $label at $(date)"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PROJECT/experiments:$PROJECT/flash-linear-attention \
    $PYBIN $LM_EVAL \
      --model_path $PROJECT/flame/saves/reskip_transformer-340M \
      --device cuda --tasks "$TASKS" --batch_size 8 \
      --output $LOG_ROOT/b1e_random_P3orP5 \
      --label "$label" \
      --runtime_mode random \
      --random_choices "1,1,1,0,1,1,1,1;1,1,1,1,1,0,1,1" \
      --random_seed 0 \
      2>&1 | tee $LOG_ROOT/b1e_random_P3orP5.log
  echo "[queue:$SET:gpu$GPU] finished $label at $(date)"
}

case "$SET" in
  gpu1_after_cell1)
    # GPU 1 already running cell 1 (recent_weight_gt). After it ends, do 1 more.
    run_dynamic $PROJECT/outputs/reskip_340M_b1b2_recent_minus_embed_gt_q050_M1 \
                "B2c_recent_minus_embed_gt_q050_M1" \
                "b2c_recent_minus_embed_gt_q050_M1"
    ;;
  gpu2_after_4b)
    # Split: GPU 2 takes entropy_lt only; static_p3 moved to GPU 3
    run_dynamic $PROJECT/outputs/reskip_340M_b1b2_entropy_lt_q050_M1 \
                "B2b_entropy_lt_q050_M1" \
                "b2b_entropy_lt_q050_M1"
    ;;
  gpu3_static_p3)
    run_dynamic $PROJECT/outputs/reskip_340M_b1_static_p3_every \
                "B1c_static_p3_every" \
                "b1c_static_p3_every"
    ;;
  gpu3_after_4b)
    run_dynamic $PROJECT/outputs/reskip_340M_b1_static_p5_every \
                "B1d_static_p5_every" \
                "b1d_static_p5_every"
    run_random
    ;;
  *) echo "bad set: $SET"; exit 1 ;;
esac

echo "[queue:$SET:gpu$GPU] all done at $(date)"
