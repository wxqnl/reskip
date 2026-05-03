#!/bin/bash
# Launch lmms-eval on the new VLM benchmarks (mathvista_testmini, chartqa, pope)
# for one of the four cells {2B_base, 2B_retrofit, 4B_base, 4B_retrofit}.
#
# Usage:
#   bash run_vlm_extra.sh <cell> <gpu>
# where cell ∈ {2B_base, 2B_retrofit, 4B_base, 4B_retrofit}.
#
# Drops outputs into retrofit/outputs/lmms_eval_extra/<cell>/.

set -u
CELL="${1:?usage: <cell> <gpu>}"
GPU="${2:?gpu}"

PROJECT=/home/user01/Minko/reskip2/reskip
TASKS="chartqa,pope"  # mathvista_testmini dropped: requires GPT-4 judge (no API key)
OUTROOT="$PROJECT/retrofit/outputs/lmms_eval_extra/$CELL"
mkdir -p "$OUTROOT"

case "$CELL" in
  2B_base)
    MODEL=qwen3_vl
    MARGS="pretrained=/home/user01/Minko/models/Qwen3-VL-2B,max_pixels=1605632,min_pixels=200704"
    ;;
  2B_retrofit)
    MODEL=qwen3_vl_retrofit
    STATE=$PROJECT/retrofit/outputs/H_2B_r256_10k_L4_v3/retrofit_attnres_state.pt
    MARGS="pretrained=/home/user01/Minko/models/Qwen3-VL-2B,retrofit_state_path=$STATE,max_pixels=1605632,min_pixels=200704"
    ;;
  4B_base)
    MODEL=qwen3_vl
    MARGS="pretrained=/home/user01/Minko/models/Qwen3-VL-4B,max_pixels=1605632,min_pixels=200704"
    ;;
  4B_retrofit)
    MODEL=qwen3_vl_retrofit
    STATE=$PROJECT/retrofit/outputs/H_4B_r256_10k_L4_v3/retrofit_attnres_state.pt
    MARGS="pretrained=/home/user01/Minko/models/Qwen3-VL-4B,retrofit_state_path=$STATE,max_pixels=1605632,min_pixels=200704"
    ;;
  *)
    echo "bad cell: $CELL"; exit 1
    ;;
esac

LOG=$OUTROOT/eval.log
echo "[run_vlm_extra] CELL=$CELL GPU=$GPU TASKS=$TASKS OUT=$OUTROOT" | tee -a "$LOG"
echo "[run_vlm_extra] model=$MODEL" | tee -a "$LOG"
echo "[run_vlm_extra] model_args=$MARGS" | tee -a "$LOG"

cd $PROJECT
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PROJECT/retrofit:$PROJECT/retrofit/eval \
  /home/user01/Minko/reskip2/.venv/bin/python -c "
import sys
sys.path.insert(0, '$PROJECT/retrofit')
sys.path.insert(0, '$PROJECT/retrofit/eval')
import lmms_eval_retrofit  # registers qwen3_vl_retrofit
import lmms_eval.__main__ as m
sys.argv = ['lmms_eval',
  '--model', '$MODEL',
  '--model_args', '$MARGS',
  '--tasks', '$TASKS',
  '--batch_size', '1',
  '--output_path', '$OUTROOT',
]
m.cli_evaluate()
" 2>&1 | tee -a "$LOG"

echo "[run_vlm_extra] done; log=$LOG"
