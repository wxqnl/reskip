#!/bin/bash
# Re-run lmms-eval portion of static-pruning baselines without redoing lm-eval.
# Used after killing a stuck mmbench/mathvista OpenAI loop, when text/summary.json
# is already saved.
#
# Usage:
#   bash run_pruning_vlm_only.sh <drop_count> <gpu> <ranking_json>

set -e
DROP="${1:?usage: <drop_count> <gpu> <ranking_json>}"
GPU="${2:?gpu}"
RANK_JSON="${3:?ranking_json}"

PROJECT=/home/user01/Minko/reskip2/reskip
OUTROOT="$PROJECT/retrofit/outputs/static_pruning/drop_${DROP}"
mkdir -p "$OUTROOT"

LAYERS=$(/home/user01/Minko/reskip2/.venv/bin/python -c "
import json
d = json.load(open('$RANK_JSON'))
print(','.join(str(x) for x in d['ranked_low_to_high'][:$DROP]))
")
LAYERS_PIPE=$(echo "$LAYERS" | tr ',' '|')

LOG="$OUTROOT/vlm_only.log"
echo "[run_pruning_vlm_only] DROP=$DROP GPU=$GPU LAYERS=$LAYERS" | tee "$LOG"

cd $PROJECT

TASKS_VLM="mmstar,mmmu_val,ai2d,ocrbench,realworldqa,chartqa,pope"
echo "[run_pruning_vlm_only] launching lmms-eval ($TASKS_VLM)" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PROJECT/retrofit:$PROJECT/retrofit/eval \
  /home/user01/Minko/reskip2/.venv/bin/python -c "
import sys
sys.path.insert(0, '$PROJECT/retrofit')
sys.path.insert(0, '$PROJECT/retrofit/eval')
import lmms_eval_retrofit
import lmms_eval.__main__ as m
sys.argv = ['lmms_eval',
  '--model', 'qwen3_vl_pruned',
  '--model_args', 'pretrained=/home/user01/Minko/models/Qwen3-VL-2B,skip_layers=$LAYERS_PIPE,max_pixels=1605632,min_pixels=200704',
  '--tasks', '$TASKS_VLM',
  '--batch_size', '1',
  '--output_path', '$OUTROOT/vlm',
]
m.cli_evaluate()
" 2>&1 | tee -a "$LOG"

echo "[run_pruning_vlm_only] done at $(date)" | tee -a "$LOG"
