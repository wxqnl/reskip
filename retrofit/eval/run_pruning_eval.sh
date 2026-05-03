#!/bin/bash
# Run a static-pruning baseline cell on Qwen3-VL-2B:
#   1. Read Gromov-style influence ranking from JSON.
#   2. Pick the lowest-N layers as the skip set.
#   3. Run lmms-eval on the 9 paper-VLM tasks + lm-eval on LAMBADA.
#
# Usage:
#   bash run_pruning_eval.sh <drop_count> <gpu> <ranking_json>
#
# Example:
#   bash run_pruning_eval.sh 4 0 retrofit/outputs/static_pruning/gromov_ranking_2B.json

set -e
DROP="${1:?usage: <drop_count> <gpu> <ranking_json>}"
GPU="${2:?gpu}"
RANK_JSON="${3:?ranking_json}"

PROJECT=/home/user01/Minko/reskip2/reskip
OUTROOT="$PROJECT/retrofit/outputs/static_pruning/drop_${DROP}"
mkdir -p "$OUTROOT"

# Pick the lowest-influence layers as a comma-separated list.
LAYERS=$(/home/user01/Minko/reskip2/.venv/bin/python -c "
import json, sys
d = json.load(open('$RANK_JSON'))
print(','.join(str(x) for x in d['ranked_low_to_high'][:$DROP]))
")
LAYERS_PIPE=$(echo "$LAYERS" | tr ',' '|')

LOG="$OUTROOT/eval.log"
echo "[run_pruning_eval] DROP=$DROP GPU=$GPU LAYERS=$LAYERS" | tee "$LOG"

cd $PROJECT

# 1. lmms-eval on 9 VLM tasks (6 paper + 3 new).
TASKS_VLM="mmstar,mmmu_val,ai2d,ocrbench,realworldqa,chartqa,pope"  # dropped: mathvista_testmini (GPT-4 judge), mmbench_en_dev (OpenAI eval aggregator)
echo "[run_pruning_eval] launching lmms-eval ($TASKS_VLM)" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PROJECT/retrofit:$PROJECT/retrofit/eval \
  /home/user01/Minko/reskip2/.venv/bin/python -c "
import sys
sys.path.insert(0, '$PROJECT/retrofit')
sys.path.insert(0, '$PROJECT/retrofit/eval')
import lmms_eval_retrofit  # registers qwen3_vl_pruned
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

# 2. lm-eval on LAMBADA (text-side parity).
echo "[run_pruning_eval] launching lm-eval (lambada_openai,hellaswag)" | tee -a "$LOG"
/home/user01/Minko/reskip2/.venv/bin/python retrofit/eval/run_lm_eval.py \
  --model-path /home/user01/Minko/models/Qwen3-VL-2B \
  --skip-layers "$LAYERS" \
  --tasks lambada_openai,hellaswag \
  --gpu $GPU \
  --output "$OUTROOT/text" \
  --batch-size 8 \
  --label "static_prune_drop_${DROP}" 2>&1 | tee -a "$LOG"

echo "[run_pruning_eval] all done at $(date)" | tee -a "$LOG"
