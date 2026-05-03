#!/bin/bash
# Queue LIBERO 4-suite eval with multiple seeds for paper seed-repeat ablation.
# Runs the same ckpt at seeds {1,2,3} (additive on top of existing seed-7 baseline)
# across 4 LIBERO suites. One policy server per ckpt (re-used across all 12 evals).
#
# Usage:
#   bash queue_libero_seed_repeat.sh <pair_label> <server_gpu> <render_gpu> <port> <ckpt_dir> [<ckpt_dir2> ...]
#
# Each ckpt_dir is the leaf directory under starVLA/results/Checkpoints/.
# Output goes under retrofit/outputs/libero_seed_repeat/<ckpt_dir>_seed<N>_<suite>/.
set -u
PAIR="${1:?usage: <pair> <server_gpu> <render_gpu> <port> <ckpt> [<ckpt>..]}"
SERVER_GPU="${2:?}"
RENDER_GPU="${3:?}"
PORT="${4:?}"
shift 4
CKPTS=("$@")

PROJECT=/home/user01/Minko/reskip2/reskip
STARVLA=$PROJECT/starVLA
OUT_ROOT=$PROJECT/retrofit/outputs/libero_seed_repeat
PYBIN=/home/user01/Minko/reskip2/.venv/bin/python
SEEDS=(1 2 3)
SUITES=(libero_spatial libero_object libero_goal libero_10)

mkdir -p "$OUT_ROOT"

cd "$STARVLA"

echo "[$PAIR] starting at $(date) — pair=$PAIR server_gpu=$SERVER_GPU render_gpu=$RENDER_GPU port=$PORT"
echo "[$PAIR] ckpts: ${CKPTS[*]}"

for CKPT_DIR in "${CKPTS[@]}"; do
  CKPT="$STARVLA/results/Checkpoints/$CKPT_DIR/final_model/pytorch_model.pt"
  if [ ! -f "$CKPT" ]; then
    echo "[$PAIR] MISSING ckpt: $CKPT — skipping"
    continue
  fi

  SERVER_LOG="$OUT_ROOT/${PAIR}_${CKPT_DIR}_server.log"
  : > "$SERVER_LOG"
  echo "[$PAIR] === ckpt $CKPT_DIR — starting policy server on GPU $SERVER_GPU port $PORT at $(date) ==="
  CUDA_VISIBLE_DEVICES=$SERVER_GPU WANDB_MODE=offline PYTHONPATH=$(pwd) \
    "$PYBIN" deployment/model_server/server_policy.py \
      --ckpt_path "$CKPT" --port "$PORT" --use_bf16 \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "[$PAIR] server pid=$SERVER_PID"

  # Wait for server (up to 5 min)
  ready=0
  for i in $(seq 1 60); do
    if grep -q "server listening" "$SERVER_LOG" 2>/dev/null; then
      ready=1; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "[$PAIR] server died early — see $SERVER_LOG"
      break
    fi
    sleep 5
  done
  if [ "$ready" != "1" ]; then
    echo "[$PAIR] server failed to start for $CKPT_DIR; skipping"
    kill -9 $SERVER_PID 2>/dev/null
    sleep 5
    continue
  fi
  echo "[$PAIR] server listening for $CKPT_DIR"

  for SEED in "${SEEDS[@]}"; do
    for SUITE in "${SUITES[@]}"; do
      LABEL="${CKPT_DIR}_seed${SEED}_${SUITE}"
      OUT="$OUT_ROOT/$LABEL"
      mkdir -p "$OUT"
      LOG="$OUT/eval.log"
      if grep -q "Total success rate" "$LOG" 2>/dev/null; then
        SR=$(grep -oE "Total success rate: [0-9.]+" "$LOG" | tail -1)
        echo "[$PAIR] SKIP $LABEL already done ($SR)"
        continue
      fi
      echo "[$PAIR] === $LABEL begin ($(date)) ==="
      CUDA_VISIBLE_DEVICES=$RENDER_GPU \
        EGL_DEVICE_ID=$RENDER_GPU \
        SEED=$SEED \
        ENABLE_SKIPPING=0 \
        bash examples/LIBERO/eval_files/eval_libero.sh \
        "$CKPT" "$SUITE" "$PORT" 50 \
        > "$LOG" 2>&1
      SR=$(grep -oE "Total success rate: [0-9.]+" "$LOG" | tail -1)
      echo "[$PAIR] === $LABEL done ($(date)): $SR ==="
    done
  done

  echo "[$PAIR] killing server pid=$SERVER_PID for $CKPT_DIR"
  kill -9 $SERVER_PID 2>/dev/null
  sleep 10  # let GPU/port clear
done

echo "[$PAIR] ALL CKPTS DONE at $(date)"
