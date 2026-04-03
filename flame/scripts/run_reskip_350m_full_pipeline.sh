#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${REPO_ROOT}/flame"

TOKENIZER_PATH=${TOKENIZER_PATH:?Set TOKENIZER_PATH to a local HF tokenizer directory}
DATA_GLOB=${DATA_GLOB:?Set DATA_GLOB to the training parquet glob, e.g. /data/SlimPajama/train-*.parquet}
TEXT_KEY=${TEXT_KEY:-text}
NNODE=${NNODE:-1}
NGPU=${NGPU:-1}
LOG_RANK=${LOG_RANK:-0}
DEVICE=${DEVICE:-cuda:0}
TASKS=${TASKS:-lambada_openai,hellaswag,arc_easy,arc_challenge}
PROFILE_SAMPLES=${PROFILE_SAMPLES:-128}
LIMIT=${LIMIT:-}
RESULT_ROOT=${RESULT_ROOT:-../outputs/flame_reskip_350m_pipeline}
THRESHOLDS=${THRESHOLDS:-"0.01 0.02 0.05 0.10"}

BASELINE_DIR="${RESULT_ROOT}/baseline_350m"
ATTNRES_DIR="${RESULT_ROOT}/attnres_350m"
EVAL_DIR="${RESULT_ROOT}/evals"
PROFILE_DIR="${RESULT_ROOT}/profiles"
PLOT_DIR="${RESULT_ROOT}/plots"

mkdir -p "${EVAL_DIR}" "${PROFILE_DIR}" "${PLOT_DIR}"

run_train() {
  local output_dir="$1"
  local config_path="$2"
  NNODE=${NNODE} NGPU=${NGPU} LOG_RANK=${LOG_RANK} bash train.sh \
    --job.config_file flame/flame/models/fla.toml \
    --job.dump_folder "${output_dir}" \
    --model.config "${config_path}" \
    --model.tokenizer_path "${TOKENIZER_PATH}" \
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
    --training.data_files "${DATA_GLOB}" \
    --training.num_workers 8 \
    --training.prefetch_factor 2 \
    --training.seed 42 \
    --training.data_parallel_shard_degree -1 \
    --training.tensor_parallel_degree 1 \
    --checkpoint.interval 1000 \
    --checkpoint.load_step -1 \
    --metrics.log_freq 10
}

run_eval() {
  local model_path="$1"
  local scenario="$2"
  local enable_skipping="$3"
  local threshold="$4"
  local eval_out="${EVAL_DIR}/${scenario}.json"
  local profile_out="${PROFILE_DIR}/${scenario}.json"

  LIMIT_ARGS=()
  SKIP_ARGS=()
  if [[ -n "${LIMIT}" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
  fi
  if [[ "${enable_skipping}" == "true" ]]; then
    SKIP_ARGS=(--enable-skipping)
  fi

  python scripts/eval_lm_harness.py \
    --model-path "${model_path}" \
    --output-path "${eval_out}" \
    --scenario "${scenario}" \
    --tasks "${TASKS}" \
    --device "${DEVICE}" \
    --batch-size auto \
    "${LIMIT_ARGS[@]}" \
    "${SKIP_ARGS[@]}" \
    --skip-threshold "${threshold}"

  python scripts/profile_reskip_blocks.py \
    --model-path "${model_path}" \
    --data-glob "${DATA_GLOB}" \
    --output-path "${profile_out}" \
    --text-key "${TEXT_KEY}" \
    --device "${DEVICE}" \
    --max-samples "${PROFILE_SAMPLES}" \
    "${SKIP_ARGS[@]}" \
    --skip-threshold "${threshold}"
}

echo "=== Training baseline 350M ==="
run_train "${BASELINE_DIR}" configs/reskip_baseline_350M.json

echo "=== Training AttnRes 350M ==="
run_train "${ATTNRES_DIR}" configs/reskip_attnres_350M.json

echo "=== Evaluating baseline ==="
run_eval "${BASELINE_DIR}" baseline false 0.0

echo "=== Evaluating AttnRes full-depth ==="
run_eval "${ATTNRES_DIR}" attnres_full false 0.0

for threshold in ${THRESHOLDS}; do
  safe_threshold=${threshold//./p}
  echo "=== Evaluating AttnRes skip threshold ${threshold} ==="
  run_eval "${ATTNRES_DIR}" "attnres_skip_${safe_threshold}" true "${threshold}"
done

echo "=== Plotting benchmark comparisons ==="
python scripts/plot_reskip_benchmark.py \
  --eval-dir "${EVAL_DIR}" \
  --profile-dir "${PROFILE_DIR}" \
  --output-dir "${PLOT_DIR}"

echo "Pipeline complete."
echo "Baseline model: ${BASELINE_DIR}"
echo "AttnRes model: ${ATTNRES_DIR}"
echo "Eval results: ${EVAL_DIR}"
echo "Plots: ${PLOT_DIR}"
