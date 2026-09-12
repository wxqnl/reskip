#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: run_corrected_rsm_vla_cell.sh <scale:2b|4b> <visible_gpus> <port> <steps>" >&2
  exit 2
fi

scale=$1
visible_gpus=$2
port=$3
steps=$4

IFS=',' read -r -a gpu_ids <<<"${visible_gpus}"
if [[ ${#gpu_ids[@]} -ne 2 ]] || [[ "${gpu_ids[0]}" == "${gpu_ids[1]}" ]]; then
  echo "corrected formal cells require exactly two distinct GPUs" >&2
  exit 2
fi
for gpu_id in "${gpu_ids[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-6]$ ]]; then
    echo "GPU ${gpu_id} is outside the allowed node42 set 0..6; GPU 7 is forbidden" >&2
    exit 2
  fi
done

root=/data/Minko
repo=${root}/starVLA_workspace/reskip_ar_v2_skip_consistent_20260820/starVLA
python_env=${root}/starVLA_workspace/envs/starVLA
experiment=${root}/experiments/attnres_rsm_vla_transfer_fix_20260903
source_experiment=${root}/experiments/attnres_rsm_vla_paper_refresh_20260902
data_root=${root}/datasets/starvla_libero_lerobot_20260902
accelerate_config=${source_experiment}/protocol/accelerate_zero2_ga2.yaml
base_config=${repo}/examples/LIBERO/train_files/starvla_cotrain_libero_attnres.yaml

case "${scale}" in
  2b)
    model=${root}/models/Qwen3-VL-2B-Instruct
    n_blocks=7
    init_state=${experiment}/init/qwen3vl_2b_attnres_plus_identity_rsm.pt
    ;;
  4b)
    model=${root}/models/Qwen3-VL-4B-Instruct
    n_blocks=9
    init_state=${experiment}/init/qwen3vl_4b_attnres_plus_identity_rsm.pt
    ;;
  *)
    echo "unknown scale: ${scale}" >&2
    exit 2
    ;;
esac

if [[ ! -s "${init_state}" ]]; then
  echo "missing corrected initialization: ${init_state}" >&2
  exit 2
fi

run_id=${scale}_rsm_v2_identity_transfer_seed42_${steps}steps
run_root=${experiment}/train
run_dir=${run_root}/${run_id}
if [[ -e "${run_dir}" ]]; then
  echo "refusing to overwrite training directory: ${run_dir}" >&2
  exit 1
fi
mkdir -p "${run_dir}"

command=(
  "${python_env}/bin/accelerate" launch
  --config_file "${accelerate_config}"
  --num_processes 2
  --main_process_port "${port}"
  starVLA/training/train_starvla.py
  --config_yaml "${base_config}"
  --seed 42
  --framework.name QwenOFT
  --framework.qwenvl.base_vlm "${model}"
  --framework.qwenvl.attn_implementation flash_attention_2
  --framework.attnres.enabled True
  --framework.attnres.n_blocks "${n_blocks}"
  --framework.attnres.adapter_rank 256
  --framework.attnres.init_state_path "${init_state}"
  --framework.attnres.residual_strength_gate True
  --framework.attnres.residual_strength_gate_scale 0.5
  --framework.attnres.gamma_ramp_steps 9000
  --framework.attnres.gamma_target 1.0
  --framework.attnres.enable_skipping False
  --framework.attnres.skip_mode none
  --datasets.vla_data.data_root_dir "${data_root}"
  --datasets.vla_data.data_mix libero_all
  --datasets.vla_data.per_device_batch_size 8
  --datasets.vla_data.video_backend torchvision_av
  --trainer.freeze_modules qwen_vl_interface.model.model.visual
  --trainer.learning_rate.attnres_adapter 0.0001
  --trainer.gradient_accumulation_steps 2
  --trainer.max_train_steps "${steps}"
  --trainer.save_interval 5000
  --trainer.logging_frequency 10
  --trainer.eval_interval 1000000
  --run_root_dir "${run_root}"
  --run_id "${run_id}"
  --wandb_project AttnRes_RSM_LIBERO_TransferFix
  --wandb_entity reskip
)

printf '%q ' "${command[@]}" >"${run_dir}/exact_command.txt"
printf '\n' >>"${run_dir}/exact_command.txt"
cp "$0" "${run_dir}/run_corrected_rsm_vla_cell.sh"
cp "${experiment}/protocol/PREREGISTRATION.json" "${run_dir}/PREREGISTRATION.json"
cp "${repo}/src/starvla_integration.py" "${run_dir}/starvla_integration.py"
cp "${repo}/starVLA/training/train_starvla.py" "${run_dir}/train_starvla.py"

cd "${repo}"
export PATH=${python_env}/bin:${PATH}
export PYTHONPATH=${repo}
export CUDA_VISIBLE_DEVICES=${visible_gpus}
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export STARVLA_GRADIENT_ACCUMULATION_STEPS=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "${command[@]}"
