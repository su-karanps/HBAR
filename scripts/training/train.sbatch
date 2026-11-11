#!/bin/bash
# Generic Slurm launcher for training any tokenizer config on the subject-level split.
# Driven by env vars:
#   MODEL_CONFIG   path to model YAML (required), e.g. configs/fmri/hbar52_fixed.yaml
#   OUT_DIR        checkpoint/metrics directory (required)
#   SEED           training seed (default 0); does NOT affect the data split
#   DATA_CONFIG    data YAML (default: configs/fmri/data.yaml)
#   NPROC          GPUs to use (default: however many Slurm allocated)
#   GRAD_ACCUM     override train.grad_accum_steps. Keep the effective batch size
#                  (batch_size x grad_accum x n_gpu) at 32: the configs set batch 4 and
#                  accum 4 (2 GPUs); use GRAD_ACCUM=8 on 1 GPU or GRAD_ACCUM=2 on 4 GPUs.
#   RESUME         set to 1 to resume from the newest checkpoint in OUT_DIR (no-op if none)
#   FMRI_DATA_DIR  if set, pre-read its .npy files into the OS page cache before training
#
# Example:
#   MODEL_CONFIG=configs/fmri/hbar52_fixed.yaml OUT_DIR=/path/to/runs/hbar52_fixed_seed0 \
#     sbatch scripts/training/train.sbatch
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=2
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=hbar-train
#SBATCH --output=logs/hbar-train_%j.out
#SBATCH --error=logs/hbar-train_%j.err

set -euo pipefail

nvidia-smi

PROJECT_DIR="${HBAR_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_CONFIG=${DATA_CONFIG:-"configs/fmri/data.yaml"}
SEED=${SEED:-0}
: "${MODEL_CONFIG:?set MODEL_CONFIG}"
: "${OUT_DIR:?set OUT_DIR}"

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

if [[ -n "${FMRI_DATA_DIR:-}" ]]; then
  echo "[cache warmup] pre-reading fMRI data into OS page cache"
  WARMUP_START=$(date +%s)
  find "${FMRI_DATA_DIR}" -name "*.npy" -print0 \
    | xargs -0 -P 32 -n 8 cat > /dev/null
  echo "[cache warmup] done in $(( $(date +%s) - WARMUP_START ))s"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$(basename "${OUT_DIR}")}

NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}

EXTRA_ARGS=()
if [[ -n "${GRAD_ACCUM:-}" ]]; then
  EXTRA_ARGS+=(--grad_accum_steps "${GRAD_ACCUM}")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi

echo "[run] model_config=${MODEL_CONFIG}"
echo "[run] data_config=${DATA_CONFIG}"
echo "[run] out_dir=${OUT_DIR}  seed=${SEED}  nproc=${NPROC}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" train_fmri.py \
  --config "${MODEL_CONFIG}" \
  --data_config "${DATA_CONFIG}" \
  --output_dir "${OUT_DIR}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
