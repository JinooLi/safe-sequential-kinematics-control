#!/bin/bash
#SBATCH --job-name=panda-link-groups
#SBATCH --array=0-2
#SBATCH --partition=gpu1,gpu2,gpu6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=panda-link-groups-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/panda-matplotlib-${SLURM_JOB_ID}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
dataset_args=(
  --experiment configs/experiments/panda_link_dataset.yaml
  --output "${PANDA_DATASET_OUTPUT:-outputs/panda_link_dataset/production-groups-${SLURM_ARRAY_JOB_ID}}/group-${SLURM_ARRAY_TASK_ID}"
  --total-contexts "${PANDA_DATASET_CONTEXTS:-1000}" --batch-size "${PANDA_DATASET_BATCH_SIZE:-50}" --seed "${PANDA_DATASET_SEED:-2026101700}"
  --groups "${SLURM_ARRAY_TASK_COUNT}" --group-index "${SLURM_ARRAY_TASK_ID}"
  --gpus 4 --cpus 25 --workers-per-gpu 4 --prefetch-ik --ik-context-workers 2
)
if [[ "${PANDA_DATASET_RESUME:-0}" == "1" ]]; then
  dataset_args+=(--resume)
fi
.venv/bin/python -m safe_ik.application.panda_link_dataset_campaign "${dataset_args[@]}"
