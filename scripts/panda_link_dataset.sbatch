#!/bin/bash
#SBATCH --job-name=panda-link-pilot
#SBATCH --partition=gpu1,gpu2,gpu6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=panda-link-pilot-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/panda-matplotlib-${SLURM_JOB_ID}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m safe_ik.application.panda_link_dataset \
  --experiment configs/experiments/panda_link_dataset.yaml \
  --output "outputs/panda_link_dataset/pilot-${SLURM_JOB_ID}" \
  --gpus 4 --cpus 15
