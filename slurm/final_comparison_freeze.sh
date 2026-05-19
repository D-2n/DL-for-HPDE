#!/bin/bash
#SBATCH --job-name=freeze_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/final_comparison_freeze_%j.log

#SBATCH --exclude=gpu012
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

# Usage: sbatch slurm/final_comparison_freeze.sh <run-dir> <data_path> [extra args]
#   run-dir   : run directory with config.yaml + model_final.pt
#   data_path : single-shock OOD .npz (e.g. hyperbolic_dataset_ood_oneshock.npz)
RUN_DIR=${1:?"pass the run directory as the first argument"}
DATA_PATH=${2:?"pass the single-shock OOD .npz path as the second argument"}
shift 2

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/final_comparison_freeze.py \
    --run-dir "$RUN_DIR" --data_path "$DATA_PATH" "$@"
