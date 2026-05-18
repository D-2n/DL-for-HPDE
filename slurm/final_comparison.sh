#!/bin/bash
#SBATCH --job-name=final_compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/final_comparison_%j.log

#SBATCH --exclude=gpu012
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

# Usage: sbatch slurm/final_comparison.sh <run-dir> [extra args...]
# <run-dir> must contain config.yaml and model_final.pt.
RUN_DIR=${1:?"pass the run directory as the first argument"}
shift

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/final_comparison.py \
    --run-dir "$RUN_DIR" "$@"
