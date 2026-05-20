#!/bin/bash
#SBATCH --job-name=paper_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/paper_eval_%j.log

#SBATCH --exclude=gpu012
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

# Usage: sbatch slurm/paper_eval.sh <run-dir> [extra args...]
# <run-dir> must contain config.yaml and model_final.pt.
# Requires the paper_eval_data dataset (generate first with:
#   sbatch slurm/generate_ood_data.sh paper_eval_data
# )
RUN_DIR=${1:?"pass the run directory as the first argument"}
shift

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/paper_eval.py \
    --run-dir "$RUN_DIR" "$@"
