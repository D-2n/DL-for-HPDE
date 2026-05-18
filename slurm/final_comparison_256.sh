#!/bin/bash
#SBATCH --job-name=final_cmp_256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/final_comparison_256_%j.log

#SBATCH --exclude=gpu012
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

# Usage: sbatch slurm/final_comparison_256.sh <run-dir> [block] [extra args...]
#   run-dir : run directory with config.yaml and model_final.pt
#   block   : ood_data_256_wide (default) or ood_data_256_train
RUN_DIR=${1:?"pass the run directory as the first argument"}
BLOCK=${2:-ood_data_256_wide}
shift 2 2>/dev/null || shift

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/final_comparison_256.py \
    --run-dir "$RUN_DIR" --block "$BLOCK" "$@"
