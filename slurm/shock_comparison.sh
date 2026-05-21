#!/bin/bash
#SBATCH --job-name=shock_compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/shock_comparison_%j.log

#SBATCH --exclude=gpu012
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

# Defaults match the user's intended invocation. Override any of them by
# passing extra args to sbatch (they go through to the python script):
#   sbatch slurm/shock_comparison.sh --jump-threshold 0.10 --band-cells 3
#   sbatch slurm/shock_comparison.sh --no-tv-gate
# Detector defaults (set inside the script):
#   jump=0.06, band=2, TV-gate on (multiplier=1.5).
RUN_DIR=/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_st3/run_20260518_000640_128
DATA_PATH=/home/dzdrale/scratch/lwr_1d/hyperbolic_dataset_ood_grouped.npz
N_PER_GROUP=5

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.scripts.shock_comparison \
    --run-dir "$RUN_DIR" \
    --data_path "$DATA_PATH" \
    --n_per_group "$N_PER_GROUP" \
    "$@"
