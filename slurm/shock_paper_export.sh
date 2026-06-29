#!/bin/bash
#SBATCH --job-name=shock_paper
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/shock_paper_export_%j.log

#SBATCH --exclude=gpu012
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

# Detector defaults match shock_comparison.sh (jump=0.06, band=2, TV-gate on,
# multiplier=1.5). Override via passthrough args, e.g.
#   sbatch slurm/shock_paper_export.sh --jump-threshold 0.08 --band-cells 3
RUN_DIR=/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_st3/run_20260518_000640_128_paper
DATA_PATH=/home/dzdrale/scratch/lwr_1d/hyperbolic_dataset_paper_eval.npz
N_PER_GROUP=5

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.scripts.shock_paper_export \
    --run-dir "$RUN_DIR" \
    --data_path "$DATA_PATH" \
    --n_per_group "$N_PER_GROUP" \
    "$@"
