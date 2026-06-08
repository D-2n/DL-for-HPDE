#!/bin/bash
#SBATCH --job-name=paper_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/paper_eval_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

# Usage: sbatch slurm/paper_eval.sh <run-dir> [extra args...]
# <run-dir> must contain config.yaml and model_final.pt.
# Requires the paper_eval_data dataset (generate first with:
#   sbatch slurm/generate_ood_data.sh paper_eval_data
# )
#
# The FNO baseline is included by default using the trained checkpoint below.
# Override by passing --fno-weights <path> in [extra args], or drop FNO with
# --no-fno. Extra args after <run-dir> are forwarded verbatim and win over the
# default --fno-weights (argparse keeps the last value).
RUN_DIR=${1:?"pass the run directory as the first argument"}
shift

FNO_WEIGHTS=/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/fno_epoch200.pt
# Older runs (e.g. run_20260518_*) have a config.yaml that predates the
# paper_eval_data block, so the dataset must be passed explicitly. Pass a
# different --data_path in [extra args] to override (argparse keeps the last).
DATA_PATH=/home/dzdrale/scratch/lwr_1d/hyperbolic_dataset_paper_eval.npz

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/paper_eval.py \
    --run-dir "$RUN_DIR" \
    --data_path "$DATA_PATH" \
    --fno-weights "$FNO_WEIGHTS" "$@"
