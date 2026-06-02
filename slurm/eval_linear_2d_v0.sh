#!/bin/bash
#SBATCH --job-name=eval_lin2d_v0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/eval_lin2d_v0_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/logs

# $1 = run directory (reads config.yaml + weights from it).
RUN_DIR=${1:?usage: sbatch eval_linear_2d_v0.sh <run_dir>}
N_PLOTS=${2:-4}

/home/dzdrale/hypno_env/bin/python -u hyperbolic_pde/scripts/eval_vs_numerical_linear_2d_v0.py \
    --run-dir "$RUN_DIR" --eval-plots "$N_PLOTS"
