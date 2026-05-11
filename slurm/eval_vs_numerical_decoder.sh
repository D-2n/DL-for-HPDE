#!/bin/bash
#SBATCH --job-name=eval_dec
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/eval_vs_numerical_decoder_%j.log

#SBATCH --exclude=gpu012
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

RUN_DIR=${1:-}
N_SAMPLES=${2:-50}
N_PLOTS=${3:-3}

if [ -n "$RUN_DIR" ]; then
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/eval_vs_numerical_decoder.py \
        --run-dir "$RUN_DIR" --n_samples "$N_SAMPLES" --n_plots "$N_PLOTS"
else
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/eval_vs_numerical_decoder.py \
        --n_samples "$N_SAMPLES" --n_plots "$N_PLOTS"
fi
