#!/bin/bash
#SBATCH --job-name=eval_st3_2d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=96G
#SBATCH --time=0:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/eval_st3_2d_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/logs

# $1 = run directory (reads config.yaml + weights from it).
RUN_DIR=${1:?usage: sbatch eval_vs_numerical_2d.sh <run_dir>}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/eval_vs_numerical_2d.py --run-dir $RUN_DIR
