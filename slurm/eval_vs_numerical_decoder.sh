#!/bin/bash
#SBATCH --job-name=eval_dec
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/eval_vs_numerical_decoder_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/eval_vs_numerical_decoder.py
