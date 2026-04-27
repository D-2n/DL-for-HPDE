#!/bin/bash
#SBATCH --job-name=eval_vs_num
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/dzdrale/scratch/logs/eval_vs_numerical_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/logs

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/eval_vs_numerical.py 
