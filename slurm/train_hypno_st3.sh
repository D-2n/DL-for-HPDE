#!/bin/bash
#SBATCH --job-name=hypno_st3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_st3_%j.log

source /home/dzdrale/hypno_env/bin/activate
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/logs

python hyperbolic_pde/scripts/train_hypno_st3.py
