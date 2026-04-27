#!/bin/bash
#SBATCH --job-name=hypno_st3_mlp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_st3_mlp_%j.log
#SBATCH --constraint=256go
#SBATCH --exclude=gpu012
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/logs

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_st3.py \
    --config hyperbolic_pde/configs/hyperbolic_pde_cleps_mlp.yaml
