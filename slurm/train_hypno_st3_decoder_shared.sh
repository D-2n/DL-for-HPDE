#!/bin/bash
#SBATCH --job-name=hypno_st3_dec
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_st3_decoder_shared_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/logs

CONFIG=hyperbolic_pde/configs/hyperbolic_pde_cleps_decoder_shared.yaml

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_st3.py --config $CONFIG
