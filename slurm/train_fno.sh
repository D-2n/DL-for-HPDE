#!/bin/bash
#SBATCH --job-name=fno
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/fno_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_cleps.yaml}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_fno.py --config $CONFIG
