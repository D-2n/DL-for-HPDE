#!/bin/bash
#SBATCH --job-name=eval_fno_vs_num
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/eval_fno_vs_numerical_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_cleps_fno.yaml}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/eval_fno_vs_numerical.py --config $CONFIG
