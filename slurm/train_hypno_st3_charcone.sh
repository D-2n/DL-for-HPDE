#!/bin/bash
#SBATCH --job-name=hypno_st3_charcone
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_st3_charcone_%j.log

#SBATCH --exclude=gpu012
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde.yaml}
RESUME=${2:-}

if [ -n "$RESUME" ]; then
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_st3_charcone.py --config $CONFIG --resume_run $RESUME
else
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_st3_charcone.py --config $CONFIG
fi
