#!/bin/bash
#SBATCH --job-name=hypno_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=parq-gpu001
#SBATCH --exclude=gpu013
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_resume_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_st3.py \
    --config hyperbolic_pde/configs/hyperbolic_pde_cleps_decoder_shared.yaml \
    --resume_run hyperbolic_pde/runs/hypno_st3/run_20260513_140206