#!/bin/bash
#SBATCH --job-name=ft_strat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/finetune_stratified_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

# Fine-tune the 12L decoder-shared model on the stratified dataset.
# Uses resume_from (set inside the config) -> loads weights only, fresh run dir
# and fresh cosine schedule. Do NOT pass --resume_run here.
#
# Prerequisite: hyperbolic_dataset_stratified.npz must already exist
#   (sbatch slurm/generate_data.sh).
CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_cleps_finetune_stratified.yaml}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_st3.py --config "$CONFIG"
