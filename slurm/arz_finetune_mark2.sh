#!/bin/bash
#SBATCH --job-name=hypno_arz_mark2_finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_mark2_finetune_%j.log

# FINETUNE a pretrained Mark 2 checkpoint on the stratified multi-discontinuity
# HOMOGENEOUS set (arz_stratified_homog_prho), using --finetune_from: loads ONLY
# the weights into a NEW run dir with a fresh optimizer and the gentle
# hypno_arz_mark2_finetune schedule (low lr, no warmup). Does NOT touch the
# source run. epoch counter restarts at 1 (the new section's `epochs` is the cap).
#
# $1 CKPT  = the pretrained mark2 checkpoint to warm-start from (REQUIRED).
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CKPT=${1:?usage: sbatch arz_finetune_mark2.sh <pretrained_mark2_checkpoint.pt> [config] [data_section] [model_section]}
CONFIG=${2:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
DATA_SECTION=${3:-arz_stratified_homog_prho}
MODEL_SECTION=${4:-hypno_arz_mark2_finetune}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
    --config $CONFIG \
    --data-section $DATA_SECTION \
    --model-section $MODEL_SECTION \
    --model-variant mark2 \
    --finetune_from "$CKPT"
