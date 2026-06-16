#!/bin/bash
#SBATCH --job-name=hypno_arz_orig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_orig_%j.log

# Train the ORIGINAL (first) HypNO-ARZ -- the frozen pre-pure-pairwise 2d+12
# HypNO_ARZ_Orig (model_arz_orig, snapshot of commit 11ae6bc) -- on the
# stratified multi-discontinuity HOMOGENEOUS set (arz_stratified_homog_prho).
# Mark1 frame: (rho,w) decoder + arz_total_loss. The model-section name contains
# 'orig', so --model-variant auto resolves to HypNO_ARZ_Orig; passed explicitly.
#
# Preemption-safe: --requeue (same job ID), --open-mode=append, --auto_resume.
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
DATA_SECTION=${2:-arz_stratified_homog_prho}
MODEL_SECTION=${3:-hypno_arz_orig}
RESUME=${4:-}

if [ -n "$RESUME" ]; then
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant orig \
        --resume_run $RESUME
else
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant orig \
        --auto_resume
fi
