#!/bin/bash
#SBATCH --job-name=hypno_arz_mark1_router
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_mark1_router_%j.log

# Train HypNO-ARZ Mark 1 (router-aware): the EXACT pre-split single-MLP
# pure-pairwise model, with the family-router theta added to the adjacent edge
# message (2d+4 vs 2d+3). NO GNL/LD two-family split. Mark1 frame: (rho,w)
# decoder + arz_total_loss. Same recipe as mark2/mark2_1 (kx7/kt5, 12 layers,
# d96, batch 8, mae) so it is a clean A/B on the 2000-sample exact-Riemann set.
#
# The model-section name contains 'mark1_router', so --model-variant auto would
# resolve correctly; we pass it explicitly for clarity.
#
# Preemption-safe: --requeue (same job ID), --open-mode=append, --auto_resume
# (per-job pointer latest_run_job<ID>.txt) -> resumes the latest checkpoint.
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
DATA_SECTION=${2:-arz_riemann_mark2_prho}
MODEL_SECTION=${3:-hypno_arz_mark1_router}
RESUME=${4:-}

if [ -n "$RESUME" ]; then
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant mark1_router \
        --resume_run $RESUME
else
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant mark1_router \
        --auto_resume
fi
