#!/bin/bash
#SBATCH --job-name=hypno_arz_mark2_1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_mark2_1_%j.log

# Train HypNO-ARZ Mark 2.1 (two-family + ALL jumps in both branches). Identical
# to mark2 except the message feature vector (2d+6 vs 2d+3). The model-section
# name contains 'mark2_1', so --model-variant auto resolves to the Mark2_1 class
# and the (rho,v)-frame loss. Trains in PARALLEL with mark2 as an A/B probe for
# the spurious-3rd-w-band failure. Default data = exact homogeneous Riemann.
#
# Preemption-safe: --requeue (same job ID), --open-mode=append, --auto_resume
# (per-job pointer latest_run_job<ID>.txt) -> resumes the latest checkpoint.
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
DATA_SECTION=${2:-arz_riemann_mark2_prho}
MODEL_SECTION=${3:-hypno_arz_mark2_1}
RESUME=${4:-}

if [ -n "$RESUME" ]; then
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant mark2_1 \
        --resume_run $RESUME
else
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant mark2_1 \
        --auto_resume
fi
