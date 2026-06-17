#!/bin/bash
#SBATCH --job-name=hypno_arz_mark1_normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_mark1_normal_%j.log

# Train HypNO-ARZ Mark 1 (NORMAL, current 2d+3 pure-pairwise HypNO_ARZ) on the
# big 5400-sample exact-Riemann set. Homogeneous (tau=inf), p=rho, (rho,w)
# decoder + arz_total_loss. The model-section name (hypno_arz_mark1_normal) has
# no 'mark2'/'mark1_router', so --model-variant auto resolves to the mark1 class;
# passed explicitly here for clarity.
#
# Preemption-safe: --requeue (same job ID), --open-mode=append, --auto_resume.
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
DATA_SECTION=${2:-arz_riemann_strat_prho}
MODEL_SECTION=${3:-hypno_arz_mark1_normal}
RESUME=${4:-}

if [ -n "$RESUME" ]; then
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant mark1 \
        --resume_run $RESUME
else
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant mark1 \
        --auto_resume
fi
