#!/bin/bash
#SBATCH --job-name=hypno_arz_mark2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_mark2_%j.log

# Train HypNO-ARZ Mark 2 (two-family, homogeneous ARZ). The model-section name
# contains 'mark2', so --model-variant auto resolves to the Mark2 class and the
# (rho,v)-frame loss. Default data = exact homogeneous Riemann (tau=inf).
#
# Preemption-safe: --requeue puts the job back in the queue when preempted
# (same job ID), --open-mode=append keeps the log, and --auto_resume makes the
# rerun pick up its own run dir (per-job pointer latest_run_job<ID>.txt) and
# continue from the latest checkpoint_epoch*.pt.
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
DATA_SECTION=${2:-arz_riemann_strat_prho}
MODEL_SECTION=${3:-hypno_arz_mark2}
RESUME=${4:-}

if [ -n "$RESUME" ]; then
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant mark2 \
        --resume_run $RESUME
else
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant mark2 \
        --auto_resume
fi
