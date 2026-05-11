#!/bin/bash
#SBATCH --job-name=hypno_st3_baseline_gelu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_st3_baseline_gelu_%j.log
#SBATCH --requeue

# Auto-resume on preemption:
#   --requeue  tells slurm to re-submit this job after preemption / node fail.
#              CLEPS default is no-requeue, so this is required.
#
# On requeue ($SLURM_RESTART_COUNT > 0) we read the run dir from
# latest_run.txt (written by train_hypno_st3.py right after run_dir creation)
# and pass it as --resume_run, which loads the highest-epoch checkpoint and
# fast-forwards the LR scheduler.

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/logs

CONFIG=hyperbolic_pde/configs/hyperbolic_pde_cleps_baseline_gelu.yaml
LATEST_RUN_FILE=hyperbolic_pde/runs/hypno_st3/latest_run.txt

if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && [ -f "$LATEST_RUN_FILE" ]; then
    RUN_DIR=$(cat "$LATEST_RUN_FILE")
    echo "[requeue ${SLURM_RESTART_COUNT}] resuming from $RUN_DIR"
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_st3.py \
        --config "$CONFIG" --resume_run "$RUN_DIR"
else
    echo "[fresh launch] starting new run"
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_st3.py \
        --config "$CONFIG"
fi
