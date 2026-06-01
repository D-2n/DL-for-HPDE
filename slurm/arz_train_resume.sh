#!/bin/bash
# Preemption-tolerant ARZ training launcher.
#
# Key behaviours:
#   * --requeue: SLURM auto-requeues the job after preemption.
#   * --auto_resume passed to the python trainer: on every (re)start,
#     reads hyperbolic_pde/runs/hypno_arz/latest_run.txt and resumes from
#     the run it points to (continuing from the highest checkpoint_epoch*.pt
#     in that dir). The first invocation of the job creates the run dir and
#     writes latest_run.txt; every subsequent requeued invocation reads it.
#   * --signal=B:USR1@120: SLURM sends SIGUSR1 to the batch script 120s
#     before walltime expiry. We trap it, scancel-requeue ourselves so the
#     pending checkpoint at the end of the next epoch lands cleanly. Without
#     this, jobs hitting --time would lose any post-last-checkpoint progress.
#
# Usage:
#   sbatch slurm/arz_train_resume.sh                       # uses _cleps.yaml
#   sbatch slurm/arz_train_resume.sh <CONFIG_YAML>
#
# The config is used ONLY on the first launch (fresh run). On every requeue
# the trainer reads run_dir/config.yaml instead, so config edits after the
# first launch will NOT take effect -- edit run_dir/config.yaml directly.

#SBATCH --job-name=hypno_arz_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_resume_%j.log
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps.yaml}

echo "=========================================="
echo "SLURM_JOB_ID       = $SLURM_JOB_ID"
echo "SLURM_RESTART_COUNT= ${SLURM_RESTART_COUNT:-0}"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "CONFIG             = $CONFIG"
echo "=========================================="

# Show what latest_run.txt says BEFORE we start, so the log makes it obvious
# whether this invocation is a fresh start or a resume.
LATEST_FILE=hyperbolic_pde/runs/hypno_arz/latest_run.txt
if [ -f "$LATEST_FILE" ]; then
    LATEST_DIR=$(cat "$LATEST_FILE")
    echo "[launcher] latest_run.txt -> $LATEST_DIR"
    if [ -d "$LATEST_DIR" ]; then
        LAST_CKPT=$(ls -v "$LATEST_DIR"/checkpoint_epoch*.pt 2>/dev/null | tail -1)
        if [ -n "$LAST_CKPT" ]; then
            echo "[launcher] last checkpoint: $LAST_CKPT"
        else
            echo "[launcher] no checkpoints in $LATEST_DIR yet (fresh-ish run)"
        fi
    else
        echo "[launcher] latest_run.txt points to a missing directory -- will start fresh"
    fi
else
    echo "[launcher] no latest_run.txt -- this is the first invocation, will start fresh"
fi

# --- Preemption / signal handling ----------------------------------------- #
# On SIGUSR1 from SLURM (sent 120s before walltime ends), forward SIGTERM to
# the python child so PyTorch's atexit handlers run, then requeue this job.
# The next invocation will auto-resume.
_cleanup() {
    echo "[launcher] caught SIGUSR1 at $(date) -- forwarding SIGTERM to python child PID=$PYPID"
    if [ -n "${PYPID:-}" ] && kill -0 "$PYPID" 2>/dev/null; then
        kill -TERM "$PYPID"
        # Give python a chance to flush its log / save best.pt etc.
        wait "$PYPID"
    fi
    echo "[launcher] requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap _cleanup USR1

# --- Launch trainer in the background so the trap can fire --------------- #
/home/dzdrale/hypno_env/bin/python \
    /home/dzdrale/DL-for-HPDE/hyperbolic_pde/scripts/train_hypno_arz.py \
    --config "$CONFIG" \
    --auto_resume &
PYPID=$!
echo "[launcher] python PID=$PYPID"

# `wait` is interrupted by signals, which is what we want.
wait $PYPID
PYEXIT=$?
echo "[launcher] python exited with status $PYEXIT"
exit $PYEXIT
