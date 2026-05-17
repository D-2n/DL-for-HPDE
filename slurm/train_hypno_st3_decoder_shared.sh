#!/bin/bash
#SBATCH --job-name=hypno_st3_dec
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_st3_decoder_shared_%j.log
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@90
#SBATCH --open-mode=append
#SBATCH --exclude=parq-gpu001
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

CONFIG=hyperbolic_pde/configs/hyperbolic_pde_cleps_decoder_shared.yaml

# Per-job marker file; survives across requeues for THIS slurm script only.
MARKER=/home/dzdrale/scratch/runs/hypno_st3_decoder_shared.run_dir

# --- resume detection ---
# Resume is OPT-IN: pass `--resume` as the first sbatch arg to enable.
# A bare `sbatch slurm/train_hypno_st3_decoder_shared.sh` always starts fresh
# and clears any stale marker from a previous run.
# Slurm requeues (SIGTERM -> requeue) re-exec the script with the same args,
# so a job started with `--resume` will continue to auto-resume across requeues.
RESUME_ARG=""
if [ "${1:-}" = "--resume" ]; then
    if [ -f "$MARKER" ]; then
        RUN_DIR=$(tr -d '[:space:]' < "$MARKER")
        if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ] && ls "$RUN_DIR"/checkpoint_epoch*.pt >/dev/null 2>&1; then
            RESUME_ARG="--resume_run $RUN_DIR"
            echo "[slurm] --resume requested; auto-resume from $RUN_DIR (marker: $MARKER)"
        else
            echo "[slurm] --resume requested but marker invalid or no checkpoints; starting fresh"
            rm -f "$MARKER"
        fi
    else
        echo "[slurm] --resume requested but no marker found; starting fresh"
    fi
else
    if [ -f "$MARKER" ]; then
        echo "[slurm] clearing stale marker $MARKER (pass --resume to resume)"
        rm -f "$MARKER"
    fi
    echo "[slurm] starting fresh run"
fi

# Record job start time BEFORE launching python so we can identify our own
# run dir even if a parallel HypNO-ST3 run with the same save_path exists.
# Any run dir created at or after this time is ours; older ones belong to
# other jobs (e.g. the 7-layer ablation running in parallel).
T_START=$(($(date +%s) - 5))

# --- launch training in background ---
/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_st3.py \
    --config $CONFIG $RESUME_ARG &
PYPID=$!

# If we started fresh, identify OUR run dir by mtime — it must have been
# created at or after T_START (so it's not from a pre-existing parallel run).
# We don't rely on save_path matching because parallel configs may have the
# same save_path. Pick the newest dir whose mtime is >= T_START.
if [ -z "$RESUME_ARG" ]; then
    (
        RUNS_BASE="hyperbolic_pde/runs/hypno_st3"
        for i in $(seq 1 120); do
            # Find run_* dirs created since T_START, newest first.
            CANDIDATE=$(find "$RUNS_BASE" -maxdepth 1 -mindepth 1 -type d \
                       -newermt "@$T_START" -name 'run_*' 2>/dev/null | sort -r | head -n 1)
            if [ -n "$CANDIDATE" ] && [ -d "$CANDIDATE" ]; then
                echo "$CANDIDATE" > "$MARKER"
                echo "[slurm] pinned run dir to marker (by mtime >= $T_START): $CANDIDATE -> $MARKER"
                exit 0
            fi
            sleep 1
        done
        echo "[slurm] WARNING: could not identify run dir within 120s"
    ) &
fi

# --- signal forwarding for graceful shutdown ---
term_handler() {
    echo "[slurm] received SIGTERM, forwarding to python (pid $PYPID) and waiting"
    kill -TERM "$PYPID" 2>/dev/null
    wait "$PYPID"
}
trap term_handler SIGTERM

wait "$PYPID"
