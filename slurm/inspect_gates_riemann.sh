#!/bin/bash
#SBATCH --job-name=inspect_gates
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/inspect_gates_%j.log

#SBATCH --exclude=gpu012
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

RUN_DIR=${1:?"usage: sbatch inspect_gates_riemann.sh <run_dir> [u_R] [query_x] [times...]"}
U_R=${2:-0.2}
QUERY_X=${3:-0.5}
shift 3 2>/dev/null || true
TIMES="${*:-0.25 0.5 0.75 1.0}"

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/inspect_gates_riemann.py \
    --run-dir "$RUN_DIR" \
    --u_R "$U_R" \
    --query_x "$QUERY_X" \
    --times $TIMES
