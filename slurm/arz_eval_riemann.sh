#!/bin/bash
#SBATCH --job-name=arz_eval_riemann
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_eval_riemann_%j.log

set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/results /home/dzdrale/scratch/logs

# Usage: sbatch slurm/arz_eval_riemann.sh <run_dir> [ckpt_name] [data] [n_samples] [n_plots]
RUN_DIR=${1:?usage: sbatch arz_eval_riemann.sh <run_dir> [ckpt_name] [data] [n_samples] [n_plots]}
CKPT_NAME=${2:-model_final.pt}
DATA=${3:-/home/dzdrale/scratch/arz_1d/arz_riemann_trial.npz}
N=${4:-20}
N_PLOTS=${5:-10}

CKPT="$RUN_DIR/$CKPT_NAME"
OUTDIR="$RUN_DIR/eval"

if [ ! -f "$CKPT" ]; then
    echo "checkpoint not found: $CKPT" >&2
    ls "$RUN_DIR" >&2
    exit 1
fi
mkdir -p "$OUTDIR"

echo "[arz_eval_riemann] ckpt = $CKPT"
echo "[arz_eval_riemann] data = $DATA"
echo "[arz_eval_riemann] out  = $OUTDIR"

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.eval_vs_numerical_arz \
    --ckpt "$CKPT" \
    --data "$DATA" \
    --baselines weno5,godunov \
    --samples "$N" \
    --out "$OUTDIR/vs_numerical.csv" \
    --figures "$OUTDIR/figs_vs_numerical" \
    --n-plots "$N_PLOTS"
