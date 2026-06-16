#!/bin/bash
#SBATCH --job-name=arz_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_eval_%j.log

set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/results /home/dzdrale/scratch/logs

CKPT=${1:-/home/dzdrale/scratch/runs/arz_st3/best.pt}
DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_eval.npz}
OUTDIR=${3:-/home/dzdrale/scratch/results}
N_PLOTS=${4:-5}
# Bare runs/*.pt checkpoints have no sibling config.yaml to auto-locate, so pass
# the repo config explicitly as $5 (the loader picks the section by node-MLP
# channel count: 7 -> hypno_arz_riemann, 9 -> hypno_arz).
CONFIG=${5:-}

CONFIG_ARG=()
if [ -n "$CONFIG" ]; then
    CONFIG_ARG=(--config "$CONFIG")
fi

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.eval_vs_numerical_arz \
  --ckpt "$CKPT" --data "$DATA" "${CONFIG_ARG[@]}" \
  --baselines weno5,godunov --samples 20 \
  --out "$OUTDIR/arz_vs_numerical.csv" \
  --figures "$OUTDIR/figs_vs_numerical" --n-plots "$N_PLOTS"

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.eval_shock_arz \
  --ckpt "$CKPT" --data "$DATA" "${CONFIG_ARG[@]}" \
  --baselines weno5,godunov --samples 20 \
  --tau-shock 0.06 --band-halfwidth 2 --tv-mult 1.5 \
  --out "$OUTDIR/arz_shock.csv" \
  --figures "$OUTDIR/figs_shock" --n-plots "$N_PLOTS"
