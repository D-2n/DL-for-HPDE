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
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/results /home/dzdrale/scratch/logs

CKPT=${1:-/home/dzdrale/scratch/runs/arz_st3/best.pt}
DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_eval.npz}
OUTDIR=${3:-/home/dzdrale/scratch/results}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.eval_vs_numerical_arz \
  --ckpt "$CKPT" --data "$DATA" \
  --baselines weno5,godunov --samples 20 \
  --out "$OUTDIR/arz_vs_numerical.csv"

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.eval_shock_arz \
  --ckpt "$CKPT" --data "$DATA" \
  --baselines weno5,godunov --samples 20 \
  --tau-shock 0.06 --band-halfwidth 2 --tv-mult 1.5 \
  --out "$OUTDIR/arz_shock.csv"
