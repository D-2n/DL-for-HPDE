#!/bin/bash
#SBATCH --job-name=arz_train_trial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_train_trial_%j.log

# POC trial: small model, short epochs.
# Architecture matched to LWR hypno_st3 trial-scale: d_latent=64, depth=6.
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

DATA=${1:-/home/dzdrale/scratch/arz_1d/arz_trial.npz}
CKPT=${2:-/home/dzdrale/scratch/runs/arz_trial}
RESUME=${3:-}

if [ -n "$RESUME" ]; then
  RESUME_FLAG="--resume"
else
  RESUME_FLAG=""
fi

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.train_arz \
  --data "$DATA" \
  --ckpt "$CKPT" \
  --kx 2 --kt 2 \
  --d-latent 64 --d-hidden 96 --depth 6 --decoder-depth 3 \
  --lambda-state 1.0 --lambda-cons 0.1 --lambda-balance 0.1 --lambda-probe 0.1 \
  --epochs 30 --batch-size 8 --lr 3e-4 --seed 0 --amp \
  $RESUME_FLAG
