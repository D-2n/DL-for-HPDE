#!/bin/bash
#SBATCH --job-name=arz_datagen_trial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_datagen_trial_%j.log

# POC trial: small N, LWR-shaped data envelope.
# 18 cells (3 families x 6 segments) * 10 samples/cell = 180 samples.
# Matches LWR: nx=128, nt=128, x in [-1,1], t_max=1.0, value range [0.1, 0.9].
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_trial.npz}
N=${2:-180}
TAU=${3:-0.1}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
  --out "$OUT" \
  --N "$N" \
  --nx 128 --nt 128 \
  --x-min -1.0 --x-max 1.0 --t-max 1.0 \
  --tau "$TAU" \
  --families riemann_stratified,piecewise_constant_stratified,piecewise_sine \
  --segments 2,3,5,7,10,25 \
  --rho-min 0.1 --rho-max 0.9 \
  --v-min   0.1 --v-max   0.9 \
  --refine 4 \
  --boundary ghost \
  --seed 0
