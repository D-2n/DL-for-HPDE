#!/bin/bash
#SBATCH --job-name=arz_datagen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_datagen_%j.log

set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_train.npz}
N=${2:-5400}
TAU=${3:-0.1}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
  --out "$OUT" \
  --N "$N" \
  --nx 256 --nt 128 --t-max 0.3 \
  --tau "$TAU" \
  --families riemann_stratified,piecewise_constant_stratified,piecewise_sine \
  --segments 2,3,5,7,8,10 \
  --refine 8 \
  --boundary ghost \
  --seed 0
