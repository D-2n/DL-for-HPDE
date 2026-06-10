#!/bin/bash
#SBATCH --job-name=arz_gen_mark2_data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_mark2_data_%j.log

# Generate the HypNO-ARZ Mark 2 training data: 2000 exact homogeneous (tau=inf)
# Riemann samples, p=rho, evaluated at cell midpoints (no FVM).
#
# Vacuum-free by construction: the generator samples rho_L, rho_*, rho_R in
# [rho-min, rho-max] and v_* directly, deriving v_L from w-preservation -- so
# every density state is in range and the intermediate state rho_* is never 0
# (no vacuum, no over-jam). rho-max is 0.95 per the Mark2 data decision.
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_riemann_mark2_prho.npz}
N=${2:-2000}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
  --out "$OUT" \
  --N "$N" \
  --nx 128 --nt 128 \
  --x-min -1.0 --x-max 1.0 --t-max 1.0 \
  --exact-riemann-only \
  --pressure-form rho \
  --seed 42 \
  --rho-min 0.1 --rho-max 0.95 \
  --v-min 0.1 --v-max 0.9

echo "Done: $OUT"
