#!/bin/bash
#SBATCH --job-name=arz_gen_riemann_1k
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_riemann_1k_%j.log

# Generate 1000 exact-Riemann ARZ samples (same grid/tau/p-form as wft set).
# Used to augment arz_mixed_wft_prho.npz with Riemann structure.
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_riemann_1k.npz}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
    --out "$OUT" \
    --N 1000 \
    --nx 128 --nt 128 \
    --x-min -1.0 --x-max 1.0 --t-max 1.0 \
    --tau inf \
    --families riemann_stratified \
    --segments 2 \
    --use-exact-riemann \
    --n-jump-bins 8 \
    --num-workers 8 \
    --cfl 0.4 \
    --boundary ghost \
    --pressure-form rho \
    --rho-min 0.1 --rho-max 0.9 \
    --v-min 0.0 --v-max 1.0 \
    --seed 99

echo "[arz_gen_riemann_1k] done: $OUT"
