#!/bin/bash
#SBATCH --job-name=arz_gen_mixed_weno
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_mixed_weno_%j.log

# Generate the mixed general ARZ dataset:
#   - riemann_stratified ICs  -> exact analytic GT (--use-exact-riemann)
#   - piecewise_constant + piecewise_sine ICs -> WENO5+SSP-RK3 (--fv-solver weno5)
# tau=inf (homogeneous), p=rho, 5400 samples, 128x128.
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_mixed_weno_prho.npz}
N=${2:-5400}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
    --out "$OUT" \
    --N "$N" \
    --nx 128 --nt 128 \
    --x-min -1.0 --x-max 1.0 --t-max 1.0 \
    --tau inf \
    --families riemann_stratified,piecewise_constant_stratified,piecewise_sine \
    --segments 2,3,5,7,10,25 \
    --fv-solver weno5 \
    --use-exact-riemann \
    --n-jump-bins 8 \
    --num-workers 8 \
    --cfl 0.4 \
    --refine 1 \
    --boundary ghost \
    --pressure-form rho \
    --rho-min 0.1 --rho-max 0.9 \
    --v-min 0.0 --v-max 1.0 \
    --seed 42

echo "[arz_gen_mixed_weno] done: $OUT"
