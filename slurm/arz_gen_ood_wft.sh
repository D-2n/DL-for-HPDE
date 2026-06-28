#!/bin/bash
#SBATCH --job-name=arz_gen_ood_wft
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_ood_wft_%j.log

# Generate a HELD-OUT / OOD ARZ eval set with WAVE-FRONT TRACKING ground truth,
# the OOD companion to arz_gen_mixed_wft.sh (which makes the TRAINING set with
# segments [2,3,5,7,10,25], seed 42). This set differs in TWO ways so it is a
# genuine out-of-distribution probe for the WFT-trained mark0:
#   - UNSEEN num_segments (default 4,6,8,15,40 -- none in the training set)
#   - a DIFFERENT seed (default 2026, != the training seed 42)
# Same physics/families/grid/GT recipe as the training set otherwise, so the
# per-(family, num_segments) eval cells stay comparable.
#
# Families/GT routing mirror arz_gen_mixed_wft.sh:
#   riemann_stratified -> exact analytic;  piecewise_constant_stratified -> WFT;
#   piecewise_sine -> HLL fallback (not piecewise-const, cannot be fronts).
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_ood_wft_prho.npz}
# N must divide n_families * n_segments = 3 * 5 = 15. 300 -> 20/cell.
N=${2:-300}
SEGMENTS=${3:-4,6,8,15,40}
SEED=${4:-2026}
RARE_DELTA=${5:-0}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
    --out "$OUT" \
    --N "$N" \
    --nx 128 --nt 128 \
    --x-min -1.0 --x-max 1.0 --t-max 1.0 \
    --tau inf \
    --families riemann_stratified,piecewise_constant_stratified,piecewise_sine \
    --segments "$SEGMENTS" \
    --fv-solver wft \
    --wft-rare-delta "$RARE_DELTA" \
    --use-exact-riemann \
    --n-jump-bins 8 \
    --num-workers 8 \
    --cfl 0.4 \
    --refine 1 \
    --boundary ghost \
    --pressure-form rho \
    --rho-min 0.1 --rho-max 0.9 \
    --v-min 0.0 --v-max 1.0 \
    --seed "$SEED"

echo "[arz_gen_ood_wft] done: $OUT"
