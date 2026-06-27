#!/bin/bash
#SBATCH --job-name=arz_gen_mixed_wft
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_mixed_wft_%j.log

# Generate the mixed general ARZ dataset with WAVE-FRONT TRACKING ground truth:
#   - riemann_stratified ICs           -> exact analytic GT (--use-exact-riemann)
#   - piecewise_constant_stratified ICs -> WFT, machine-precision (--fv-solver wft)
#   - piecewise_sine ICs               -> HLL fallback (sine is not piecewise-const,
#                                          so it cannot be represented as fronts)
# tau=inf (homogeneous -- WFT is source-free), p=rho, 5400 samples, 128x128.
#
# WFT vs the old weno5 path (arz_gen_mixed_weno.sh): exact on shocks & contacts,
# no overshoot/vacuum instability, no crashes, and cheaper than WENO5. The only
# approximation is the rarefaction-fan resolution (rare_delta=0.002, baked in).
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho.npz}
N=${2:-6000}
# WFT rarefaction-fan resolution (absolute, rho units). 0 = auto = 0.1*dx
# (dx = 2/128 = 0.0156 -> 0.00156). Lower = sharper fans but more fronts / slower
# on busy multi-segment ICs. Pass e.g. 0.0008 for ~2x finer.
RARE_DELTA=${3:-0}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
    --out "$OUT" \
    --N "$N" \
    --nx 128 --nt 128 \
    --x-min -1.0 --x-max 1.0 --t-max 1.0 \
    --tau inf \
    --families riemann_stratified,piecewise_constant_stratified,piecewise_sine \
    --segments 2,3,5,7,10, \
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
    --seed 42

echo "[arz_gen_mixed_wft] done: $OUT"
