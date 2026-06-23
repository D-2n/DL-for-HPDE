#!/bin/bash
#SBATCH --job-name=arz_gen_wft_clean
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_wft_clean_%j.log

# Generate a CLEAN, FULLY WFT-EXACT mixed ARZ dataset (NO HLL anywhere):
#   - riemann_stratified            -> exact analytic GT (--use-exact-riemann)
#   - piecewise_constant_stratified -> WFT, machine-precision
#   - piecewise_sine                -> WFT, machine-precision  ** NOW a sine
#       STAIRCASE (sine_staircase_ic): a single global sine quantised into
#       num_segments piecewise-CONSTANT levels. Finite discontinuities, so WFT
#       evolves it exactly. (The old script mapped this family to the SMOOTH
#       piecewise_sine_ic, which the wft path silently solved with HLL -> 1/3 of
#       arz_mixed_wft_prho.npz was HLL-diffused. Fixed in datagen_arz.py
#       2026-06-23; the wft path now has NO HLL fallback and RAISES on a smooth IC.)
#
# Defaults match the live arz_mixed_wft_prho.npz shape (6000 = 2000 x 3 families,
# segments 2,3,5,7,10) so this is a DROP-IN replacement. Writes to a NEW path by
# default so it can never clobber the file the running jobs read.
#
# Usage:
#   sbatch slurm/arz_gen_mixed_wft_clean.sh                       # -> _clean.npz, 6000
#   sbatch slurm/arz_gen_mixed_wft_clean.sh <out.npz> <N> <rare_delta>
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho_clean.npz}
N=${2:-6000}
# WFT rarefaction-fan resolution (absolute, rho units).
#   0 (default) = auto = 0.1*dx = 0.1*(2/128) ~= 0.00156. Coarsest fans that are
#     still sub-cell -> fewest fronts -> FASTEST. Safe default.
#   Finer (e.g. 0.0008) = ~2x sharper rarefactions but ~2x the fronts per fan.
#     The known-good 6000-sample run used 0.0008 -- but the sine STAIRCASE has
#     more interfaces than the old smooth->HLL path, so each fan is multiplied by
#     more interfaces. Going much below ~0.0005 risks the O(n^2) front-collision
#     blow-up (the 2026-06-20 datagen hang) on the busy 7/10-segment ICs.
# Sweet spot: leave at 0 (auto) for the first clean regen; only drop to 0.0008 if
# rarefaction sharpness in the output looks grid-limited.
RARE_DELTA=${3:-0}

# Safety: refuse to overwrite the contaminated live file the running jobs read.
if [ "$OUT" = "/home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho.npz" ]; then
    echo "REFUSING to write to the live (contaminated) dataset path. Pick a new OUT." >&2
    exit 1
fi

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
    --out "$OUT" \
    --N "$N" \
    --nx 128 --nt 128 \
    --x-min -1.0 --x-max 1.0 --t-max 1.0 \
    --tau inf \
    --families riemann_stratified,piecewise_constant_stratified,piecewise_sine \
    --segments 2,3,5,7,10 \
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

echo "[arz_gen_mixed_wft_clean] done: $OUT"
echo "Verify it is HLL-free before training:"
echo "  sbatch slurm/arz_inspect_dataset.sh $OUT"
