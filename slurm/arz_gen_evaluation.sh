#!/bin/bash
#SBATCH --job-name=arz_gen_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_eval_%j.log

# Generate the dedicated ARZ_Evaluation set (config section arz_evaluation_data).
# This is the ARZ analogue of the LWR paper_eval_data set -- same num_segments
# grid so the "#-of-discontinuities ID vs OOD" story is directly comparable.
#
#   GT       : WFT, machine precision (NO HLL, NO WENO5). riemann_stratified ->
#              exact analytic GT (--use-exact-riemann); the two staircase families
#              -> WFT.
#   physics  : tau=inf (homogeneous), p=rho, rho in [0.1,0.9], v in [0,1].
#   grid     : nx=nt=128, x in [-1,1], t in [0,1], cfl 0.4, refine 1 (match the
#              clean WFT training set so eval is on the same regime).
#   families : riemann_stratified, piecewise_constant_stratified, piecewise_sine
#              (piecewise_sine maps to the sine STAIRCASE -- WFT-able).
#   segments : 2,3,5,7,8,10,30,40
#              vs clean-WFT training segs [2,3,5,7,10]: {2,3,5,7,10} ID, {8,30,40} OOD.
#   N = 480  = 3 families x 8 segments x 20/cell (divisible by 24).
#   seed 2026 (!= training seed 42) -> ID-segment cells are held-out samples too.
#
# Usage:
#   sbatch slurm/arz_gen_evaluation.sh                       # -> arz_evaluation_prho.npz, 480
#   sbatch slurm/arz_gen_evaluation.sh <out.npz> <N> <rare_delta>
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_evaluation_prho.npz}
N=${2:-480}
# WFT rarefaction-fan resolution. 0 = auto (0.1*dx) = coarsest sub-cell fans =
# fewest fronts = fastest/safest. The high-segment ICs (30/40) have many
# interfaces; do NOT push rare_delta much below ~0.0005 here or the O(n^2)
# front-collision blow-up can hang datagen on the busy cells.
RARE_DELTA=${3:-0}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
    --out "$OUT" \
    --N "$N" \
    --nx 128 --nt 128 \
    --x-min -1.0 --x-max 1.0 --t-max 1.0 \
    --tau inf \
    --families riemann_stratified,piecewise_constant_stratified,piecewise_sine \
    --segments 2,3,5,7,8,10,30,40 \
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
    --seed 2026

echo "[arz_gen_evaluation] done: $OUT"
echo "Verify it is HLL-free + sharp before evaluating:"
echo "  sbatch slurm/arz_inspect_dataset.sh $OUT"
