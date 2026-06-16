#!/bin/bash
#SBATCH --job-name=arz_gen_stratified_homog
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_stratified_homog_%j.log

# CPU-only data generation (no --partition/--gres: CLEPS routes to a CPU default,
# matches slurm/arz_datagen.sh). FV ground truth is the slow part (Strang-split
# solver per sample, refine=4) -> 8h walltime for 5400 samples.
#
# STRATIFIED multi-discontinuity HOMOGENEOUS (tau=inf) ARZ set, p=rho, for the
# ORIGINAL HypNO-ARZ (model_arz_orig, pre-pure-pairwise 2d+12). Mixed 3-family
# arz_data-style mix (piecewise_constant_stratified + piecewise_sine +
# riemann_stratified) with num_segments [2,3,5,7,10,25] -> many ICs with varying
# numbers of discontinuities, like the LWR final-stages stratified set.
# 3 families x 6 segments x 300/cell = 5400 (divisible by 18).
#
# tau=inf -> the FV relaxation step is a no-op (exp(-dt/tau)=1), so this is the
# pure hyperbolic system. _solve_one now allows FV at tau=inf for non-Riemann ICs.
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_stratified_homog_prho.npz}
N=${2:-5400}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
  --out "$OUT" \
  --N "$N" \
  --nx 128 --nt 128 \
  --x-min -1.0 --x-max 1.0 --t-max 1.0 \
  --tau inf \
  --families piecewise_constant_stratified,piecewise_sine,riemann_stratified \
  --segments 2,3,5,7,10,25 \
  --cfl 0.4 --refine 4 --boundary ghost \
  --pressure-form rho \
  --seed 42 \
  --rho-min 0.1 --rho-max 0.9 \
  --v-min 0.1 --v-max 0.9

echo "Done: $OUT"
