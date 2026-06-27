#!/bin/bash
#SBATCH --job-name=arz_wft_vs_exact
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_wft_vs_exact_%j.log

# Verify the WFT dataset's riemann_stratified samples against the EXACT analytic
# ARZ Riemann solver, cell-by-cell. Answers: is the "intermediate state / smear"
# between two values a REAL star state rho_* / rarefaction fan (exact physics), or
# a numerical artifact?
#   * max|WFT-exact| ~ 1e-6  -> WFT == exact, intermediate cells are real, NO smear.
#   * shock samples with >2 non-(L/rho_*/R) cells -> spurious smear, investigate.
#
# Usage:
#   sbatch slurm/arz_check_wft_vs_exact.sh                              # clean set, 40 samples
#   sbatch slurm/arz_check_wft_vs_exact.sh <data.npz> <n> <out_dir>
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}

DATA=${1:-/home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho_clean.npz}
N=${2:-10}
OUT=${3:-/home/dzdrale/scratch/results/wft_vs_exact}

mkdir -p "$OUT" /home/dzdrale/scratch/logs

echo "Checking WFT vs exact Riemann for: $DATA"
/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.check_wft_vs_exact_riemann \
    --data "$DATA" \
    --pressure-form rho \
    --n "$N" \
    --tol 1e-5 \
    --plot-worst 4 \
    --out "$OUT"

echo "[arz_check_wft_vs_exact] done -> $OUT"
