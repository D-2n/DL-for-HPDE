#!/bin/bash
#SBATCH --job-name=arz_plot_strat
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_plot_strat_%j.log

# Plot a few samples from EACH (ic_type x num_segments) bin of an ARZ dataset,
# with a shock-sharpness zoom + per-bin mean jump-width summary. Use this to
# eyeball that every family/bin is SHARP (WFT-exact, ~1-cell jumps) and not
# smeared (HLL-diffused, >=3-cell jumps).
#
# Usage:
#   sbatch slurm/arz_plot_dataset_stratified.sh                                  # clean set, 2/bin
#   sbatch slurm/arz_plot_dataset_stratified.sh <data.npz> <per_bin> <out_dir>
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}

DATA=${1:-/home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho_clean.npz}
PER_BIN=${2:-2}
OUT=${3:-/home/dzdrale/scratch/results/wft_clean_check}

mkdir -p "$OUT" /home/dzdrale/scratch/logs

echo "Plotting stratified samples from: $DATA"
/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.plot_dataset_stratified \
    --data "$DATA" \
    --pressure-form rho \
    --per-bin "$PER_BIN" \
    --out "$OUT"

echo "[arz_plot_dataset_stratified] figures in: $OUT"
