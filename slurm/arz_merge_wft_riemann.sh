#!/bin/bash
#SBATCH --job-name=arz_merge_wft_riemann
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_merge_wft_riemann_%j.log

# Merge arz_mixed_wft_prho.npz + arz_riemann_1k.npz -> arz_mixed_wft_riemann_prho.npz
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}

A=${1:-/home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho.npz}
B=${2:-/home/dzdrale/scratch/arz_1d/arz_riemann_1k.npz}
OUT=${3:-/home/dzdrale/scratch/arz_1d/arz_mixed_wft_riemann_prho.npz}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.merge_arz_datasets \
    --a "$A" \
    --b "$B" \
    --out "$OUT"

echo "[arz_merge] done: $OUT"
