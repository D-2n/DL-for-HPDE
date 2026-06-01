#!/bin/bash
#SBATCH --job-name=arz_plot_riemann
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_plot_riemann_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/figures /home/dzdrale/scratch/logs

NPZ=${1:-/home/dzdrale/scratch/arz_1d/arz_riemann_trial.npz}
OUT=${2:-/home/dzdrale/scratch/figures/arz_riemann_trial}
N=${3:-12}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/plot_arz_data.py \
    --npz "$NPZ" --out "$OUT" --n "$N"
