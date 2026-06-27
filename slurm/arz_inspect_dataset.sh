#!/bin/bash
#SBATCH --job-name=arz_inspect_data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_inspect_data_%j.log

# Inspect an ARZ npz dataset: shape, field ranges, IC type counts, and
# pressure-form residual check (w - v - rho vs w - v - rho - rho²).
#
# Usage:
#   sbatch slurm/arz_inspect_dataset.sh                        # default dataset
#   sbatch slurm/arz_inspect_dataset.sh /path/to/dataset.npz   # custom

set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}

DATA=${1:-/home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho.npz}

echo "Inspecting: $DATA"
/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.inspect_dataset --data "$DATA"
