#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/generate_data_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/lwr_1d

# Usage: sbatch slurm/generate_ood_data.sh [block] [config]
#   block : ood_data (default) or e.g. ood_data_oneshock
BLOCK=${1:-ood_data}
CONFIG=${2:-hyperbolic_pde/configs/hyperbolic_pde.yaml}
/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/generate_ood_data.py \
    --config "$CONFIG" --block "$BLOCK"
