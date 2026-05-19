#!/bin/bash
#SBATCH --job-name=gen_ood_256
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/generate_ood_data_256_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/lwr_1d /home/dzdrale/scratch/logs

# Usage: sbatch slurm/generate_ood_data_256.sh [block] [config]
#   block  : ood_data_256_wide (default) or ood_data_256_train
BLOCK=${1:-ood_data_256_train}
CONFIG=${2:-hyperbolic_pde/configs/hyperbolic_pde.yaml}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/generate_ood_data_256.py \
    --config "$CONFIG" --block "$BLOCK"
