#!/bin/bash
#SBATCH --job-name=gen_lwr2d
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/gen_lwr2d_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/logs /home/dzdrale/scratch/lwr_2d

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_2d_cleps.yaml}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/generate_data_2d.py --config $CONFIG
