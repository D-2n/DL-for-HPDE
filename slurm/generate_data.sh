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

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/generate_data.py
