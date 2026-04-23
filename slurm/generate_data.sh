#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/generate_data_%j.log

source /home/dzdrale/hypno_env/bin/activate
cd /home/dzdrale/DL-for-HPDE
mkdir -p /home/dzdrale/scratch/lwr_1d

python hyperbolic_pde/scripts/generate_data.py
