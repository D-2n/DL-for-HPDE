#!/bin/bash
#SBATCH --job-name=2d_lwr_viz
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/2d_lwr_visualization_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

/home/dzdrale/hypno_env/bin/python 2d_lwr_visualization.py
