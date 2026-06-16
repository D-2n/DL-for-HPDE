#!/bin/bash
#SBATCH --job-name=gen_lin2d_v0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/gen_lin2d_v0_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p /home/dzdrale/scratch/logs /home/dzdrale/scratch/lwr_2d_linear_v0

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_linear_2d_v0.yaml}

/home/dzdrale/hypno_env/bin/python -u hyperbolic_pde/scripts/generate_data_linear_2d_v0.py \
    --config $CONFIG --num-workers 4
