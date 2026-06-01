#!/bin/bash
#SBATCH --job-name=arz_datagen_riemann
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_datagen_riemann_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps.yaml}
SECTION=${2:-arz_riemann_trial}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/generate_arz_data.py \
    --config $CONFIG --section $SECTION
