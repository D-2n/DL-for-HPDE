#!/bin/bash
#SBATCH --job-name=arz_datagen_bigger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_datagen_bigger_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

# Generates the larger (~6k-sample) ARZ training set, section arz_data_bigger.
# Defaults to the prho config (pressure_form=rho). Override config/section as args.
CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
SECTION=${2:-arz_data_bigger}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/generate_arz_data.py \
    --config $CONFIG --section $SECTION
