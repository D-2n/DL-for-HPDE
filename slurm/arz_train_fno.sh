#!/bin/bash
#SBATCH --job-name=fno_arz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/fno_arz_%j.log

# Train the 2-channel FNO baseline on the SAME ARZ data that "arz hypno orig"
# trains on (data section arz_stratified_homog_prho by default). Input
# (X, T, rho0, w0) -> output (rho, w); identical train/val split + seed as
# train_hypno_arz.py, so it is a clean baseline against HypNO_ARZ_Orig.
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
DATA_SECTION=${2:-arz_stratified_homog_prho}
FNO_SECTION=${3:-fno_arz}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/arz/train_fno_arz.py \
    --config $CONFIG \
    --data-section $DATA_SECTION \
    --fno-section $FNO_SECTION
