#!/bin/bash
#SBATCH --job-name=diag_st3_2d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=96G
#SBATCH --time=0:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/diag_st3_2d_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
mkdir -p /home/dzdrale/scratch/logs

# $1 = config (default: 2d cleps), $2 = checkpoint path (required for a
# specific run; if omitted the script picks the latest under run_base).
CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_2d_cleps.yaml}
CKPT=${2:-}

if [ -n "$CKPT" ]; then
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/diagnose_2d.py --config $CONFIG --ckpt $CKPT
else
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/diagnose_2d.py --config $CONFIG
fi
