#!/bin/bash
#SBATCH --job-name=arz_gen_paper_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_paper_eval_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

# Generates the stratified ARZ paper-eval set (section arz_paper_eval_data).
# Used by both arz_paper_eval.sh and arz_shock_paper_export.sh. Finite tau, so
# WENO5/Godunov baselines are valid. Defaults to the prho config.
CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
SECTION=${2:-arz_paper_eval_data}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/generate_arz_data.py \
    --config $CONFIG --section $SECTION
