#!/bin/bash
#SBATCH --job-name=hypno_arz_orig_wft_clean
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_orig_wft_clean_%j.log

# Train mark0 = HypNO_ARZ_Orig (model_arz_orig, new_entropy=true) FROM SCRATCH on
# the CLEAN, fully WFT-exact mixed dataset (arz_mixed_wft_clean_prho). This is the
# replacement for the contaminated arz_mixed_wft_prho run whose piecewise_sine
# third was HLL-diffused. A fresh run (not a finetune/resume of the contaminated
# checkpoint), because ~1/3 of those gradients pulled toward HLL targets.
#
# Prereq: generate + verify the clean dataset first:
#   sbatch slurm/arz_gen_mixed_wft_clean.sh
#   sbatch slurm/arz_inspect_dataset.sh /home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho_clean.npz
#
# Usage:
#   sbatch slurm/arz_train_orig_wft_clean.sh                          # defaults
#   sbatch slurm/arz_train_orig_wft_clean.sh <config> <data_section> <model_section>
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
DATA_SECTION=${2:-arz_mixed_wft_clean_prho}
MODEL_SECTION=${3:-hypno_arz_orig}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
    --config $CONFIG \
    --data-section $DATA_SECTION \
    --model-section $MODEL_SECTION \
    --model-variant orig \
    --auto_resume
