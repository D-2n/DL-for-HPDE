#!/bin/bash
#SBATCH --job-name=hypno_arz_orig_wft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_orig_wft_%j.log

# Train mark0 = HypNO_ARZ_Orig (model_arz_orig, new_entropy=true) on the mixed
# general dataset with WAVE-FRONT TRACKING ground truth (arz_mixed_wft_prho).
# Generate the dataset first with:  sbatch slurm/arz_gen_mixed_wft.sh
# Same recipe as arz_train_orig_mixed.sh; only the data section differs (WFT GT
# instead of the WENO5 GT), so a WFT-vs-WENO5 mark0 A/B is a clean swap.
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
DATA_SECTION=${2:-arz_mixed_wft_prho}
MODEL_SECTION=${3:-hypno_arz_orig}
RESUME=${4:-}

if [ -n "$RESUME" ]; then
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant orig \
        --resume_run $RESUME
else
    /home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
        --config $CONFIG \
        --data-section $DATA_SECTION \
        --model-section $MODEL_SECTION \
        --model-variant orig \
        --auto_resume
fi
