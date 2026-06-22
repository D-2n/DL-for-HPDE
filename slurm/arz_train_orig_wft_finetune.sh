#!/bin/bash
#SBATCH --job-name=hypno_arz_orig_wft_ft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_orig_wft_ft_%j.log

# Fine-tune a pre-trained HypNO_ARZ_Orig checkpoint on the WFT dataset.
# First launch: --finetune_from loads weights only, fresh optimizer+scheduler,
#               new run dir, epoch counter reset to 1.
# On requeue:   --auto_resume finds the new run dir via the job-keyed pointer
#               and continues from the latest checkpoint there.
#
# Usage:
#   sbatch slurm/arz_train_orig_wft_finetune.sh                        # defaults
#   sbatch slurm/arz_train_orig_wft_finetune.sh <ckpt> <data_section>  # overrides
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml
FINETUNE_CKPT=${1:-/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_arz/run_20260617_195405/model_final.pt}
DATA_SECTION=${2:-arz_mixed_wft_prho}
MODEL_SECTION=${3:-hypno_arz_orig}

# On requeue, SLURM re-runs this script with the same args. --auto_resume
# detects the existing run dir via the job-keyed pointer and resumes from
# the latest checkpoint, skipping --finetune_from entirely.
/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
    --config $CONFIG \
    --data-section $DATA_SECTION \
    --model-section $MODEL_SECTION \
    --model-variant orig \
    --finetune_from $FINETUNE_CKPT \
    --auto_resume
