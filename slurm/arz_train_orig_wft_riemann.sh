#!/bin/bash
#SBATCH --job-name=hypno_arz_orig_wft_riemann
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_orig_wft_riemann_%j.log

# Resume HypNO_ARZ_Orig from epoch-90 wft checkpoint, continuing on the
# merged wft+Riemann dataset (arz_mixed_wft_riemann_prho).
# Generate merged dataset first:
#   sbatch slurm/arz_gen_riemann_1k.sh
#   sbatch slurm/arz_merge_wft_riemann.sh
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
DATA_SECTION=${2:-arz_mixed_wft_riemann_prho}
MODEL_SECTION=${3:-hypno_arz_orig}
RESUME=${4:-run_20260620_222959}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
    --config $CONFIG \
    --data-section $DATA_SECTION \
    --model-section $MODEL_SECTION \
    --model-variant orig \
    --resume_run $RESUME
