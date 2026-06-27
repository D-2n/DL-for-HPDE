#!/bin/bash
#SBATCH --job-name=hypno_arz_orig_wft_clean_fast
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_orig_wft_clean_fast_%j.log

# SPEED-TUNED twin of slurm/arz_train_orig_wft_clean.sh: same model + recipe +
# CLEAN WFT dataset, but the hypno_arz_orig_fast model section turns on three
# numerically-equivalent perf flags -- fast_message_factor (point 2, the big
# k_x-preserving win), fast_static_cache (point 3), checkpoint_stride=2 (point 1)
# -- all verified fp64 in hyperbolic_pde/arz/tests/test_orig_fast_equiv.py, plus
# the trainer's per-step CUDA-sync removal. It trains the SAME model as
# hypno_arz_orig, just faster -- safe to compare against the canonical run.
#
# Memory: checkpoint_stride=2 recomputes only every 2nd MP layer in backward
# (faster) at higher peak memory than stride=1. The nodelist is the 80GB+/H200
# pool. If this OOMs, either set checkpoint_stride:1 in the config (you still get
# fast_static_cache + the sync removal), or pin to the H200 (gpu018, 141GB) and
# set use_checkpoint:false for the full no-recompute speedup.
#
# Usage:
#   sbatch slurm/arz_train_orig_wft_clean_fast.sh                          # defaults
#   sbatch slurm/arz_train_orig_wft_clean_fast.sh <config> <data_section> <model_section>
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

CONFIG=${1:-hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml}
DATA_SECTION=${2:-arz_mixed_wft_clean_prho}
MODEL_SECTION=${3:-hypno_arz_orig_fast}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
    --config $CONFIG \
    --data-section $DATA_SECTION \
    --model-section $MODEL_SECTION \
    --model-variant orig \
    --auto_resume
