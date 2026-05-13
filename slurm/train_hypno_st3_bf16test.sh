#!/bin/bash
#SBATCH --job-name=hypno_st3_bf16test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_st3_bf16test_%j.log

# 2-epoch smoke test for bf16 autocast.
# Expected outcomes (from the log):
#   - "Mixed precision: torch.autocast(cuda, dtype=torch.bfloat16)"
#   - No CUDA errors / no "no kernel for bfloat16" exceptions
#   - Per-batch loss values in the same order of magnitude as fp32 run
#   - GPU memory well under 80 GB at bs=16 (was ~60 GB at bs=8 fp32)
# If all three pass, edit the main decoder_shared.yaml to add `amp_dtype: bf16`
# and resubmit the full 200-epoch job.

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

CONFIG=hyperbolic_pde/configs/hyperbolic_pde_cleps_decoder_shared_bf16test.yaml

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_st3.py \
    --config $CONFIG
