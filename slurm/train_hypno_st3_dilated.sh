#!/bin/bash
#SBATCH --job-name=hypno_st3_dilated
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=parq-gpu001
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_st3_dilated_%j.log

# Sanity-test run for the dilated spatial stencil (dilated_spatial: true).
# Short (epochs: 10 in the config) -- no requeue/marker logic; if preempted,
# just resubmit. Fresh run only: a dilated-stencil model is NOT weight-
# compatible with a dense-stencil checkpoint.
#
# Node selection:
#   --gres=gpu:1            any GPU type (A100/H100/H200 all fit this run)
#   --exclude=parq-gpu001   skip the node that has been running a stale
#                           train_hypno_st3.py (argparse rejects --resume_run)
# ONE --exclude line: slurm silently keeps only the LAST --exclude directive,
# so multiple lines would clobber each other.
#
# Verify in the log:
#   - SLURM_JOB_NODELIST is NOT parq-*
#   - [HypNO_ST3] banner shows  dilated_spatial = True
#   - trains without OOM (edge count 54/node at k_x=5,k_t=4 -- lower than the
#     dense decoder_shared run, so memory is comfortable)

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

echo "=========================================="
echo "SLURM_JOB_ID       = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "=========================================="

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_st3.py \
    --config hyperbolic_pde/configs/hyperbolic_pde_cleps_dilated.yaml
