#!/bin/bash
#SBATCH --job-name=hypno_arz_resume_hll_ft_wft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu012,gpu013,gpu015,gpu016,gpu017,gpu018
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_arz_resume_hll_ft_wft_%j.log

# Explicitly resume the HLL->WFT mark0 finetune run from its latest checkpoint.
# (Generic under the hood: works for any HypNO-ARZ run dir, but named for the
# HLL-pretrained orig model being finetuned on the clean WFT set.)
#
# Unlike the finetune/from-scratch wrappers (which rely on --auto_resume + the
# latest_run.txt pointer), this uses --resume_run <dir> directly, so it ALWAYS
# continues the run you name -- no pointer lottery, no chance of falling through
# to a fresh --finetune_from restart. --resume_run re-reads the run dir's own
# config.yaml (data/model sections, lr, schedule) and continues the optimizer +
# epoch counter from the highest checkpoint_epoch*.pt in that dir.
#
# Usage:
#   sbatch slurm/arz_resume.sh <run_dir> [--model-variant orig|mark2|mark2_1|...]
# Example (the HLL->WFT finetune, currently at epoch 88):
#   sbatch slurm/arz_resume.sh \
#     /home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_arz/run_20260624_090720
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/runs /home/dzdrale/scratch/logs

RUN_DIR=${1:?"pass the run directory to resume as the first argument"}
shift

# --model-variant defaults to orig (mark0); override by passing it in [extra args].
# Anything after <run_dir> is forwarded verbatim to the trainer.
VARIANT_ARGS=("--model-variant" "orig")
if [[ "$*" == *"--model-variant"* ]]; then
    VARIANT_ARGS=()  # caller supplied their own; don't double it
fi

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/scripts/train_hypno_arz.py \
    --resume_run "$RUN_DIR" \
    "${VARIANT_ARGS[@]}" \
    "$@"
