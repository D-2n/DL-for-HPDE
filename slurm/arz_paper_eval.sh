#!/bin/bash
#SBATCH --job-name=arz_paper_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_paper_eval_%j.log

set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/results /home/dzdrale/scratch/logs

# ARZ paper eval: stratified (ic_type, num_segments) MAE tables + per-cell
# figures + ID/OOD split, the ARZ analogue of slurm/paper_eval.sh.
#
# Usage: sbatch slurm/arz_paper_eval.sh <ckpt> [data] [out_dir] [train_section]
#   <ckpt>          full path to model_final.pt (config.yaml auto-located beside it)
#   [data]          eval .npz (default: the arz_paper_eval_data path)
#   [out_dir]       output dir (default: <ckpt_dir>/arz_paper_eval)
#   [train_section] config section defining the ID set (default arz_data)
# This is a finite-tau set, so WENO5+Godunov baselines are both valid.
CKPT=${1:?usage: sbatch arz_paper_eval.sh <ckpt> [data] [out_dir] [train_section]}
DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_dataset_paper_eval_prho.npz}
OUTDIR=${3:-}
TRAIN_SECTION=${4:-arz_data}

OUT_ARG=()
if [ -n "$OUTDIR" ]; then
    OUT_ARG=(--out_dir "$OUTDIR")
fi

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.arz_paper_eval \
    --ckpt "$CKPT" \
    --data "$DATA" \
    --baselines godunov,weno5 \
    --train-section "$TRAIN_SECTION" \
    "${OUT_ARG[@]}"
