#!/bin/bash
#SBATCH --job-name=arz_mark0_paper_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_mark0_paper_eval_%j.log

set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/results /home/dzdrale/scratch/logs

# ARZ mark-0 (orig) paper eval: stratified (ic_type, num_segments) MAE tables +
# per-cell figures + ID/OOD split + optional FNO column. The mark-0 analogue of
# slurm/arz_paper_eval.sh (which targets the mark-1 model).
#
# Usage: sbatch slurm/arz_mark0_paper_eval.sh [ckpt] [data] [out_dir] [fno_weights] [fno_section] [train_section]
#   [ckpt]          checkpoint (config.yaml auto-located beside it for the model)
#   [data]          stratified eval .npz, must carry ic_type/num_segments
#   [out_dir]       output dir (default: <ckpt_dir>/arz_mark0_paper_eval)
#   [fno_weights]   FNO checkpoint; set to "" or a missing path to skip FNO
#   [fno_section]   config section for the FNO architecture (default fno_arz_bigger,
#                   matching the locally-trained weights scp'd to CLEPS)
#   [train_section] config section defining the ID set (default arz_data)
# This is a finite-tau set, so WENO5+Godunov baselines are both valid.
CKPT=${1:-/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_arz/run_20260616_173152/checkpoint_epoch180.pt}
DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_dataset_paper_eval_prho.npz}
OUTDIR=${3:-}
FNO_WEIGHTS=${4:-/home/dzdrale/scratch/runs/fno_arz_bigger_prho.pt}
FNO_SECTION=${5:-fno_arz_bigger}
TRAIN_SECTION=${6:-arz_data}

CONFIG=hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml

OUT_ARG=()
if [ -n "$OUTDIR" ]; then
    OUT_ARG=(--out_dir "$OUTDIR")
fi

# Add the FNO column only if a weights file actually exists. The architecture is
# read from $FNO_SECTION of $CONFIG -- it MUST match the trained weights, or
# load_state_dict will fail (the scp'd bigger weights -> fno_arz_bigger: 12 layers).
FNO_ARG=()
if [ -n "$FNO_WEIGHTS" ] && [ -f "$FNO_WEIGHTS" ]; then
    FNO_ARG=(--fno-weights "$FNO_WEIGHTS" --fno-config "$CONFIG" --fno-section "$FNO_SECTION")
else
    echo "[arz_mark0_paper_eval] FNO weights not found ('$FNO_WEIGHTS'); skipping FNO column."
fi

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.arz_mark0_paper_eval \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --model-section hypno_arz_orig \
    --data "$DATA" \
    --baselines godunov,weno5 \
    --train-section "$TRAIN_SECTION" \
    "${FNO_ARG[@]}" \
    "${OUT_ARG[@]}"
