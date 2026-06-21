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
# Defaults target the WFT-trained mark0 (run_20260620_222959) vs Godunov on a
# HELD-OUT / OOD WFT set (unseen segments -- generate with slurm/arz_gen_ood_wft.sh).
# This is a homogeneous (tau=inf) set, so WENO5 is unstable/crashes -> Godunov
# ONLY by default. FNO is off unless you pass a weights path.
#
# Usage: sbatch slurm/arz_mark0_paper_eval.sh [ckpt] [data] [baselines] [out_dir] [fno_weights] [fno_section] [train_section]
#   [ckpt]          checkpoint (config.yaml auto-located beside it for the model arch)
#   [data]          stratified eval .npz, must carry ic_type/num_segments
#   [baselines]     comma list (default godunov; WENO5 crashes on tau=inf ARZ)
#   [out_dir]       output dir (default: <ckpt_dir>/arz_mark0_paper_eval)
#   [fno_weights]   FNO checkpoint; "" (default) skips the FNO column
#   [fno_section]   config section for the FNO architecture (must match the weights)
#   [train_section] config section defining the ID set (default arz_data)
CKPT=${1:-/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_arz/run_20260620_222959/checkpoint_epoch40.pt}
DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_ood_wft_prho.npz}
BASELINES=${3:-godunov}
OUTDIR=${4:-}
FNO_WEIGHTS=${5:-}
FNO_SECTION=${6:-fno_arz}
TRAIN_SECTION=${7:-arz_data}

CONFIG=hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml

OUT_ARG=()
if [ -n "$OUTDIR" ]; then
    OUT_ARG=(--out_dir "$OUTDIR")
fi

# Add the FNO column only if a weights file actually exists. The architecture is
# read from $FNO_SECTION of $CONFIG -- it MUST match the trained weights, or
# load_state_dict will fail.
FNO_ARG=()
if [ -n "$FNO_WEIGHTS" ] && [ -f "$FNO_WEIGHTS" ]; then
    FNO_ARG=(--fno-weights "$FNO_WEIGHTS" --fno-config "$CONFIG" --fno-section "$FNO_SECTION")
elif [ -n "$FNO_WEIGHTS" ]; then
    echo "[arz_mark0_paper_eval] FNO weights not found ('$FNO_WEIGHTS'); skipping FNO column."
fi

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.arz_mark0_paper_eval \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --model-section hypno_arz_orig \
    --data "$DATA" \
    --baselines "$BASELINES" \
    --train-section "$TRAIN_SECTION" \
    "${FNO_ARG[@]}" \
    "${OUT_ARG[@]}"
