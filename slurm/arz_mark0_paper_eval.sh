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
# Defaults target the dedicated ARZ_Evaluation set (arz_evaluation_data /
# arz_evaluation_prho.npz -- generate with slurm/arz_gen_evaluation.sh): WFT GT,
# segments [2,3,5,7,8,10,30,40], ID/OOD vs the clean WFT training set.
# Comparison: HypNO-orig vs FNO, Godunov, HLL, WENO5.
#
# ** WENO5 CRASH RISK: ** on tau=inf / high-segment (30,40) ICs WENO5 weights can
# overflow -> NaN -> a NATIVE access violation that takes the WHOLE process down
# (uncatchable, so the per-baseline try/except can't save it). If a run dies mid-way,
# rerun WENO5 ISOLATED: pass baselines "weno5" (separate out_dir) so a crash can't
# kill the HypNO/FNO/Godunov/HLL results, then splice its column into the table.
#
# Usage: sbatch slurm/arz_mark0_paper_eval.sh [ckpt] [data] [baselines] [out_dir] [fno_weights] [fno_section] [train_section]
#   [ckpt]          checkpoint (config.yaml auto-located beside it for the model arch)
#   [data]          stratified eval .npz, must carry ic_type/num_segments
#   [baselines]     comma list (default godunov,hll,weno5 -- see WENO crash risk)
#   [out_dir]       output dir (default: <ckpt_dir>/arz_mark0_paper_eval)
#   [fno_weights]   FNO checkpoint; "" (default) skips the FNO column
#   [fno_section]   config section for the FNO architecture (must match the weights)
#   [train_section] config section defining the ID set (default arz_mixed_wft_clean_prho)
CKPT=${1:-/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_arz/run_20260620_222959/checkpoint_epoch40.pt}
DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_evaluation_prho.npz}
BASELINES=${3:-godunov,hll,weno5}
OUTDIR=${4:-}
FNO_WEIGHTS=${5:-}
FNO_SECTION=${6:-fno_arz}
TRAIN_SECTION=${7:-arz_mixed_wft_clean_prho}

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
