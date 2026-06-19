#!/bin/bash
#SBATCH --job-name=arz_eval_mark0_paper
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_eval_mark0_paper_%j.log

# ARZ mark-0 (orig) paper evaluation: HypNO-orig vs FNO vs Godunov vs WENO5
# on both the homogeneous multi-disc dataset AND the stratified Riemann dataset.
#
# Positional overrides (all optional -- defaults to the mark-0 run):
#   $1  checkpoint path
#   $2  homog dataset path    (set to "" to skip)
#   $3  riemann dataset path  (set to "" to skip)
#   $4  FNO weights path      (set to "" to skip FNO column)
#   $5  output directory
#   $6  n_samples per dataset
#   $7  n_plots per dataset
#   $8  baselines             ("godunov" for riemann-only; "godunov,weno5" for homog)
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/results /home/dzdrale/scratch/logs

CKPT=${1:-/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_arz/run_20260616_173152/checkpoint_epoch170.pt}
HOMOG_DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_stratified_homog_prho.npz}
RIEMANN_DATA=${3:-/home/dzdrale/scratch/arz_1d/arz_riemann_exact_prho_strat.npz}
FNO_WEIGHTS=${4:-/home/dzdrale/scratch/runs/fno_arz_prho.pt}
OUTDIR=${5:-/home/dzdrale/scratch/results/mark0_paper_eval}
N=${6:-100}
N_PLOTS=${7:-10}
BASELINES=${8:-godunov,weno5}

CONFIG=hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml
mkdir -p "$OUTDIR"

# Build optional args.
HOMOG_ARG=()
[ -n "$HOMOG_DATA" ] && HOMOG_ARG=(--homog-data "$HOMOG_DATA")

RIEMANN_ARG=()
[ -n "$RIEMANN_DATA" ] && RIEMANN_ARG=(--riemann-data "$RIEMANN_DATA")

FNO_ARG=()
[ -n "$FNO_WEIGHTS" ] && [ -f "$FNO_WEIGHTS" ] && FNO_ARG=(--fno-weights "$FNO_WEIGHTS" --fno-config "$CONFIG")

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.arz_orig_paper_eval \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --model-section hypno_arz_orig \
    "${HOMOG_ARG[@]}" \
    "${RIEMANN_ARG[@]}" \
    "${FNO_ARG[@]}" \
    --baselines "$BASELINES" \
    --n-samples "$N" \
    --n-plots "$N_PLOTS" \
    --out_dir "$OUTDIR"
