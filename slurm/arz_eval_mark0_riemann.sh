#!/bin/bash
#SBATCH --job-name=arz_eval_mark0_riemann
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_eval_mark0_riemann_%j.log

# Evaluate the mark0 (orig) HypNO-ARZ checkpoint trained on the stratified
# exact-Riemann homogeneous dataset (arz_riemann_strat_prho).
# Model: HypNO_ARZ_Orig (model_arz_orig), --model-variant orig.
# Data:  arz_riemann_exact_prho_strat.npz (canonical 5000-sample stratified set).
# Baselines: godunov (exact-Riemann Godunov flux). WENO5 skipped -- unstable on
# exact-Riemann ARZ data (overflow near vacuum, walltime risk).
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/results /home/dzdrale/scratch/logs

CKPT=${1:-hyperbolic_pde/runs/hypno_arz/run_20260624_001010/checkpoint_epoch20.pt}
DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho_clean.npz}
OUTDIR=${3:-/home/dzdrale/scratch/results/mark0_riemann_wft}
N_PLOTS=${4:-10}
N=${5:-50}
BASELINES=${6:-godunov}
CONFIG=${7:-}

mkdir -p "$OUTDIR"

CONFIG_ARG=()
if [ -n "$CONFIG" ]; then
    CONFIG_ARG=(--config "$CONFIG")
fi

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.eval_vs_numerical_arz \
  --ckpt "$CKPT" --data "$DATA" \
  --model-variant orig --model-section hypno_arz_orig "${CONFIG_ARG[@]}" \
  --baselines "$BASELINES" --samples "$N" \
  --out "$OUTDIR/mark0_riemann_vs_numerical.csv" \
  --figures "$OUTDIR/figs_vs_numerical" --n-plots "$N_PLOTS"

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.eval_shock_arz \
  --ckpt "$CKPT" --data "$DATA" \
  --model-variant orig --model-section hypno_arz_orig "${CONFIG_ARG[@]}" \
  --baselines "$BASELINES" --samples "$N" \
  --tau-shock 0.06 --band-halfwidth 2 --tv-mult 1.5 \
  --out "$OUTDIR/mark0_riemann_shock.csv" \
  --figures "$OUTDIR/figs_shock" --n-plots "$N_PLOTS"
