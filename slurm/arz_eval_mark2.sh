#!/bin/bash
#SBATCH --job-name=arz_eval_mark2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_eval_mark2_%j.log

# Evaluate a HypNO-ARZ Mark 2 checkpoint vs numerical baselines on exact-Riemann
# (homogeneous) data. Mark2's forward returns (rho, w, u_hats) with w recovered
# from the (rho, v) decoder, so the eval/plot path is identical to mark1.
#
# Pass the run's config.yaml as $7 if it isn't sitting next to the checkpoint
# (the loader auto-locates run_dir/config.yaml otherwise).
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/results /home/dzdrale/scratch/logs

CKPT=${1:?usage: sbatch arz_eval_mark2.sh <checkpoint.pt> [data] [outdir] [n_plots] [samples] [baselines] [config]}
DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_riemann_exact_prho.npz}
OUTDIR=${3:-/home/dzdrale/scratch/results}
N_PLOTS=${4:-5}
N=${5:-200}
# WENO5 unstable/slow on exact-Riemann ARZ; default godunov only (see
# arz_eval_riemann_exact.sh). Pass weno5,godunov to re-enable.
BASELINES=${6:-godunov}
CONFIG=${7:-}

CONFIG_ARG=()
if [ -n "$CONFIG" ]; then
    CONFIG_ARG=(--config "$CONFIG")
fi

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.eval_vs_numerical_arz \
  --ckpt "$CKPT" --data "$DATA" \
  --model-variant mark2 --model-section hypno_arz_mark2 "${CONFIG_ARG[@]}" \
  --baselines "$BASELINES" --samples "$N" \
  --out "$OUTDIR/arz_mark2_vs_numerical.csv" \
  --figures "$OUTDIR/figs_mark2_vs_numerical" --n-plots "$N_PLOTS"
