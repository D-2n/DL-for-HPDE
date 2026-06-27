#!/bin/bash
#SBATCH --job-name=arz_eval_orig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_eval_orig_%j.log

# Evaluate the ORIGINAL pre-pure-pairwise 2d+12 HypNO_ARZ_Orig checkpoint vs
# numerical baselines on the homogeneous stratified set it was trained on.
# Mark1 frame: forward returns (rho, w, u_hats), so the eval/plot path is
# identical to plain mark1 -- only the variant/section/output names differ.
#
# The model-section name contains 'orig', so --model-variant auto resolves to
# HypNO_ARZ_Orig; passed explicitly here. The bare-state_dict checkpoint's
# architecture is read from run_dir/config.yaml (auto-located next to the .pt);
# pass it as $7 if it isn't sitting beside the checkpoint.
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/results /home/dzdrale/scratch/logs

CKPT=${1:?usage: sbatch arz_eval_orig.sh <checkpoint.pt> [data] [outdir] [n_plots] [samples] [baselines] [config]}
DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho_clean.npz}
OUTDIR=${3:-/home/dzdrale/scratch/results}
N_PLOTS=${4:-5}
N=${5:-200}
# Homogeneous (tau=inf): godunov here = HLL + (no relaxation). weno5 is the
# sharper but slower baseline. Default to both; drop to 'godunov' (or fewer
# --samples) if walltime gets tight.
BASELINES=${6:-godunov,muscl}
CONFIG=${7:-}

CONFIG_ARG=()
if [ -n "$CONFIG" ]; then
    CONFIG_ARG=(--config "$CONFIG")
fi

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.eval_vs_numerical_arz \
  --ckpt "$CKPT" --data "$DATA" \
  --model-variant orig --model-section hypno_arz_orig "${CONFIG_ARG[@]}" \
  --baselines "$BASELINES" --samples "$N" \
  --out "$OUTDIR/arz_orig_vs_numerical.csv" \
  --figures "$OUTDIR/figs_orig_vs_numerical" --n-plots "$N_PLOTS"
