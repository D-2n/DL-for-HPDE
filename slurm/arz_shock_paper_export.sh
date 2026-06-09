#!/bin/bash
#SBATCH --job-name=arz_shock_export
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_shock_export_%j.log

set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/results /home/dzdrale/scratch/logs

# ARZ shock-band paper export: mae-vs-segments + per-seg compare/zoom/slice
# figures + shock_table.tex/shock_results.tex, the ARZ analogue of
# slurm/shock_paper_export.sh. rho headline; (full,1shock,contact) table cols.
#
# Usage: sbatch slurm/arz_shock_paper_export.sh <ckpt> [data] [out_dir] [n_per_group]
#   <ckpt>        full path to model_final.pt (config.yaml auto-located beside it)
#   [data]        eval .npz (default: the arz_paper_eval_data path)
#   [out_dir]     output dir (default: <ckpt_dir>/arz_paper_shock)
#   [n_per_group] cap samples per num_segments group (default: all)
CKPT=${1:?usage: sbatch arz_shock_paper_export.sh <ckpt> [data] [out_dir] [n_per_group]}
DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_dataset_paper_eval_prho.npz}
OUTDIR=${3:-}
N_PER_GROUP=${4:-}

OUT_ARG=()
if [ -n "$OUTDIR" ]; then
    OUT_ARG=(--out_dir "$OUTDIR")
fi
NPG_ARG=()
if [ -n "$N_PER_GROUP" ]; then
    NPG_ARG=(--n_per_group "$N_PER_GROUP")
fi

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.arz_shock_paper_export \
    --ckpt "$CKPT" \
    --data "$DATA" \
    --baselines godunov,weno5 \
    --tau-shock 0.06 --band-halfwidth 2 --tv-mult 1.5 \
    --seed 0 \
    "${OUT_ARG[@]}" "${NPG_ARG[@]}"
