#!/bin/bash
#SBATCH --job-name=arz_eval_mark2_1_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_eval_mark2_1_cpu_%j.log

# CPU-only variant of arz_eval_mark2_1.sh. No --partition/--gres: CLEPS routes
# partition-less jobs to a CPU default (matches slurm/arz_datagen.sh). Forces
# --device cpu so the eval doesn't look for a GPU. Eval is forward-only (no
# training) + numpy baselines (hll/weno5), so CPU is fine; just slower than A100.
#
# Mark2.1's forward returns (rho, w, u_hats) with w recovered from the (rho, v)
# decoder, so the eval/plot path is identical to mark2 -- only the variant/section
# differ. Pass the run's config.yaml as $7 if it isn't next to the checkpoint.
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/results /home/dzdrale/scratch/logs

CKPT=${1:?usage: sbatch arz_eval_mark2_1_cpu.sh <checkpoint.pt> [data] [outdir] [n_plots] [samples] [baselines] [config]}
DATA=${2:-/home/dzdrale/scratch/arz_1d/arz_riemann_exact_prho_strat.npz}
OUTDIR=${3:-/home/dzdrale/scratch/results}
N_PLOTS=${4:-5}
N=${5:-200}
# weno5 is the better baseline (sharper, cheaper than per-interface godunov) and
# runs on CPU fine. Pass 'hll' for the fastest, or 'godunov' (small --samples) for
# the sharp-contact reference.
BASELINES=${6:-weno5}
CONFIG=${7:-}

CONFIG_ARG=()
if [ -n "$CONFIG" ]; then
    CONFIG_ARG=(--config "$CONFIG")
fi

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.eval_vs_numerical_arz \
  --ckpt "$CKPT" --data "$DATA" --device cpu \
  --model-variant mark2_1 --model-section hypno_arz_mark2_1 "${CONFIG_ARG[@]}" \
  --baselines "$BASELINES" --samples "$N" \
  --out "$OUTDIR/arz_mark2_1_vs_numerical.csv" \
  --figures "$OUTDIR/figs_mark2_1_vs_numerical" --n-plots "$N_PLOTS"
