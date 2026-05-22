#!/bin/bash
#SBATCH --job-name=paper_figures
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/dzdrale/scratch/logs/paper_figures_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

# Build the faceted MAE-vs-num_segments figure (and qualitative ID/OOD copies)
# from <run-dir>/paper_eval/summary.csv. Pure-CSV: no GPU, no dataset, no model.
#
# Usage:
#   sbatch slurm/paper_figures.sh <run-dir> [extra args...]
# Defaults: --ood_cell piecewise_sine:30 to match the current OOD ic_types
# (piecewise_constant, piecewise_sine, riemann). Override with extra args, e.g.:
#   sbatch slurm/paper_figures.sh <run-dir> --id_cell riemann:1 --ood_cell piecewise_sine:50
RUN_DIR=${1:?"pass the run directory as the first argument"}
shift

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.scripts.paper_figures \
    --run-dir "$RUN_DIR" \
    --ood_cell piecewise_sine:30 \
    "$@"
