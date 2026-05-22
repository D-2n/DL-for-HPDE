#!/bin/bash
#SBATCH --job-name=diag_strong_shocks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/diag_strong_shocks_%j.log

#SBATCH --exclude=gpu012
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p /home/dzdrale/scratch/logs

# Run HypNO-ST3 on a deterministic grid of strong Riemann shocks
# (u_L in {0.00,0.05,0.10}, u_R in {0.90,1.00}; 6 samples).
# Outputs land in <RUN_DIR>/diag_strong_shocks/.
#
# Override RUN_DIR by passing it as the first positional arg:
#   sbatch slurm/diag_strong_shocks.sh /path/to/other/run_dir
RUN_DIR=${1:-/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_st3/run_20260518_000640_128}
shift || true

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.scripts.diag_strong_shocks \
    --run-dir "$RUN_DIR" \
    "$@"
