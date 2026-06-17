#!/bin/bash
#SBATCH --job-name=arz_gen_riemann_exact
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_riemann_exact_%j.log

# CPU-only data generation (no GPU needed -- pure NumPy exact Riemann solver).
# Set the partition with: sbatch --partition=<cpu_partition> slurm/arz_gen_riemann_exact.sh
# (find it via: sinfo -o "%P %a %l %D %t" | grep -iv gpu)
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_riemann_exact_prho_strat.npz}
N=${2:-5000}
BINS=${3:-4}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
  --out "$OUT" \
  --N "$N" \
  --nx 128 --nt 128 \
  --x-min -1.0 --x-max 1.0 --t-max 1.0 \
  --exact-riemann-only \
  --stratified-riemann \
  --n-strength-bins "$BINS" \
  --pressure-form rho \
  --seed 42 \
  --rho-min 0.1 --rho-max 0.9 \
  --v-min 0.0 --v-max 1.0

echo "Done: $OUT"
