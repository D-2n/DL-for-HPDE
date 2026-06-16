#!/bin/bash
#SBATCH --job-name=arz_gen_mark2_data_5400
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_mark2_data_5400_%j.log

# CPU-only data generation (no GPU needed -- pure NumPy exact Riemann solver).
# No --partition/--gres: CLEPS routes partition-less jobs to a CPU-capable
# default (matches slurm/arz_datagen.sh).
#
# Big 5400-sample exact homogeneous (tau=inf) Riemann ARZ dataset for a proper
# mark1 (normal) train. SAME recipe + distribution as the 2000-sample mark2 set
# (slurm/arz_gen_mark2_data.sh): p=rho, vacuum-free, rho in [0.1,0.95],
# v in [0.1,0.9], exact-Riemann midpoint GT. Only N + output path differ. SAME
# seed (42) so the first 2000 of these match the mark2 set's RNG stream.
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_riemann_mark2_prho_5400.npz}
N=${2:-5400}

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
  --out "$OUT" \
  --N "$N" \
  --nx 128 --nt 128 \
  --x-min -1.0 --x-max 1.0 --t-max 1.0 \
  --exact-riemann-only \
  --pressure-form rho \
  --seed 42 \
  --rho-min 0.1 --rho-max 0.95 \
  --v-min 0.1 --v-max 0.9

echo "Done: $OUT"
