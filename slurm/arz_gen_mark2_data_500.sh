#!/bin/bash
#SBATCH --job-name=arz_gen_mark2_data_500
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_mark2_data_500_%j.log

# CPU-only data generation (no GPU needed -- pure NumPy exact Riemann solver).
# Set the partition with: sbatch --partition=<cpu_partition> slurm/arz_gen_mark2_data_500.sh
# (find it via: sinfo -o "%P %a %l %D %t" | grep -iv gpu)
#
# Small 500-sample exact homogeneous (tau=inf) Riemann ARZ dataset for an INITIAL
# train of HypNO-ARZ Mark 1 (router-aware). Identical recipe + seed to the full
# 2000-sample set (slurm/arz_gen_mark2_data_cpu.sh) -- same nx/nt, domain,
# pressure-form, rho/v ranges, exact-Riemann midpoint GT, vacuum-free -- only the
# sample count and output path differ. SAME seed (42) so these 500 are the first
# 500 of the 2000-set's RNG stream (reproducible subset).
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/arz_1d /home/dzdrale/scratch/logs

OUT=${1:-/home/dzdrale/scratch/arz_1d/arz_riemann_mark2_prho_500.npz}
N=${2:-500}

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
