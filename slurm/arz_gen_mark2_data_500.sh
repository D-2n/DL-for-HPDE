#!/bin/bash
#SBATCH --job-name=arz_gen_mark2_data_500
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_gen_mark2_data_500_%j.log

# CPU-only data generation (no GPU needed -- pure NumPy exact Riemann solver).
# No --partition/--gres: CLEPS routes partition-less jobs to a CPU-capable
# default (matches slurm/arz_datagen.sh).
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

# v-min is NEGATIVE on purpose: v_* (= v_R = the 2-contact speed) is drawn in
# [v-min, v-max], so a negative floor adds LEFT-GOING and near-STATIONARY
# contacts -- previously absent (v_*>0 always) and a real coverage hole for a
# family-routing model. Vacuum-free still holds for any sign of v_*: rho_* is
# pinned by the directly-drawn rho_star (w_L - v_R = p(rho_*) >= p(rho_min) > 0
# by construction, independent of v_*'s sign). Kept v-max=0.9 so strong
# right-going contacts remain well represented.
/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.datagen_arz \
  --out "$OUT" \
  --N "$N" \
  --nx 128 --nt 128 \
  --x-min -1.0 --x-max 1.0 --t-max 1.0 \
  --exact-riemann-only \
  --pressure-form rho \
  --seed 42 \
  --rho-min 0.1 --rho-max 0.95 \
  --v-min -0.5 --v-max 0.9

echo "Done: $OUT"
