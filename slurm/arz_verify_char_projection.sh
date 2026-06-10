#!/bin/bash
#SBATCH --job-name=arz_verify_char_proj
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_verify_char_proj_%j.log

# Verify the ARZ characteristic projection isolates the two wave families
# (1-wave / GNL vs 2-contact / LD). Pure numpy, CPU-only, no dataset.
#
# Usage:
#   sbatch slurm/arz_verify_char_projection.sh                 # default closure
#   sbatch slurm/arz_verify_char_projection.sh rho             # p(rho)=rho
#   sbatch slurm/arz_verify_char_projection.sh rho+rho2        # p(rho)=rho+rho^2

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/logs

PFORM=${1:-rho+rho2}

/home/dzdrale/hypno_env/bin/python hyperbolic_pde/arz/verify_char_projection.py \
    --pressure-form "$PFORM"
