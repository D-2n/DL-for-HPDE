#!/bin/bash
#SBATCH --job-name=arz_bench_chunked
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/arz_bench_chunked_%j.log

# Equivalence check + memory/time benchmark for the ARZ mark2 / mark2.1 chunked
# edge-message aggregation mode (hyperbolic_pde/arz/bench_chunked_aggregation.py).
#
# No data or checkpoint needed -- it builds fresh random-weight models. Runs the
# fp64/fp32 equivalence gate (materialized vs chunked, all chunk sizes, outputs
# AND grads) then the peak-mem / fwd-bwd-step sweep + OOM frontier. On the 80GB
# A100 the production shape below actually exercises the activation tensor, so
# the peak-memory curve shows properly (a small/low-VRAM GPU is model-bound and
# hides it).
#
# Defaults match the production prho runs: nx=nt=128, batch_size=8, 12 layers,
# k_x=7/k_t=5/d_latent=96 (hyperbolic_pde_arz_cleps_prho.yaml). So the sweep
# reports memory/time in the regime you actually train in, and the OOM frontier
# tells you how much batch headroom chunked mode buys at the real grid.
#
# Args (all optional, positional):
#   $1 MODEL       mark2 | mark2_1            (default: mark2)
#   $2 BATCH       bench batch size          (default: 8  = prho batch_size)
#   $3 NX          spatial cells             (default: 128 = prho nx)
#   $4 NT          time steps                (default: 128 = prho nt)
#   $5 N_LAYERS    processor layers          (default: 12 = prho n_layers)
#   $6 CHUNKS      space-separated sizes     (default: "1 4 8 16 32 64 89")
#   $7 OOM_BATCHES space-separated batches   (default: "4 8 16 32 64 96 128")
#
# Examples:
#   sbatch slurm/arz_bench_chunked.sh                  # mark2, production shape
#   sbatch slurm/arz_bench_chunked.sh mark2_1          # mark2.1
#   sbatch slurm/arz_bench_chunked.sh mark2 8 256 256  # stress a bigger grid
set -euo pipefail
cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:${PYTHONPATH:-}
mkdir -p /home/dzdrale/scratch/logs

MODEL=${1:-mark2}
BATCH=${2:-8}
NX=${3:-128}
NT=${4:-128}
N_LAYERS=${5:-12}
CHUNKS=${6:-"1 4 8 16 32 64 89"}
OOM_BATCHES=${7:-"4 8 16 32 64 96 128"}

# expandable_segments reduces fragmentation OOMs on the materialized baseline.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/dzdrale/hypno_env/bin/python -m hyperbolic_pde.arz.bench_chunked_aggregation \
  --device cuda \
  --model "$MODEL" \
  --pressure-form rho \
  --k-x 7 --k-t 5 --d-latent 96 --n-layers "$N_LAYERS" \
  --bench-batch "$BATCH" --bench-nx "$NX" --bench-nt "$NT" \
  --iters 12 --warmup 3 \
  --chunk-sizes $CHUNKS \
  --oom-chunk 16 \
  --oom-batches $OOM_BATCHES
