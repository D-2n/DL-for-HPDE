#!/bin/bash
#SBATCH --job-name=hypno_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=parq-gpu001
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/dzdrale/scratch/logs/hypno_resume_%j.log

cd /home/dzdrale/DL-for-HPDE
export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "SLURM_JOB_ID       = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "=========================================="

echo "=== DIAGNOSTIC ==="
echo "PWD: $(pwd)"
echo "cd exit status was checked; pwd above must be /home/dzdrale/DL-for-HPDE"
echo "PYTHONPATH: $PYTHONPATH"
SCRIPT=/home/dzdrale/DL-for-HPDE/hyperbolic_pde/scripts/train_hypno_st3.py
echo "script abs path: $SCRIPT"
ls -la "$SCRIPT"
md5sum "$SCRIPT"
echo "--- resume_run lines in that exact file: ---"
grep -n resume_run "$SCRIPT"
echo "--- python executable + version: ---"
/home/dzdrale/hypno_env/bin/python -c "import sys; print('executable:', sys.executable); print('version:', sys.version)"
echo "--- sys.path: ---"
/home/dzdrale/hypno_env/bin/python -c "import sys; [print('  ', p) for p in sys.path]"
echo "--- --help of the absolute-path script: ---"
/home/dzdrale/hypno_env/bin/python "$SCRIPT" --help
echo "=== END DIAGNOSTIC ==="

/home/dzdrale/hypno_env/bin/python "$SCRIPT" \
    --config hyperbolic_pde/configs/hyperbolic_pde_cleps_decoder_shared.yaml \
    --resume_run hyperbolic_pde/runs/hypno_st3/run_20260513_140206