"""Local launcher for arz_orig_paper_eval on the 5070 Ti laptop.

Runs HypNO-orig (mark0) vs FNO-small vs Godunov vs WENO5 on the local
homogeneous dataset (arz_general_local.npz). Edit CKPT to point at the
checkpoint you scp'd from the cluster.

Usage:
    python hyperbolic_pde/arz/run_mark0_paper_eval_local.py
    python hyperbolic_pde/arz/run_mark0_paper_eval_local.py --n-samples 20 --n-plots 5
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# ── edit these paths ──────────────────────────────────────────────────────────
CKPT = REPO / "hyperbolic_pde/arz/runs/hypno_arz_orig/checkpoint_epoch170.pt"
DATA_HOMOG = REPO / "hyperbolic_pde/arz/data/arz_general_local.npz"
FNO_WEIGHTS = REPO / "hyperbolic_pde/arz/runs/fno_arz_small/fno_arz.pt"
CONFIG = REPO / "hyperbolic_pde/configs/hyperbolic_pde_arz_local.yaml"
OUT_DIR = REPO / "hyperbolic_pde/arz/runs/mark0_paper_eval"
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--data", type=Path, default=DATA_HOMOG)
    ap.add_argument("--fno-weights", type=Path, default=FNO_WEIGHTS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-plots", type=int, default=10)
    ap.add_argument("--baselines", type=str, default="godunov,weno5")
    ap.add_argument("--no-fno", action="store_true",
                    help="Skip FNO even if weights exist.")
    args = ap.parse_args()

    # The HypNO config is auto-located from the run dir's config.yaml (scp'd
    # alongside the checkpoint). The FNO config comes from the local yaml.
    cmd = [
        sys.executable, "-m", "hyperbolic_pde.arz.arz_orig_paper_eval",
        "--ckpt", str(args.ckpt),
        "--model-section", "hypno_arz_orig",
        "--homog-data", str(args.data),
        "--fno-config", str(CONFIG),
        "--fno-section", "fno_arz_small",
        "--baselines", args.baselines,
        "--n-samples", str(args.n_samples),
        "--n-plots", str(args.n_plots),
        "--out_dir", str(args.out_dir),
    ]

    if not args.no_fno and args.fno_weights.exists():
        cmd += ["--fno-weights", str(args.fno_weights)]
    elif not args.no_fno:
        print(f"[warn] FNO weights not found at {args.fno_weights} -- skipping FNO column")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO), check=True)

if __name__ == "__main__":
    main()
