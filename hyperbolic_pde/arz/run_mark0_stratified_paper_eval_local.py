"""Local launcher for arz_mark0_paper_eval on the 5070 Ti laptop.

Runs the STRATIFIED (ic_type, num_segments) paper eval -- the LWR-parity
workflow with ID/OOD cells, per-cell figures, and headline/appendix tables --
for the mark-0 (orig) HypNO-ARZ model, vs the bigger FNO + Godunov + WENO5.

This is the stratified analogue of run_mark0_paper_eval_local.py (which wraps
the older per-dataset arz_orig_paper_eval). Edit the paths below if your
checkpoint / weights live elsewhere.

Usage:
    python hyperbolic_pde/arz/run_mark0_stratified_paper_eval_local.py
    python hyperbolic_pde/arz/run_mark0_stratified_paper_eval_local.py --n_per_cell 20 --no-fno
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# -- edit these paths -------------------------------------------------------- #
CKPT = REPO / "hyperbolic_pde/runs/checkpoint_epoch180.pt"
DATA = REPO / "hyperbolic_pde/arz/data/arz_general_local.npz"
FNO_WEIGHTS = REPO / "hyperbolic_pde/arz/runs/fno_arz_local/fno_arz_bigger.pt"
CONFIG = REPO / "hyperbolic_pde/configs/hyperbolic_pde_arz_local.yaml"
OUT_DIR = REPO / "hyperbolic_pde/arz/runs/mark0_stratified_paper_eval"
# The bigger FNO (width 128, 12 layers) matches the local config's fno_arz.
FNO_SECTION = "fno_arz"
# Config section whose ic_types/num_segments define the ID set. The local
# general dataset has no matching arz_data section by default, so cells read as
# OOD unless you add one or point this at a section that has those keys.
TRAIN_SECTION = "arz_data"
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--fno-weights", type=Path, default=FNO_WEIGHTS)
    ap.add_argument("--fno-section", type=str, default=FNO_SECTION)
    ap.add_argument("--train-section", type=str, default=TRAIN_SECTION)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n_per_cell", type=int, default=None,
                    help="Cap on samples per (ic_type, num_segments) cell.")
    ap.add_argument("--baselines", type=str, default="godunov,weno5")
    ap.add_argument("--no-fno", action="store_true",
                    help="Skip FNO even if weights exist.")
    args = ap.parse_args()

    cmd = [
        sys.executable, "-m", "hyperbolic_pde.arz.arz_mark0_paper_eval",
        "--ckpt", str(args.ckpt),
        "--config", str(args.config),
        "--model-section", "hypno_arz_orig",
        "--data", str(args.data),
        "--baselines", args.baselines,
        "--train-section", args.train_section,
        "--out_dir", str(args.out_dir),
    ]
    if args.n_per_cell is not None:
        cmd += ["--n_per_cell", str(args.n_per_cell)]

    if args.no_fno:
        cmd += ["--no-fno"]
    elif args.fno_weights.exists():
        cmd += [
            "--fno-weights", str(args.fno_weights),
            "--fno-config", str(args.config),
            "--fno-section", args.fno_section,
        ]
    else:
        print(f"[warn] FNO weights not found at {args.fno_weights} -- skipping FNO column")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()
