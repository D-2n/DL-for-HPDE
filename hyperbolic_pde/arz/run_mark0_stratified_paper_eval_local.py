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
# Held-out OOD eval set (different seed from the training arz_general_local.npz).
# Generate it first:  python -m hyperbolic_pde.arz.gen_ood_eval_local
DATA = REPO / "hyperbolic_pde/arz/data/arz_ood_eval_local.npz"
FNO_WEIGHTS = REPO / "hyperbolic_pde/arz/runs/fno_arz_local/fno_arz_bigger.pt"
# CONFIG = the model's run config (must have a hypno_arz_orig section). The
# checkpoint was scp'd as a bare state_dict, so we point at the run config that
# came with it. Its filename is ckpt_180_cfg.yaml (the loader only auto-locates a
# sibling literally named config.yaml, so we pass this explicitly).
CONFIG = REPO / "hyperbolic_pde/runs/ckpt_180_cfg.yaml"
# FNO_CONFIG = the config whose fno_arz section matches fno_arz_bigger.pt
# (width 128, modes 24, 12 layers). The run config above may carry an
# 8-layer fno_arz, so keep this separate.
FNO_CONFIG = REPO / "hyperbolic_pde/configs/hyperbolic_pde_arz_local.yaml"
OUT_DIR = REPO / "hyperbolic_pde/arz/runs/mark0_stratified_paper_eval"
# The bigger FNO (width 128, 12 layers) matches FNO_CONFIG's fno_arz section.
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
    ap.add_argument("--config", type=Path, default=CONFIG,
                    help="Model run config (must have hypno_arz_orig).")
    ap.add_argument("--fno-weights", type=Path, default=FNO_WEIGHTS)
    ap.add_argument("--fno-config", type=Path, default=FNO_CONFIG,
                    help="Config whose fno_arz section matches the FNO weights.")
    ap.add_argument("--fno-section", type=str, default=FNO_SECTION)
    ap.add_argument("--train-section", type=str, default=TRAIN_SECTION)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n_per_cell", type=int, default=None,
                    help="Cap on samples per (ic_type, num_segments) cell.")
    # WENO5 is unstable on ARZ (overshoots -> NaN -> native access violation that
    # crashes the whole process, not a catchable Python exception), especially on
    # high-segment-count / OOD ICs. Default to godunov only; pass
    # --baselines godunov,weno5 to re-enable WENO5 at your own risk.
    ap.add_argument("--baselines", type=str, default="godunov")
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
            "--fno-config", str(args.fno_config),
            "--fno-section", args.fno_section,
        ]
    else:
        print(f"[warn] FNO weights not found at {args.fno_weights} -- skipping FNO column")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()
