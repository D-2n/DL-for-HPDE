"""Diagnostic: classify each sample of an ARZ Riemann dataset.

For each (rho_L, w_L, rho_R, w_R) it reports whether the 1-wave is a shock,
rarefaction, or degenerate. Prints summary counts and a few example indices
per class.

Usage:
    python hyperbolic_pde/scripts/diag_arz_riemann.py \
        --npz /home/dzdrale/scratch/arz_1d/arz_riemann_trial.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.riemann_arz import _rho_star_from_w_minus_v


def classify(rho_L, w_L, rho_R, w_R):
    v_L = w_L - P.pressure(rho_L)
    v_R = w_R - P.pressure(rho_R)
    rho_star = float(_rho_star_from_w_minus_v(w_L - v_R))
    v_star = v_R
    lam1_L = v_L - rho_L * P.dpressure(rho_L)
    lam1_star = v_star - rho_star * P.dpressure(rho_star)
    if lam1_L > lam1_star + 1e-12:
        return "shock", lam1_L, lam1_star, rho_star
    elif lam1_L < lam1_star - 1e-12:
        return "rare", lam1_L, lam1_star, rho_star
    return "degen", lam1_L, lam1_star, rho_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--show", type=int, default=5,
                    help="How many example indices to show per class.")
    args = ap.parse_args()

    b = load_arz_dataset(args.npz)
    N, nt, nx = b.rho.shape
    x = b.x
    print(f"[diag] N={N}, nt={nt}, nx={nx}, tau={b.tau}")

    counts = {"shock": 0, "rare": 0, "degen": 0}
    examples = {"shock": [], "rare": [], "degen": []}
    v_diffs = []   # v_R - v_L at the chosen interface

    for i in range(N):
        rho0 = b.rho0[i]
        w0   = b.w0[i]
        d = np.abs(np.diff(rho0)) + np.abs(np.diff(w0))
        j = int(np.argmax(d))
        rho_L, w_L = float(rho0[j]),     float(w0[j])
        rho_R, w_R = float(rho0[j + 1]), float(w0[j + 1])
        cls, lam1_L, lam1_star, rho_star = classify(rho_L, w_L, rho_R, w_R)
        counts[cls] += 1
        if len(examples[cls]) < args.show:
            examples[cls].append((i, rho_L, w_L, rho_R, w_R, lam1_L, lam1_star, rho_star))
        v_diffs.append((w_R - P.pressure(rho_R)) - (w_L - P.pressure(rho_L)))

    print("\nCounts:")
    for k, c in counts.items():
        print(f"  {k:6s}: {c:5d} ({100.0 * c / N:5.1f}%)")

    v_diffs = np.array(v_diffs)
    print(f"\nv_R - v_L : mean={v_diffs.mean():+.3f}  std={v_diffs.std():.3f}  "
          f"frac(v_L < v_R) = {(v_diffs > 0).mean():.2%}  "
          f"(rarefaction should occur when v_L < v_R)")

    for cls, rows in examples.items():
        if not rows:
            continue
        print(f"\n=== Example {cls} samples ===")
        for (i, rL, wL, rR, wR, l1L, l1s, rs) in rows:
            vL = wL - P.pressure(rL); vR = wR - P.pressure(rR)
            print(f"  i={i:4d}  rho_L={rL:.3f} v_L={vL:+.3f}  |  rho_R={rR:.3f} v_R={vR:+.3f}"
                  f"  ->  rho*={rs:.3f}  lam1_L={l1L:+.3f}  lam1*={l1s:+.3f}")


if __name__ == "__main__":
    main()
