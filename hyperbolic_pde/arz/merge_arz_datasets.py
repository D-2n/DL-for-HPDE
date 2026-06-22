"""Merge two ARZ dataset npz files into one.

Usage:
    python -m hyperbolic_pde.arz.merge_arz_datasets \
        --a /path/to/dataset_a.npz \
        --b /path/to/dataset_b.npz \
        --out /path/to/merged.npz

Both datasets must share the same grid (nx, nt, x, t) and pressure form.
Arrays concatenated along axis 0: rho, w, v, rho0, w0, v0, num_segments, ic_type.
Scalar metadata (tau, p_form) taken from dataset A; a warning is printed if B differs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def merge(path_a: Path, path_b: Path, out: Path) -> None:
    a = np.load(path_a, allow_pickle=True)
    b = np.load(path_b, allow_pickle=True)

    # Sanity checks.
    for key in ("x", "t"):
        if not np.allclose(a[key], b[key], atol=1e-6):
            raise ValueError(f"Grid mismatch on '{key}': datasets have different coordinates.")

    for key in ("tau", "p_form"):
        va = str(a[key]) if key in a else "missing"
        vb = str(b[key]) if key in b else "missing"
        if va != vb:
            print(f"[merge] WARNING: '{key}' differs: A={va!r}  B={vb!r} — using A's value.")

    arrays = {}
    for key in ("x", "t"):
        arrays[key] = a[key]

    for key in ("rho", "w", "v", "rho0", "w0", "v0", "num_segments", "ic_type"):
        if key in a and key in b:
            arrays[key] = np.concatenate([a[key], b[key]], axis=0)
        elif key in a:
            print(f"[merge] WARNING: '{key}' missing from B — keeping A only.")
            arrays[key] = a[key]
        elif key in b:
            print(f"[merge] WARNING: '{key}' missing from A — keeping B only.")
            arrays[key] = b[key]

    for key in ("tau", "p_form"):
        if key in a:
            arrays[key] = a[key]

    n_a = int(a["rho"].shape[0])
    n_b = int(b["rho"].shape[0])
    print(f"[merge] A={n_a} samples  B={n_b} samples  -> merged={n_a+n_b} samples")
    print(f"[merge] IC type counts in merged set:")
    for ic in np.unique(arrays["ic_type"]):
        print(f"  {ic}: {int(np.sum(arrays['ic_type'] == ic))}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"[merge] saved -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    merge(args.a, args.b, args.out)


if __name__ == "__main__":
    main()
