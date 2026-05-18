"""Preview a few samples from the generated 2D LWR dataset.

Usage:
    python hyperbolic_pde/scripts/preview_dataset_2d.py
    python hyperbolic_pde/scripts/preview_dataset_2d.py --n 6 --frames 6

Saves one PNG per sample to hyperbolic_pde/scripts/_sanity_out/preview_sample_<i>.png
(each PNG has ``frames`` time snapshots across the row).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from hyperbolic_pde.data.fvm_2d import load_dataset_2d

OUT_DIR = ROOT / "scripts" / "_sanity_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview samples from 2D LWR dataset.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "hyperbolic_pde_2d.yaml"))
    parser.add_argument("--n", type=int, default=4, help="Number of samples to preview.")
    parser.add_argument("--frames", type=int, default=6, help="Number of time snapshots per sample.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sample selection.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    path = Path(cfg["data"]["path"])
    if not path.is_absolute():
        path = ROOT.parent / path

    bundle = load_dataset_2d(path)
    N, nt, ny, nx = bundle.u.shape
    print(f"loaded {path}")
    print(f"  u : {bundle.u.shape}  range [{bundle.u.min():.3f}, {bundle.u.max():.3f}]")
    print(f"  x : [{bundle.x[0]:.2f}, {bundle.x[-1]:.2f}], y : [{bundle.y[0]:.2f}, {bundle.y[-1]:.2f}], t : [{bundle.t[0]:.2f}, {bundle.t[-1]:.2f}]")

    rng = np.random.default_rng(args.seed)
    n = min(args.n, N)
    idxs = rng.choice(N, size=n, replace=False)

    frame_count = min(args.frames, nt)
    frame_indices = np.linspace(0, nt - 1, frame_count).round().astype(int)

    extent = (float(bundle.x[0]), float(bundle.x[-1]), float(bundle.y[0]), float(bundle.y[-1]))

    for s_idx in idxs:
        u = bundle.u[s_idx]  # [nt, ny, nx]
        fig, axes = plt.subplots(1, frame_count, figsize=(2.6 * frame_count, 2.8))
        for col, k in enumerate(frame_indices):
            ax = axes[col]
            im = ax.imshow(
                u[k], origin="lower", extent=extent,
                vmin=0.0, vmax=1.0, cmap="viridis",
            )
            ax.set_title(f"t={bundle.t[k]:.2f}")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"sample {s_idx}")
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        out = OUT_DIR / f"preview_sample_{s_idx}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
