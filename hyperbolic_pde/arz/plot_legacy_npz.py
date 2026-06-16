"""Plot the fields saved by run_legacy_2d5.py -- offline, no model/dataset reload.

Reads the .npz (keys: rho, v, w predictions; rho_gt, w_gt; x, t) and writes the
same per-sample 3x3 space-time panels (rows rho/v/w, cols pred|GT|err). v_gt is
reconstructed as w_gt - p(rho_gt); pass --pressure-form to match how the data
was generated (default rho -> v_gt = w_gt - rho_gt).

    python -m hyperbolic_pde.arz.plot_legacy_npz \
        --npz /home/dzdrale/scratch/results/legacy2d5_outputs.npz \
        --figures /home/dzdrale/scratch/results/figs_legacy2d5 --n-plots 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _p(rho, form):
    return rho if form == "rho" else rho + rho * rho


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--figures", type=Path, required=True)
    ap.add_argument("--n-plots", type=int, default=5)
    ap.add_argument("--pressure-form", default="rho", choices=["rho", "rho+rho2"])
    args = ap.parse_args()

    d = np.load(args.npz)
    rho, v, w = d["rho"], d["v"], d["w"]
    rho_gt, w_gt = d["rho_gt"], d["w_gt"]
    v_gt = w_gt - _p(rho_gt, "rho" if args.pressure_form == "rho" else "rho+rho2")
    x, t = d["x"].astype(np.float64), d["t"].astype(np.float64)

    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    channels = [("rho", rho, rho_gt), ("v", v, v_gt), ("w", w, w_gt)]
    n_fig = min(args.n_plots, rho.shape[0])

    for i in range(n_fig):
        fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
        for row, (name, preds, gts) in enumerate(channels):
            pred = preds[i]; gt = gts[i]; err = pred - gt
            vmin = min(pred.min(), gt.min()); vmax = max(pred.max(), gt.max())
            im0 = axes[row, 0].pcolormesh(x, t, pred, shading="auto",
                                          vmin=vmin, vmax=vmax, cmap="viridis")
            axes[row, 0].set_title(f"{name} pred"); fig.colorbar(im0, ax=axes[row, 0])
            im1 = axes[row, 1].pcolormesh(x, t, gt, shading="auto",
                                          vmin=vmin, vmax=vmax, cmap="viridis")
            axes[row, 1].set_title(f"{name} GT"); fig.colorbar(im1, ax=axes[row, 1])
            amax = float(np.abs(err).max()) or 1.0
            im2 = axes[row, 2].pcolormesh(x, t, err, shading="auto",
                                          vmin=-amax, vmax=amax, cmap="RdBu_r")
            axes[row, 2].set_title(f"{name} err (MAE {np.mean(np.abs(err)):.3e})")
            fig.colorbar(im2, ax=axes[row, 2])
            for c in range(3):
                axes[row, c].set_xlabel("x"); axes[row, c].set_ylabel("t")
        fig.suptitle(f"legacy2d5  sample {i}")
        fig.savefig(fig_dir / f"legacy2d5_sample_{i}.png", dpi=130)
        plt.close(fig)
    print(f"wrote {n_fig} figures -> {fig_dir}")


if __name__ == "__main__":
    main()
