"""Gallery of ARZ initial conditions and their evolved solutions.

ARZ analogue of `plot_ic_gallery.py`. For each (IC family, number of
segments) the script samples an initial condition, evolves it with the ARZ
reference solver, and renders the density rho(x,t) and velocity v(x,t)
space--time fields side by side. Intended for the paper writeup, to show what
ARZ solutions look like across IC families and discontinuity counts.

Layout: 2 IC families (rows) x 4 segment counts (cols) = 8 IC samples. Each
cell is a rho|v heatmap pair, so the underlying axes grid is 2 x 8.

    rows = {piecewise_constant_stratified, piecewise_sine}
    cols = segment counts (2, 4, 6, 8)

Usage:
    python hyperbolic_pde/scripts/plot_arz_ic_gallery.py \
        --out figures/arz_ic_gallery.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.datagen_arz import _sample_rho_v_pair, _solve_one

# Rows: multi-discontinuity families (riemann has a single jump, so it is a
# poor fit for a "variety" gallery and is excluded).
FAMILIES = ("piecewise_constant_stratified", "piecewise_sine")
FAMILY_LABELS = {
    "piecewise_constant_stratified": "piecewise constant",
    "piecewise_sine": "piecewise sine",
}
SEGMENT_COUNTS = (2, 4, 6, 8)


def main() -> None:
    p = argparse.ArgumentParser(description="ARZ IC gallery for the writeup.")
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--nt", type=int, default=128)
    p.add_argument("--t_max", type=float, default=1.0)
    p.add_argument("--x_min", type=float, default=-1.0)
    p.add_argument("--x_max", type=float, default=1.0)
    p.add_argument("--rho_min", type=float, default=0.1)
    p.add_argument("--rho_max", type=float, default=0.9)
    p.add_argument("--v_min", type=float, default=0.1)
    p.add_argument("--v_max", type=float, default=0.9)
    p.add_argument("--tau", type=float, default=0.1,
                   help="Relaxation time for the reference solver.")
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--boundary", type=str, default="ghost")
    p.add_argument("--refine", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="figures/arz_ic_gallery.png")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    x = np.linspace(args.x_min, args.x_max, args.nx, dtype=np.float64)

    n_rows = len(FAMILIES)
    n_cols = len(SEGMENT_COUNTS)
    # Each cell is a rho|v pair -> 2 sub-columns per logical column.
    fig, axes = plt.subplots(
        n_rows, 2 * n_cols,
        figsize=(2.4 * 2 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
    )

    extent = [x[0], x[-1], 0.0, args.t_max]

    for r, family in enumerate(FAMILIES):
        for c, n_seg in enumerate(SEGMENT_COUNTS):
            rho0, v0 = _sample_rho_v_pair(
                family, x, n_seg, rng,
                args.rho_min, args.rho_max, args.v_min, args.v_max,
            )
            w0 = v0 + P.pressure(rho0)
            rho_hist, w_hist = _solve_one(
                family, rho0, w0,
                args.x_min, args.x_max, args.t_max, args.nt,
                tau=args.tau, cfl=args.cfl, boundary=args.boundary,
                refine=args.refine, use_exact_riemann=False,
            )
            v_hist = w_hist - P.pressure(rho_hist)

            # omega = v / rho; check if it stays constant over time
            omega_hist = v_hist / np.where(rho_hist > 1e-12, rho_hist, np.nan)
            print(
                f"  [{family[:20]:20s}, seg={n_seg}] "
                f"omega: min={np.nanmin(omega_hist):.4f}  max={np.nanmax(omega_hist):.4f}  "
                f"std={np.nanstd(omega_hist):.4f}  "
                f"(t=0: mean={np.nanmean(omega_hist[0]):.4f}, t=T: mean={np.nanmean(omega_hist[-1]):.4f})"
            )

            ax_rho = axes[r, 2 * c]
            ax_v = axes[r, 2 * c + 1]

            im_rho = ax_rho.imshow(
                rho_hist, extent=extent, origin="lower", aspect="auto",
                cmap="viridis", interpolation="nearest",
                vmin=args.rho_min, vmax=args.rho_max,
            )
            im_v = ax_v.imshow(
                v_hist, extent=extent, origin="lower", aspect="auto",
                cmap="cividis", interpolation="nearest",
            )

            ax_rho.set_title(rf"$\rho$  ({n_seg} seg)", fontsize=8)
            ax_v.set_title(rf"$v$  ({n_seg} seg)", fontsize=8)
            for ax in (ax_rho, ax_v):
                ax.set_xlabel("x", fontsize=7)
                ax.tick_params(labelsize=6)
            ax_rho.set_ylabel("t", fontsize=7)

            fig.colorbar(im_rho, ax=ax_rho, shrink=0.8)
            fig.colorbar(im_v, ax=ax_v, shrink=0.8)

        # Row label (family) on the left-most axis.
        axes[r, 0].text(
            -0.55, 0.5, FAMILY_LABELS[family],
            transform=axes[r, 0].transAxes,
            rotation=90, va="center", ha="center",
            fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        r"ARZ initial conditions and their evolved solutions "
        r"$\rho(x,t)$, $v(x,t)$  "
        rf"($\tau={args.tau}$)",
        fontsize=12,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
