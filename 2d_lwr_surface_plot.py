"""3D surface plot of rho(x, y, t) for one 2D LWR case.

Reuses the solver from ``2d_lwr_visualization.py``. Renders rho as a 3D
surface (z = rho) at one or more snapshot times in a single PNG, so the
patch's deformation into shocks/rarefactions is visible in relief.

Run locally::

    python 2d_lwr_surface_plot.py
    python 2d_lwr_surface_plot.py --case circular_patch --times 0,0.5,1.0
    python 2d_lwr_surface_plot.py --stride 2 --elev 35 --azim -60

Output lands at ``outputs/2d_lwr_<case>_surface.png``.
"""
from __future__ import annotations

import argparse
import os
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Import the existing solver so dynamics match the static plots.
_viz = import_module("2d_lwr_visualization")
step = _viz.step
make_initial_condition = _viz.make_initial_condition
N = _viz.N
CFL = _viz.CFL
T = _viz.T
BETA = _viz.BETA
OUT_DIR = _viz.OUT_DIR


def solve_at_times(case: str, snapshot_times):
    """Solve the chosen case and return snapshots at the nearest steps."""
    dx = 2.0 / N
    dy = 2.0 / N
    xs = np.linspace(-1.0 + 0.5 * dx, 1.0 - 0.5 * dx, N)
    ys = np.linspace(-1.0 + 0.5 * dy, 1.0 - 0.5 * dy, N)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    rho, _ = make_initial_condition(case, X, Y)

    dt = CFL / (1.0 / dx + BETA / dy)
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps                                # land exactly on T
    target_steps = sorted({max(int(round(ts / dt)), 0) for ts in snapshot_times})
    target_set = set(target_steps)

    saved = {}
    if 0 in target_set:
        saved[0] = rho.copy()
    for n in range(1, n_steps + 1):
        rho = step(rho, dx, dy, dt, BETA)
        if n in target_set:
            saved[n] = rho.copy()

    out = []
    for ts in snapshot_times:
        k = max(int(round(ts / dt)), 0)
        out.append((k * dt, saved[k]))
    return X, Y, out


def plot_surfaces(case, X, Y, snaps, stride, elev, azim, out_path):
    n = len(snaps)
    n_cols = min(n, 2)
    n_rows = int(np.ceil(n / n_cols))
    fig = plt.figure(figsize=(7.0 * n_cols, 5.5 * n_rows), constrained_layout=True)

    # Downsample to keep the surface readable; full N=256 mesh looks like mush.
    Xs = X[::stride, ::stride]
    Ys = Y[::stride, ::stride]

    for i, (t, rho) in enumerate(snaps):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        Zs = rho[::stride, ::stride]
        surf = ax.plot_surface(
            Xs, Ys, Zs,
            cmap=cm.viridis, vmin=0.0, vmax=1.0,
            linewidth=0, antialiased=True, rstride=1, cstride=1,
        )
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel(r"$\rho$")
        ax.set_title(f"t = {t:.3f}")
        ax.view_init(elev=elev, azim=azim)

    # Single shared colorbar.
    cbar = fig.colorbar(surf, ax=fig.get_axes(), shrink=0.6, pad=0.02,
                        label=r"$\rho$")
    cbar.mappable.set_clim(0.0, 1.0)
    fig.suptitle(f"2D LWR — {case} — rho(x,y,t) as 3D surface", fontsize=13)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _parse_times(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(
        description="3D surface plot of rho(x,y,t) for a 2D LWR case."
    )
    p.add_argument("--case", type=str, default="circular_patch",
                   choices=("planar_shock", "planar_rarefaction",
                            "oblique_shock", "oblique_rarefaction",
                            "circular_patch", "quadrant_riemann",
                            "fast_oblique_shock"))
    p.add_argument("--times", type=_parse_times, default=[0.0, 0.33, 0.66, 1.0],
                   help="Comma-separated snapshot times in [0,1] "
                        "(default 0,0.33,0.66,1.0).")
    p.add_argument("--stride", type=int, default=4,
                   help="Downsample the surface mesh by this stride for "
                        "readability (default 4 => 64x64 of 256x256).")
    p.add_argument("--elev", type=float, default=30.0,
                   help="3D view elevation in degrees (default 30).")
    p.add_argument("--azim", type=float, default=-55.0,
                   help="3D view azimuth in degrees (default -55).")
    args = p.parse_args()

    X, Y, snaps = solve_at_times(args.case, args.times)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"2d_lwr_{args.case}_surface.png")
    plot_surfaces(args.case, X, Y, snaps, args.stride, args.elev, args.azim,
                  out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
