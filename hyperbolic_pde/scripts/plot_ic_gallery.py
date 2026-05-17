"""Gallery of LWR initial conditions and their FVM-evolved solutions.

For each (IC type, number of segments) the script generates an initial
condition, evolves it with the conservation-law FVM solver, and plots the
space-time field u(x, t) -- the t=0 row is the IC itself, the interior
shows the shocks and rarefactions it develops.

Rows  = IC type   (piecewise_constant, piecewise_sine, riemann)
Cols  = num_segments (riemann ignores this -- always one discontinuity)

Output: a single figure `ic_gallery.png` (or --out), intended for the
paper writeup.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hyperbolic_pde.data.fvm import (
    piecewise_constant_ic,
    piecewise_sine_ic,
    riemann_ic,
    solve_conservation_fvm,
)

IC_TYPES = {
    "piecewise_constant": piecewise_constant_ic,
    "piecewise_sine": piecewise_sine_ic,
    "riemann": riemann_ic,
}
SEGMENT_COUNTS = (2, 5, 7, 10)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--nt", type=int, default=128)
    p.add_argument("--t_max", type=float, default=1.0)
    p.add_argument("--x_min", type=float, default=-1.0)
    p.add_argument("--x_max", type=float, default=1.0)
    p.add_argument("--u_min", type=float, default=0.0)
    p.add_argument("--u_max", type=float, default=1.0)
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--boundary", type=str, default="periodic")
    p.add_argument("--method", type=str, default="godunov")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="ic_gallery.png")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    x = np.linspace(args.x_min, args.x_max, args.nx, dtype=np.float32)

    n_rows = len(IC_TYPES)
    n_cols = len(SEGMENT_COUNTS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
    )

    for r, (ic_name, ic_fn) in enumerate(IC_TYPES.items()):
        for c, n_seg in enumerate(SEGMENT_COUNTS):
            ax = axes[r, c]
            u0 = ic_fn(
                x, num_segments=n_seg,
                u_min=args.u_min, u_max=args.u_max, rng=rng,
            )
            x_grid, t_grid, u_truth = solve_conservation_fvm(
                u0, args.x_min, args.x_max, args.t_max, args.nt,
                cfl=args.cfl, boundary=args.boundary, method=args.method,
            )
            im = ax.pcolormesh(
                x_grid, t_grid, u_truth, shading="auto",
                cmap="jet", vmin=0.0, vmax=1.0,
            )
            if ic_name == "riemann":
                title = f"{ic_name}  (1 discontinuity)"
            else:
                title = f"{ic_name}  ({n_seg} segments)"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.colorbar(im, ax=ax)

    fig.suptitle("LWR initial conditions and their evolved solutions u(x, t)")
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
