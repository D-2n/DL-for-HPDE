"""Space-time isosurface of rho for a 2D LWR case.

A shock in 2D is a curve in (x, y) that moves over time; in (x, y, t)-space
it sweeps out a 2-surface. This script renders that surface directly:

    1. solve the FV scheme to T, saving rho on every timestep so we have a
       dense (nt, ny, nx) volume,
    2. extract the level set { (x, y, t) : rho(x, y, t) = iso } via marching
       cubes (skimage.measure.marching_cubes), and
    3. plot the triangulated surface inside a 3D box with axes x, y, t.

Default isolevel is 0.475 = midpoint of the circular-patch states (0.20,
0.75), so the compression-side shock surface shows up clearly.

Run locally::

    python 2d_lwr_spacetime_surface.py
    python 2d_lwr_spacetime_surface.py --iso 0.5
    python 2d_lwr_spacetime_surface.py --case oblique_shock --iso 0.35

Output: ``outputs/2d_lwr_<case>_spacetime.png``.
"""
from __future__ import annotations

import argparse
import os
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Reuse the solver / IC builders so dynamics match the static plots.
_viz = import_module("2d_lwr_visualization")
step = _viz.step
make_initial_condition = _viz.make_initial_condition
N = _viz.N
CFL = _viz.CFL
T = _viz.T
BETA = _viz.BETA
OUT_DIR = _viz.OUT_DIR


def solve_volume(case: str, t_stride: int):
    """Solve the chosen case, returning (X, Y, t_arr, vol) with
    vol of shape (nt, ny, nx). ``t_stride`` skips every k-th solver step
    when recording to keep memory and the marching-cubes mesh manageable."""
    dx = 2.0 / N
    dy = 2.0 / N
    xs = np.linspace(-1.0 + 0.5 * dx, 1.0 - 0.5 * dx, N)
    ys = np.linspace(-1.0 + 0.5 * dy, 1.0 - 0.5 * dy, N)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    rho, _ = make_initial_condition(case, X, Y)

    dt = CFL / (1.0 / dx + BETA / dy)
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps

    times = [0.0]
    vol = [rho.copy()]
    for n in range(1, n_steps + 1):
        rho = step(rho, dx, dy, dt, BETA)
        if n % t_stride == 0 or n == n_steps:
            times.append(n * dt)
            vol.append(rho.copy())

    print(f"steps={n_steps} dt={dt:.5g} recorded_frames={len(vol)}")
    return X, Y, np.array(times), np.array(vol), dx, dy


def extract_isosurface(vol, t_arr, dx, dy):
    """Marching cubes on the (nt, ny, nx) volume. Returns (verts_xyt, faces, vals).

    verts come out in voxel-index space (i_t, i_y, i_x); we map them back to
    physical coordinates (x, y, t) on the cell-centered mesh.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError as e:
        raise SystemExit(
            "This script needs scikit-image (skimage.measure.marching_cubes). "
            "Install with: conda install -c conda-forge scikit-image"
        ) from e

    nt, ny, nx = vol.shape
    iso_min = float(vol.min())
    iso_max = float(vol.max())
    return iso_min, iso_max, marching_cubes, nt, ny, nx


def voxel_to_physical(verts, t_arr, dx, dy):
    """Map (i_t, i_y, i_x) marching-cubes verts to (x, y, t)."""
    i_t = verts[:, 0]
    i_y = verts[:, 1]
    i_x = verts[:, 2]
    # cell-centered x, y on [-1, 1]; t from the recorded times.
    x = -1.0 + (i_x + 0.5) * dx
    y = -1.0 + (i_y + 0.5) * dy
    # Linear interp into t_arr by float index (marching cubes returns fractional
    # voxel coordinates between recorded slices).
    i0 = np.floor(i_t).astype(int).clip(0, len(t_arr) - 1)
    i1 = np.minimum(i0 + 1, len(t_arr) - 1)
    frac = i_t - i0
    t = (1.0 - frac) * t_arr[i0] + frac * t_arr[i1]
    return x, y, t


def plot_spacetime(case, X, Y, t_arr, vol, iso, elev, azim, out_path):
    iso_min, iso_max, marching_cubes, nt, ny, nx = extract_isosurface(
        vol, t_arr, X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0]
    )
    if not (iso_min < iso < iso_max):
        raise SystemExit(
            f"iso={iso} is outside the volume's value range "
            f"[{iso_min:.3f}, {iso_max:.3f}] — pick something inside."
        )

    verts, faces, normals, _ = marching_cubes(vol, level=iso)
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    xv, yv, tv = voxel_to_physical(verts, t_arr, dx, dy)

    fig = plt.figure(figsize=(8.5, 7.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # Color faces by their mean t so the surface reads as "moving in time".
    tri_t = tv[faces].mean(axis=1)
    norm = plt.Normalize(vmin=float(t_arr[0]), vmax=float(t_arr[-1]))
    colors = cm.plasma(norm(tri_t))

    tri_verts = np.stack([xv[faces], yv[faces], tv[faces]], axis=-1)  # (F, 3, 3)
    coll = Poly3DCollection(tri_verts, facecolors=colors,
                            edgecolors="none", linewidths=0)
    coll.set_alpha(0.85)
    ax.add_collection3d(coll)

    # Bound the box and label axes.
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(float(t_arr[0]), float(t_arr[-1]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title(
        f"2D LWR — {case}\n"
        f"isosurface  rho(x,y,t) = {iso:.3f}   (shock manifold in space-time)"
    )
    ax.view_init(elev=elev, azim=azim)

    # Colorbar keyed to t.
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08, label="t")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Space-time isosurface of rho for a 2D LWR case."
    )
    p.add_argument("--case", type=str, default="circular_patch",
                   choices=("planar_shock", "planar_rarefaction",
                            "oblique_shock", "oblique_rarefaction",
                            "circular_patch", "quadrant_riemann",
                            "fast_oblique_shock"))
    p.add_argument("--iso", type=float, default=0.475,
                   help="rho level for the isosurface. Default 0.475 "
                        "(midpoint of circular-patch states 0.20 and 0.75).")
    p.add_argument("--t-stride", type=int, default=2,
                   help="Record every N-th solver step into the volume "
                        "(default 2). Lower = smoother surface + more memory.")
    p.add_argument("--elev", type=float, default=20.0,
                   help="3D view elevation in degrees (default 20).")
    p.add_argument("--azim", type=float, default=-60.0,
                   help="3D view azimuth in degrees (default -60).")
    args = p.parse_args()

    X, Y, t_arr, vol, dx, dy = solve_volume(args.case, args.t_stride)
    print(f"volume shape: (nt, ny, nx) = {vol.shape}, "
          f"rho range = [{vol.min():.3f}, {vol.max():.3f}]")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"2d_lwr_{args.case}_spacetime.png")
    plot_spacetime(args.case, X, Y, t_arr, vol, args.iso,
                   args.elev, args.azim, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
