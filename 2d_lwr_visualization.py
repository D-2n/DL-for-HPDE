"""2D LWR-type scalar conservation law: visual examples of shocks and rarefactions.

Solves
    rho_t + d_x f(rho) + d_y g(rho) = 0,    f(rho)=rho(1-rho), g(rho)=beta*rho(1-rho)
on (x,y) in [-1,1]^2, t in [0,1], with Rusanov / local Lax-Friedrichs fluxes and
zero-gradient (outflow) boundary conditions.

In 2D, a shock at fixed time is a curve in physical space. In space-time (x,y,t)
that moving curve sweeps out a surface. For planar initial data, the 2D problem
reduces to a 1D Riemann problem in the normal coordinate xi = n_x*x + n_y*y; the
normal shock speed is
    s_n = (F(rho_R) - F(rho_L)).n / (rho_R - rho_L).
For LWR (concave flux), rho_L < rho_R gives a shock; rho_L > rho_R gives a
rarefaction.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

# ---- domain / discretization (spec-driven) -------------------------------- #
N = 256
CFL = 0.45
T = 1.0
BETA = 0.7
SNAPSHOT_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
OUT_DIR = "outputs"
CASES = [
    "planar_shock",
    "planar_rarefaction",
    "oblique_shock",
    "oblique_rarefaction",
    "circular_patch",
    "quadrant_riemann",
    "fast_oblique_shock",
]
CASE_TITLES = {
    "planar_shock":        "1. Planar shock (x)",
    "planar_rarefaction":  "2. Planar rarefaction (x)",
    "oblique_shock":       "3. Oblique shock (n=(1,1)/sqrt2)",
    "oblique_rarefaction": "4. Oblique rarefaction (n=(1,1)/sqrt2)",
    "circular_patch":      "5. Circular patch (curved shock)",
    "quadrant_riemann":    "6. Quadrant Riemann (wave interaction)",
    "fast_oblique_shock":  "7. Fast oblique shock (extremal states)",
}


# --------------------------------------------------------------------------- #
# Fluxes
# --------------------------------------------------------------------------- #
def flux_x(rho):
    return rho * (1.0 - rho)


def flux_y(rho, beta=BETA):
    return beta * rho * (1.0 - rho)


def rusanov_flux_x(left, right):
    """Rusanov flux on x-faces. f'(rho) = 1 - 2*rho."""
    a = np.maximum(np.abs(1.0 - 2.0 * left), np.abs(1.0 - 2.0 * right))
    return 0.5 * (flux_x(left) + flux_x(right)) - 0.5 * a * (right - left)


def rusanov_flux_y(bottom, top, beta=BETA):
    """Rusanov flux on y-faces. g'(rho) = beta*(1 - 2*rho)."""
    a = beta * np.maximum(np.abs(1.0 - 2.0 * bottom), np.abs(1.0 - 2.0 * top))
    return 0.5 * (flux_y(bottom, beta) + flux_y(top, beta)) - 0.5 * a * (top - bottom)


# --------------------------------------------------------------------------- #
# Time stepping
# --------------------------------------------------------------------------- #
def step(rho, dx, dy, dt, beta=BETA):
    """One unsplit conservative FV update with Rusanov fluxes and outflow BCs."""
    # Pad by one cell on each side with edge values => zero-gradient BC.
    rp = np.pad(rho, 1, mode="edge")

    # x-faces: there are N+1 internal faces per row (between rp[:, i] and rp[:, i+1])
    # but we only need the (N+1) faces touching the interior strip rp[1:-1, :].
    left  = rp[1:-1, :-1]
    right = rp[1:-1, 1:]
    F = rusanov_flux_x(left, right)            # shape (N, N+1)

    # y-faces likewise.
    bottom = rp[:-1, 1:-1]
    top    = rp[1:,  1:-1]
    G = rusanov_flux_y(bottom, top, beta)      # shape (N+1, N)

    rho_new = rho - (dt / dx) * (F[:, 1:] - F[:, :-1]) \
                  - (dt / dy) * (G[1:, :] - G[:-1, :])
    # light clip to [0,1] to suppress tiny numerical overshoots
    np.clip(rho_new, 0.0, 1.0, out=rho_new)
    return rho_new


def solve(rho0, T, dx, dy, beta, cfl, snapshot_times):
    """Integrate to T, returning snapshots at the requested times (nearest step)."""
    dt = cfl / (1.0 / dx + beta / dy)          # spec's CFL formula
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps                            # land exactly on T

    # Index of nearest step for each snapshot time (0 included).
    target_steps = sorted({int(round(ts / dt)) for ts in snapshot_times})
    target_set = set(target_steps)

    snapshots = {}
    rho = rho0.copy()
    if 0 in target_set:
        snapshots[0] = rho.copy()
    for n in range(1, n_steps + 1):
        rho = step(rho, dx, dy, dt, beta)
        if n in target_set:
            snapshots[n] = rho.copy()

    # Map back to the requested times.
    out_times, out_fields = [], []
    for ts in snapshot_times:
        k = int(round(ts / dt))
        out_times.append(k * dt)
        out_fields.append(snapshots[k])
    return out_times, out_fields


# --------------------------------------------------------------------------- #
# Initial conditions
# --------------------------------------------------------------------------- #
def make_initial_condition(case_name, X, Y):
    """Return (rho0, theory_dict). theory_dict carries overlay info or is empty."""
    if case_name == "planar_shock":
        rL, rR = 0.1, 0.6
        rho0 = np.where(X < 0.0, rL, rR).astype(np.float64)
        s = 1.0 - rL - rR                       # = 0.3
        return rho0, {"kind": "shock_x", "rL": rL, "rR": rR, "s": s}

    if case_name == "planar_rarefaction":
        rL, rR = 0.8, 0.2
        rho0 = np.where(X < 0.0, rL, rR).astype(np.float64)
        lamL = 1.0 - 2.0 * rL                   # = -0.6
        lamR = 1.0 - 2.0 * rR                   # = +0.6
        return rho0, {"kind": "rarefaction_x", "rL": rL, "rR": rR,
                      "lamL": lamL, "lamR": lamR}

    if case_name == "oblique_shock":
        rL, rR = 0.1, 0.6
        nx, ny = 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)
        xi = nx * X + ny * Y
        rho0 = np.where(xi < 0.0, rL, rR).astype(np.float64)
        s_n = (nx + BETA * ny) * (1.0 - rL - rR)
        return rho0, {"kind": "shock_oblique", "rL": rL, "rR": rR,
                      "nx": nx, "ny": ny, "s_n": s_n}

    if case_name == "oblique_rarefaction":
        rL, rR = 0.8, 0.2
        nx, ny = 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)
        xi = nx * X + ny * Y
        rho0 = np.where(xi < 0.0, rL, rR).astype(np.float64)
        lamL = (nx + BETA * ny) * (1.0 - 2.0 * rL)
        lamR = (nx + BETA * ny) * (1.0 - 2.0 * rR)
        return rho0, {"kind": "rarefaction_oblique", "rL": rL, "rR": rR,
                      "nx": nx, "ny": ny, "lamL": lamL, "lamR": lamR}

    if case_name == "circular_patch":
        r = np.sqrt((X + 0.25) ** 2 + Y ** 2)
        rho0 = np.where(r < 0.35, 0.75, 0.20).astype(np.float64)
        return rho0, {}                          # no theory overlay

    if case_name == "quadrant_riemann":
        rho0 = np.empty_like(X, dtype=np.float64)
        rho0[(X <  0.0) & (Y <  0.0)] = 0.8
        rho0[(X >= 0.0) & (Y <  0.0)] = 0.2
        rho0[(X <  0.0) & (Y >= 0.0)] = 0.6
        rho0[(X >= 0.0) & (Y >= 0.0)] = 0.1
        return rho0, {}

    if case_name == "fast_oblique_shock":
        # Both states near the low-density extremum -> jump small (admissible
        # shock, rho_L < rho_R) but |1 - rho_L - rho_R| ~ 1, so the normal
        # speed s_n = (n_x + beta*n_y)(1 - rho_L - rho_R) ~ 1.14 -- the
        # shock crosses the whole [-1,1] domain inside t in [0,1].
        rL, rR = 0.0, 0.05
        nx, ny = 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)
        # Start the shock at the lower-left corner so it has the full domain
        # diagonal to traverse.
        xi = nx * X + ny * Y - (nx * (-0.9) + ny * (-0.9))
        rho0 = np.where(xi < 0.0, rL, rR).astype(np.float64)
        s_n = (nx + BETA * ny) * (1.0 - rL - rR)
        return rho0, {"kind": "shock_oblique_offset", "rL": rL, "rR": rR,
                      "nx": nx, "ny": ny, "s_n": s_n,
                      "x0": -0.9, "y0": -0.9}

    raise ValueError(f"unknown case: {case_name}")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _overlay_theory(ax, theory, t, extent=(-1.0, 1.0, -1.0, 1.0)):
    """Overlay shock line / rarefaction-fan boundaries on a single panel."""
    if not theory or t <= 0.0:
        return
    xlim = (extent[0], extent[1])
    ylim = (extent[2], extent[3])

    kind = theory["kind"]
    if kind == "shock_x":
        x = theory["s"] * t
        ax.axvline(x, color="k", linestyle="--", linewidth=1.2)

    elif kind == "rarefaction_x":
        for lam in (theory["lamL"], theory["lamR"]):
            ax.axvline(lam * t, color="k", linestyle="--", linewidth=1.2)

    elif kind == "shock_oblique":
        # Line: n_x * x + n_y * y = s_n * t  =>  y = (s_n*t - n_x*x) / n_y
        xs = np.linspace(xlim[0], xlim[1], 200)
        ys = (theory["s_n"] * t - theory["nx"] * xs) / theory["ny"]
        m = (ys >= ylim[0]) & (ys <= ylim[1])
        ax.plot(xs[m], ys[m], "k--", linewidth=1.2)

    elif kind == "shock_oblique_offset":
        # Initial line passes through (x0, y0), so the constant in
        # n.x + n.y = c + s_n * t is c = n_x*x0 + n_y*y0.
        c = theory["nx"] * theory["x0"] + theory["ny"] * theory["y0"]
        xs = np.linspace(xlim[0], xlim[1], 200)
        ys = (c + theory["s_n"] * t - theory["nx"] * xs) / theory["ny"]
        m = (ys >= ylim[0]) & (ys <= ylim[1])
        ax.plot(xs[m], ys[m], "k--", linewidth=1.2)

    elif kind == "rarefaction_oblique":
        xs = np.linspace(xlim[0], xlim[1], 200)
        for lam in (theory["lamL"], theory["lamR"]):
            ys = (lam * t - theory["nx"] * xs) / theory["ny"]
            m = (ys >= ylim[0]) & (ys <= ylim[1])
            ax.plot(xs[m], ys[m], "k--", linewidth=1.2)


def plot_case(case_name, snapshots, times, theory, out_path):
    """One row, five panels: rho(x,y,t) at each snapshot time."""
    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4.2),
                             constrained_layout=True)
    for ax, t, rho in zip(axes, times, snapshots):
        im = ax.imshow(rho, origin="lower", extent=[-1, 1, -1, 1],
                       vmin=0.0, vmax=1.0, interpolation="nearest", cmap="viridis")
        _overlay_theory(ax, theory, t)
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.85, label=r"$\rho$")
    fig.suptitle(CASE_TITLES[case_name])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary(all_results, out_path):
    """6 rows (one per case) x 5 columns (snapshot times)."""
    n_rows = len(all_results)
    n_cols = len(SNAPSHOT_TIMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows),
                             constrained_layout=True)
    for r, (case_name, (times, snaps, theory)) in enumerate(all_results.items()):
        for c, (t, rho) in enumerate(zip(times, snaps)):
            ax = axes[r, c]
            im = ax.imshow(rho, origin="lower", extent=[-1, 1, -1, 1],
                           vmin=0.0, vmax=1.0, interpolation="nearest", cmap="viridis")
            _overlay_theory(ax, theory, t)
            if r == 0:
                ax.set_title(f"t = {t:.2f}")
            if c == 0:
                ax.set_ylabel(CASE_TITLES[case_name], fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label=r"$\rho$")
    fig.suptitle("2D LWR: shocks, rarefactions, and interactions")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gradient_diagnostics(all_results, out_path, dx, dy):
    """|grad rho| at t=1 for each case (centered differences). Shocks become
    bright curves; rarefactions appear as broad smooth bands."""
    cases = list(all_results.keys())
    n = len(cases)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 4.3 * n_rows),
                             constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, case_name in zip(axes_flat, cases):
        _, snaps, _ = all_results[case_name]
        rho_T = snaps[-1]
        # centered diffs (edge values for boundary, matching outflow BCs)
        grx = np.zeros_like(rho_T)
        gry = np.zeros_like(rho_T)
        grx[:, 1:-1] = (rho_T[:, 2:] - rho_T[:, :-2]) / (2.0 * dx)
        gry[1:-1, :] = (rho_T[2:, :] - rho_T[:-2, :]) / (2.0 * dy)
        mag = np.sqrt(grx ** 2 + gry ** 2)
        im = ax.imshow(mag, origin="lower", extent=[-1, 1, -1, 1],
                       interpolation="nearest", cmap="magma")
        ax.set_title(CASE_TITLES[case_name], fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85, label=r"$|\nabla\rho|$")
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(r"$|\nabla\rho|$ at $t=1$: shocks as curves, rarefactions as bands")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Cell-centered grid on [-1, 1]^2 with N cells per side.
    dx = 2.0 / N
    dy = 2.0 / N
    xs = np.linspace(-1.0 + 0.5 * dx, 1.0 - 0.5 * dx, N)
    ys = np.linspace(-1.0 + 0.5 * dy, 1.0 - 0.5 * dy, N)
    X, Y = np.meshgrid(xs, ys, indexing="xy")   # imshow with origin=lower wants this

    all_results = {}
    for case_name in CASES:
        print(f"[{case_name}] building IC + solving ...", flush=True)
        rho0, theory = make_initial_condition(case_name, X, Y)
        times, snaps = solve(rho0, T, dx, dy, BETA, CFL, SNAPSHOT_TIMES)
        out_path = os.path.join(OUT_DIR, f"2d_lwr_{case_name}.png")
        plot_case(case_name, snaps, times, theory, out_path)
        print(f"  saved {out_path}")
        all_results[case_name] = (times, snaps, theory)

    summary_path = os.path.join(OUT_DIR, "2d_lwr_summary.png")
    plot_summary(all_results, summary_path)
    print(f"saved {summary_path}")

    grad_path = os.path.join(OUT_DIR, "2d_lwr_gradients.png")
    plot_gradient_diagnostics(all_results, grad_path, dx, dy)
    print(f"saved {grad_path}")


if __name__ == "__main__":
    main()
