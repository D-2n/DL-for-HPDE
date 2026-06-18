"""Plot Godunov-FV vs HLL-FV vs analytic GT across the datagen stratification grid.

For each cell of the stratified-Riemann plan used by `generate_arz_riemann_exact_dataset`
(see `_build_stratified_riemann_plan`):

    (1-wave type)  x  (1-wave strength bin)  x  (contact strength bin)
        2         x          n_bins          x          n_bins

we draw ONE representative Riemann IC via the SAME sampler the dataset uses
(`_sample_riemann_stratified_cell`), then solve it three ways and overlay the
final-time rho profile:

    * analytic  : exact self-similar solution at cell midpoints (the GOLD GT,
                  same path the single-jump dataset subset uses)
    * godunov   : first-order FV with exact-Riemann (Godunov) interface flux
    * hll       : first-order FV with HLL interface flux (the paper's ARZ flux)

This makes the contact-smearing visible per wave-structure cell: Godunov sharpens
the genuinely-nonlinear 1-shock vs HLL, but resolves the linearly-degenerate
2-contact no better than HLL. Shocks (1-wave) get sharper; contacts stay fat.

One figure per 1-wave type (shock / rarefaction): a (n_bins x n_bins) grid of
panels, rows = 1-wave strength, cols = contact strength.

Usage (headless -> PNGs):
    /c/Users/dimit/anaconda3/python.exe hyperbolic_pde/arz/plot_godunov_strata.py \
        --n-strength-bins 4 --nx 256 --t-max 0.15 --out-dir figures/godunov_strata
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz.riemann_arz import solve_riemann_arz_xt
from hyperbolic_pde.arz.datagen_arz import _sample_riemann_stratified_cell


def _strength_bins(rho_min: float, rho_max: float, n_bins: int):
    """Reproduce the jump-strength bin edges from _build_stratified_riemann_plan."""
    jump_span = rho_max - rho_min
    edges = np.linspace(0.0, jump_span, n_bins + 1)
    edges[0] = min(0.02 * jump_span, 0.5 * edges[1])  # floor the first bin (no 0-width 1-wave)
    return [(float(edges[b]), float(edges[b + 1])) for b in range(n_bins)]


def _solve_three(rho_L, v_L, rho_R, v_R, x0, x_min, x_max, t_max, nx, nt, cfl):
    """Return (x_mid, t, analytic, godunov, hll) dicts of rho/v/w final-time rows.

    analytic is sampled at cell midpoints (no FV); godunov/hll are first-order FV.
    """
    w_L = float(v_L + P.pressure(np.array(float(rho_L))))
    w_R = float(v_R + P.pressure(np.array(float(rho_R))))
    dx = (x_max - x_min) / nx
    xm = x_min + (np.arange(nx) + 0.5) * dx
    t = np.linspace(0.0, t_max, nt)

    rA, wA, vA = solve_riemann_arz_xt(rho_L, w_L, rho_R, w_R, xm, t, x0=x0)

    rho0 = np.where(xm <= x0, rho_L, rho_R).astype(float)
    v0 = np.where(xm <= x0, v_L, v_R).astype(float)
    w0 = v0 + P.pressure(rho0)

    _, _, rG, wG = Ref.solve_arz_reference(
        rho0, w0, x_min, x_max, t_max, nt, tau=1e9, cfl=cfl,
        boundary="ghost", refine=1, flux_scheme="godunov",
    )
    _, _, rH, wH = Ref.solve_arz_reference(
        rho0, w0, x_min, x_max, t_max, nt, tau=1e9, cfl=cfl,
        boundary="ghost", refine=1, flux_scheme="hll",
    )
    return xm, t, (rA, vA, wA), (rG, wG), (rH, wH)


def plot_wave_type(is_shock, bins, args, rng, out_path):
    n = len(bins)
    fig, axes = plt.subplots(n, n, figsize=(3.4 * n, 2.7 * n), squeeze=False)
    wave_name = "1-shock" if is_shock else "1-rarefaction"

    for r, jb in enumerate(bins):          # rows: 1-wave strength
        for c, cb in enumerate(bins):      # cols: contact strength
            ax = axes[r][c]
            dx = (args.x_max - args.x_min) / args.nx
            xm0 = (args.x_min + (np.arange(args.nx) + 0.5) * dx).astype(np.float64)
            rho_L, v_L, rho_R, v_R, x0 = _sample_riemann_stratified_cell(
                xm0, rng, args.rho_min, args.rho_max, args.v_min, args.v_max,
                one_wave_is_shock=is_shock,
                one_wave_jump_range=jb, contact_jump_range=cb,
            )
            xm, t, (rA, vA, wA), (rG, wG), (rH, wH) = _solve_three(
                rho_L, v_L, rho_R, v_R, x0,
                args.x_min, args.x_max, args.t_max, args.nx, args.nt, args.cfl,
            )
            eG = float(np.abs(rG[-1] - rA[-1]).mean())
            eH = float(np.abs(rH[-1] - rA[-1]).mean())

            ax.plot(xm, rA[-1], color="k", lw=2.0, label="analytic")
            ax.plot(xm, rG[-1], color="tab:blue", lw=1.3, ls="--", label="godunov")
            ax.plot(xm, rH[-1], color="tab:red", lw=1.3, ls=":", label="hll")
            ax.plot(xm, rA[0], color="0.6", lw=0.8, alpha=0.7, label="IC")
            ax.set_title(
                f"1w|{jb[0]:.2f}-{jb[1]:.2f}  c|{cb[0]:.2f}-{cb[1]:.2f}\n"
                f"G={eG:.1e} H={eH:.1e} (G/H={eG/max(eH,1e-12):.2f})",
                fontsize=8,
            )
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"ARZ {wave_name}: rho at t={args.t_max:g}  |  rows=1-wave strength, "
        f"cols=contact strength  |  p={P.get_pressure_form()}  nx={args.nx} cfl={args.cfl}",
        fontsize=12,
    )
    for r in range(n):
        axes[r][0].set_ylabel(f"1w bin {r}\nrho", fontsize=8)
    for c in range(n):
        axes[n - 1][c].set_xlabel(f"contact bin {c}\nx", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot Godunov/HLL/analytic across datagen strata.")
    ap.add_argument("--n-strength-bins", type=int, default=4)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--nt", type=int, default=2)
    ap.add_argument("--x-min", type=float, default=0.0)
    ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--t-max", type=float, default=0.15)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--rho-min", type=float, default=0.1)
    ap.add_argument("--rho-max", type=float, default=0.9)
    ap.add_argument("--v-min", type=float, default=0.0)
    ap.add_argument("--v-max", type=float, default=1.0)
    ap.add_argument("--pressure-form", type=str, default="rho", choices=("rho", "rho+rho2"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("figures/godunov_strata"))
    args = ap.parse_args()

    P.set_pressure_form(args.pressure_form)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bins = _strength_bins(args.rho_min, args.rho_max, args.n_strength_bins)
    rng = np.random.default_rng(args.seed)

    print(f"[plot] grid = 2 wave-types x {args.n_strength_bins} 1-wave bins x "
          f"{args.n_strength_bins} contact bins  |  p={args.pressure_form}")
    plot_wave_type(True, bins, args, rng, args.out_dir / "godunov_strata_1shock.png")
    plot_wave_type(False, bins, args, rng, args.out_dir / "godunov_strata_1rarefaction.png")


if __name__ == "__main__":
    main()
