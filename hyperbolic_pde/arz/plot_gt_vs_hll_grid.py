"""Per-family stratification grid of GT vs HLL vs |error| space-time heatmaps.

For each cell of the datagen stratification grid
    (1-wave type) x (1-wave strength bin) x (contact strength bin)
draw one representative Riemann IC via the SAME sampler the dataset uses
(`_sample_riemann_stratified_cell`), solve it two ways, and show a triptych:

    GT (exact analytic)  |  HLL (reference FV)  |  |HLL - GT|

GT is the exact self-similar solution at cell midpoints (the dataset's single-jump
GT path). HLL is the first-order finite-volume reference flux (`flux_scheme='hll'`,
the paper's ARZ flux and our datagen default). The |error| column isolates where
HLL departs from GT -- a thin stripe along the 1-shock and a WIDENING wedge along
the linearly-degenerate 2-contact (the contact smear grows in time; the 1-wave
self-sharpens). v makes the contact smear most visible.

One figure per (field x wave-type): {rho,v} x {shock,rarefaction} = 4 figures.

Usage (headless):
    .../python.exe hyperbolic_pde/arz/plot_gt_vs_hll_grid.py \
        --pressure-form rho --out-dir figures/gt_vs_hll_grid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz.riemann_arz import solve_riemann_arz_xt
from hyperbolic_pde.arz.datagen_arz import _sample_riemann_stratified_cell
from hyperbolic_pde.arz.eval_vs_numerical_arz import solve_arz_weno5


def _strength_bins(rho_min, rho_max, n_bins):
    """Reproduce the jump-strength bin edges from _build_stratified_riemann_plan."""
    span = rho_max - rho_min
    edges = np.linspace(0.0, span, n_bins + 1)
    edges[0] = min(0.02 * span, 0.5 * edges[1])  # floor: no zero-width 1-wave
    return [(float(edges[b]), float(edges[b + 1])) for b in range(n_bins)]


def _solve_gt_scheme(rL, vL, rR, vR, x0, args):
    """Return dict field -> (GT, scheme, |err|) arrays of shape (nt, nx).

    GT is the exact analytic Riemann solution; the comparison solver is
    args.scheme in {hll, godunov, weno5}.
    """
    wL = float(vL + P.pressure(np.array(float(rL))))
    wR = float(vR + P.pressure(np.array(float(rR))))
    dx = (args.x_max - args.x_min) / args.nx
    xm = args.x_min + (np.arange(args.nx) + 0.5) * dx
    t = np.linspace(0.0, args.t_max, args.nt)

    rA, wA, vA = solve_riemann_arz_xt(rL, wL, rR, wR, xm, t, x0=x0)  # analytic GT

    rho0 = np.where(xm <= x0, rL, rR).astype(float)
    v0 = np.where(xm <= x0, vL, vR).astype(float)
    w0 = v0 + P.pressure(rho0)
    if args.scheme == "weno5":
        rS, wS = solve_arz_weno5(
            rho0, w0, args.x_min, args.x_max, args.t_max, args.nt,
            tau=1e9, cfl=args.cfl, boundary="ghost")
    else:
        _, _, rS, wS = Ref.solve_arz_reference(
            rho0, w0, args.x_min, args.x_max, args.t_max, args.nt,
            tau=1e9, cfl=args.cfl, boundary="ghost", refine=1, flux_scheme=args.scheme)
    vS = wS - P.pressure(rS)  # recover v from (rho, w)

    return {
        "rho": (rA, rS, np.abs(rS - rA)),
        "v": (vA, vS, np.abs(vS - vA)),
    }


def plot_field_wavetype(field, is_shock, bins, args, out_path):
    n = len(bins)
    extent = [args.x_min, args.x_max, args.t_max, 0.0]  # imshow origin upper -> t down
    field_cmap = {"rho": "viridis", "v": "plasma"}[field]
    err_cmap = "magma"
    wave_name = "1-shock" if is_shock else "1-rarefaction"

    rng = np.random.default_rng(args.seed)
    dx = (args.x_max - args.x_min) / args.nx
    xm0 = (args.x_min + (np.arange(args.nx) + 0.5) * dx).astype(np.float64)

    fig = plt.figure(figsize=(4.6 * n, 4.0 * n))
    outer = GridSpec(n, n, figure=fig, hspace=0.55, wspace=0.42)

    for r, jb in enumerate(bins):
        for c, cb in enumerate(bins):
            rL, vL, rR, vR, x0 = _sample_riemann_stratified_cell(
                xm0, rng, args.rho_min, args.rho_max, args.v_min, args.v_max,
                one_wave_is_shock=is_shock,
                one_wave_jump_range=jb, contact_jump_range=cb,
            )
            G, H, E = _solve_gt_scheme(rL, vL, rR, vR, x0, args)[field]
            vlo = float(min(G.min(), H.min()))
            vhi = float(max(G.max(), H.max()))
            elo, ehi = 0.0, float(max(E.max(), 1e-9))

            inner = outer[r, c].subgridspec(1, 3, wspace=0.12)
            axg = fig.add_subplot(inner[0])
            axh = fig.add_subplot(inner[1])
            axe = fig.add_subplot(inner[2])

            imf = axg.imshow(G, aspect="auto", extent=extent, cmap=field_cmap, vmin=vlo, vmax=vhi)
            axh.imshow(H, aspect="auto", extent=extent, cmap=field_cmap, vmin=vlo, vmax=vhi)
            ime = axe.imshow(E, aspect="auto", extent=extent, cmap=err_cmap, vmin=elo, vmax=ehi)

            sch = args.scheme.upper()
            axg.set_title("GT", fontsize=9, pad=3)
            axh.set_title(sch, fontsize=9, pad=3)
            axe.set_title(f"|{sch}−GT|", fontsize=9, pad=3)

            # Axis labels: only outermost to avoid clutter.
            axg.set_ylabel("t", fontsize=8)
            for a in (axh, axe):
                a.set_yticklabels([])
            for a in (axg, axh, axe):
                a.set_xlabel("x", fontsize=8)
                a.tick_params(labelsize=6)

            # Shared field colorbar (GT/HLL) + error colorbar, compact.
            cbf = fig.colorbar(imf, ax=[axg, axh], fraction=0.035, pad=0.02)
            cbf.ax.tick_params(labelsize=6)
            cbe = fig.colorbar(ime, ax=axe, fraction=0.07, pad=0.04)
            cbe.ax.tick_params(labelsize=6)

            mae = float(E.mean())
            axg.text(
                0.0, 1.42,
                f"1-wave |{jb[0]:.2f}–{jb[1]:.2f}|   contact |{cb[0]:.2f}–{cb[1]:.2f}|\n"
                f"mean |{sch}−GT| = {mae:.2e}",
                transform=axg.transAxes, fontsize=8, color="0.25", va="bottom",
            )

    fig.suptitle(
        f"ARZ {wave_name} — {field}(x, t):  GT  vs  {args.scheme.upper()}  vs  |error|\n"
        f"rows = 1-wave strength (top→weak, bottom→strong)   |   "
        f"cols = contact strength (left→weak, right→strong)   |   "
        f"scheme={args.scheme} cfl={args.cfl}  p(ρ)={P.get_pressure_form()}  nx={args.nx}  nt={args.nt}",
        fontsize=14, y=0.998,
    )
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="GT vs HLL vs |error| stratification grids.")
    ap.add_argument("--n-strength-bins", type=int, default=4)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--nt", type=int, default=64)
    ap.add_argument("--x-min", type=float, default=0.0)
    ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--t-max", type=float, default=0.3)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--rho-min", type=float, default=0.1)
    ap.add_argument("--rho-max", type=float, default=0.9)
    ap.add_argument("--v-min", type=float, default=0.0)
    ap.add_argument("--v-max", type=float, default=1.0)
    ap.add_argument("--fields", type=str, default="rho,v")
    ap.add_argument("--scheme", type=str, default="hll", choices=("hll", "godunov", "weno5"),
                    help="Comparison solver vs the analytic GT.")
    ap.add_argument("--pressure-form", type=str, default="rho", choices=("rho", "rho+rho2"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("figures/gt_vs_hll_grid"))
    args = ap.parse_args()

    P.set_pressure_form(args.pressure_form)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bins = _strength_bins(args.rho_min, args.rho_max, args.n_strength_bins)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    print(f"[plot] grid = 2 wave-types x {args.n_strength_bins} 1-wave x "
          f"{args.n_strength_bins} contact  |  scheme={args.scheme} cfl={args.cfl}  "
          f"fields={fields}  p={args.pressure_form}")
    for field in fields:
        for is_shock, wname in [(True, "shock"), (False, "rare")]:
            plot_field_wavetype(
                field, is_shock, bins, args,
                args.out_dir / f"grid_{args.scheme}_{field}_{wname}.png",
            )


if __name__ == "__main__":
    main()
