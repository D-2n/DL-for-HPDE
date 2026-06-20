"""Space-time (x-t) heatmaps of ARZ schemes vs the exact Riemann ground truth.

For each Riemann case and each field (rho, v, w), lay out one x-t heatmap per
scheme next to the exact GT, so the full space-time evolution is visible (not
just a final-time slice). WFT should be visually identical to the exact GT;
the FV schemes smear the discontinuities (HLL most, then weno5/muscl).

One figure per field. Rows = cases, columns = [exact, wft, muscl, weno5, hll]
(restrict via --schemes). A shared colour scale per (case, field) row makes the
smearing directly comparable; an optional --error mode plots |scheme - GT|.

Usage:
    python hyperbolic_pde/arz/plot_scheme_heatmaps_arz.py \
        --pressure-form rho --schemes wft,muscl,weno5,hll \
        --out-dir figures/scheme_sweep/heatmaps
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as R
from hyperbolic_pde.arz.riemann_arz import solve_riemann_arz_xt
from hyperbolic_pde.arz.wft_arz import solve_arz_wft_riemann_xt


CASES = {
    "1-shock + contact":       dict(rL=0.30, vL=0.85, rR=0.70, vR=0.30),
    "1-rarefaction + contact": dict(rL=0.75, vL=0.25, rR=0.25, vR=0.65),
    "pure 2-contact":          dict(rL=0.30, vL=0.50, rR=0.75, vR=0.50),
    "strong two-wave":         dict(rL=0.85, vL=0.10, rR=0.15, vR=0.90),
}

CMAPS = {"rho": "viridis", "v": "cividis", "w": "magma"}
LABELS = {"rho": r"$\rho$", "v": r"$v$", "w": r"$w$"}


def _run_full(scheme, rL, vL, rR, vR, x0, xm, t, x_min, x_max, t_max, nt,
              cfl, rare_delta):
    """Return full-field (rho, v, w) each shape (nt, nx), or None on failure."""
    rho0 = np.where(xm <= x0, rL, rR).astype(float)
    v0 = np.where(xm <= x0, vL, vR).astype(float)
    w0 = v0 + P.pressure(rho0)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if scheme == "wft":
                rH, wH, vH = solve_arz_wft_riemann_xt(
                    rL, vL, rR, vR, xm, t, x0=x0, rare_delta=rare_delta)
                return rH, vH, wH
            if scheme == "muscl":
                rH, wH = R.solve_arz_muscl(rho0, w0, x_min, x_max, t_max, nt,
                                           tau=1e9, cfl=cfl, boundary="ghost")
            elif scheme == "weno5":
                rH, wH = R.solve_arz_weno5(rho0, w0, x_min, x_max, t_max, nt,
                                           tau=1e9, cfl=cfl, boundary="ghost")
            elif scheme == "hll":
                _, _, rH, wH = R.solve_arz_reference(
                    rho0, w0, x_min, x_max, t_max, nt_out=nt, tau=1e9,
                    cfl=cfl, boundary="ghost", refine=1, flux_scheme="hll")
            elif scheme == "hllc":
                _, _, rH, wH = R.solve_arz_reference(
                    rho0, w0, x_min, x_max, t_max, nt_out=nt, tau=1e9,
                    cfl=cfl, boundary="ghost", refine=1, flux_scheme="hllc")
            else:
                raise ValueError(scheme)
            if not (np.isfinite(rH).all() and np.isfinite(wH).all()):
                raise FloatingPointError("non-finite")
        vH = wH - P.pressure(rH)
        return rH, vH, wH
    except (FloatingPointError, RuntimeError, ZeroDivisionError):
        return None


_DEFAULT_CFL = {"muscl": 0.4, "weno5": 0.2, "hll": 0.4, "hllc": 0.4, "wft": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pressure-form", default="rho", choices=("rho", "rho+rho2"))
    ap.add_argument("--schemes", default="wft,muscl,weno5,hll",
                    help="comma list from wft,muscl,weno5,hll,hllc")
    ap.add_argument("--fields", default="rho,v,w",
                    help="comma list from rho,v,w")
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--t-max", type=float, default=0.3)
    ap.add_argument("--x0", type=float, default=0.5)
    ap.add_argument("--rare-delta", type=float, default=0.002)
    ap.add_argument("--error", action="store_true",
                    help="plot |scheme - exact GT| instead of the raw field")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("figures/scheme_sweep/heatmaps"))
    args = ap.parse_args()

    P.set_pressure_form(args.pressure_form)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    x_min, x_max = 0.0, 1.0
    dx = (x_max - x_min) / args.nx
    xm = x_min + (np.arange(args.nx) + 0.5) * dx
    t = np.linspace(0.0, args.t_max, args.nt)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Solve once per (case, scheme); cache full fields.
    cache = {}   # (case, scheme) -> dict(field -> (nt,nx)) or None
    gt = {}      # case -> dict(field -> (nt,nx))
    for cname, c in CASES.items():
        rL, vL, rR, vR = c["rL"], c["vL"], c["rR"], c["vR"]
        wL = float(vL + P.pressure(np.array(rL)))
        wR = float(vR + P.pressure(np.array(rR)))
        rA, wA, vA = solve_riemann_arz_xt(rL, wL, rR, wR, xm, t, x0=args.x0)
        gt[cname] = {"rho": rA, "v": vA, "w": wA}
        for scheme in schemes:
            out = _run_full(scheme, rL, vL, rR, vR, args.x0, xm, t,
                            x_min, x_max, args.t_max, args.nt,
                            _DEFAULT_CFL.get(scheme, 0.4), args.rare_delta)
            cache[(cname, scheme)] = (
                None if out is None else {"rho": out[0], "v": out[1], "w": out[2]})

    # Columns: exact GT first (unless --error), then each scheme.
    cols = (["exact"] if not args.error else []) + schemes
    extent = [x_min, x_max, 0.0, args.t_max]

    for fld in fields:
        ncase, ncol = len(CASES), len(cols)
        fig, axes = plt.subplots(ncase, ncol,
                                 figsize=(2.9 * ncol, 2.7 * ncase),
                                 squeeze=False)
        for r, cname in enumerate(CASES):
            gfield = gt[cname][fld]
            if args.error:
                # Per-row shared scale across schemes' error.
                errs = {}
                for scheme in schemes:
                    cur = cache[(cname, scheme)]
                    errs[scheme] = (None if cur is None
                                    else np.abs(cur[fld] - gfield))
                finite = [e for e in errs.values() if e is not None]
                vmax = max((float(e.max()) for e in finite), default=1.0) or 1.0
                for ci, scheme in enumerate(cols):
                    ax = axes[r][ci]
                    e = errs[scheme]
                    if e is None:
                        ax.text(0.5, 0.5, "CRASH", ha="center", va="center",
                                transform=ax.transAxes, color="red")
                        ax.set_xticks([]); ax.set_yticks([])
                    else:
                        im = ax.imshow(e, origin="lower", aspect="auto",
                                       extent=extent, cmap="magma",
                                       vmin=0.0, vmax=vmax)
                    if r == 0:
                        ax.set_title(f"|{scheme} - GT|", fontsize=10)
            else:
                # Per-row shared scale across exact + schemes (raw field).
                vmin = float(gfield.min()); vmax = float(gfield.max())
                for scheme in schemes:
                    cur = cache[(cname, scheme)]
                    if cur is not None:
                        vmin = min(vmin, float(cur[fld].min()))
                        vmax = max(vmax, float(cur[fld].max()))
                if vmax - vmin < 1e-9:
                    vmax = vmin + 1e-9
                for ci, col in enumerate(cols):
                    ax = axes[r][ci]
                    if col == "exact":
                        field = gfield
                    else:
                        cur = cache[(cname, col)]
                        field = None if cur is None else cur[fld]
                    if field is None:
                        ax.text(0.5, 0.5, "CRASH", ha="center", va="center",
                                transform=ax.transAxes, color="red")
                        ax.set_xticks([]); ax.set_yticks([])
                    else:
                        im = ax.imshow(field, origin="lower", aspect="auto",
                                       extent=extent, cmap=CMAPS[fld],
                                       vmin=vmin, vmax=vmax)
                    if r == 0:
                        ax.set_title(col, fontsize=10)
            axes[r][0].set_ylabel(f"{cname}\n\nt", fontsize=8)
            for ci in range(ncol):
                axes[r][ci].set_xlabel("x", fontsize=8)
            # One colorbar per row (rightmost axis).
            try:
                fig.colorbar(im, ax=axes[r].tolist(), fraction=0.025, pad=0.01)
            except Exception:
                pass

        mode = "error" if args.error else "field"
        fig.suptitle(
            f"ARZ x-t heatmaps — {LABELS[fld]} ({mode})  "
            f"p={args.pressure_form}, nx={args.nx}, nt={args.nt}, "
            f"t_max={args.t_max}", fontsize=12)
        out = args.out_dir / (
            f"heatmap_{fld}_{'error' if args.error else 'field'}_"
            f"{args.pressure_form.replace('+','')}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[heatmap] wrote {out}")


if __name__ == "__main__":
    main()
