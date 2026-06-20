"""Head-to-head ARZ scheme comparison against the exact Riemann ground truth.

Schemes compared (all vs the analytic `solve_riemann_arz_xt` GT):
    wft    - wave-front tracking (NEW). Machine-precision: exact on shocks &
             contacts, error = rare_delta on rarefactions only. No grid -> no
             diffusion, no vacuum/overshoot instability.
    muscl  - 2nd-order MUSCL-Hancock + HLLC (NEW FV). Contact-sharp, robust.
    weno5  - 5th-order WENO + global Lax-Friedrichs (the old "better" scheme;
             diffusive splitting, crashes on extreme states).
    hll    - 1st-order HLL (the old default; very diffusive, contact overshoot).
             Included only as a reference floor.

Produces:
  * a CSV of final-time L1 error (rho, v, w) per case x scheme
  * an overlay figure: each scheme's final-time profile vs exact GT
  * a console table

Usage:
    python hyperbolic_pde/arz/compare_schemes_arz.py --pressure-form rho \
        --out-fig figures/scheme_sweep/wft_vs_fv.png \
        --out-csv figures/scheme_sweep/wft_vs_fv.csv
"""
from __future__ import annotations

import argparse
import csv
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

SCHEMES = {
    "wft":   dict(color="tab:green",  cfl=None, ls="-",  lw=1.6),
    "muscl": dict(color="tab:blue",   cfl=0.4,  ls="-",  lw=1.3),
    "weno5": dict(color="tab:orange", cfl=0.2,  ls="-",  lw=1.3),
    "hll":   dict(color="tab:red",    cfl=0.4,  ls=":",  lw=1.1),
}


def _run(scheme, rL, vL, rR, vR, x0, xm, t, x_min, x_max, t_max, nt, cfl, rare_delta):
    """Return (rho_T, v_T, w_T) at final time, or None on failure."""
    rho0 = np.where(xm <= x0, rL, rR).astype(float)
    v0 = np.where(xm <= x0, vL, vR).astype(float)
    w0 = v0 + P.pressure(rho0)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if scheme == "wft":
                rH, wH, vH = solve_arz_wft_riemann_xt(
                    rL, vL, rR, vR, xm, t, x0=x0, rare_delta=rare_delta)
                return rH[-1], vH[-1], wH[-1]
            if scheme == "muscl":
                rH, wH = R.solve_arz_muscl(rho0, w0, x_min, x_max, t_max, nt,
                                           tau=1e9, cfl=cfl, boundary="ghost")
            elif scheme == "weno5":
                rH, wH = R.solve_arz_weno5(rho0, w0, x_min, x_max, t_max, nt,
                                           tau=1e9, cfl=cfl, boundary="ghost")
            else:  # hll
                _, _, rH, wH = R.solve_arz_reference(
                    rho0, w0, x_min, x_max, t_max, nt_out=nt, tau=1e9,
                    cfl=cfl, boundary="ghost", refine=1, flux_scheme="hll")
            if not (np.isfinite(rH).all() and np.isfinite(wH).all()):
                raise FloatingPointError("non-finite")
        vH = wH - P.pressure(rH)
        return rH[-1], vH[-1], wH[-1]
    except (FloatingPointError, RuntimeError, ZeroDivisionError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pressure-form", default="rho", choices=("rho", "rho+rho2"))
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--nt", type=int, default=64)
    ap.add_argument("--t-max", type=float, default=0.3)
    ap.add_argument("--x0", type=float, default=0.5)
    ap.add_argument("--rare-delta", type=float, default=0.002,
                    help="WFT fan resolution (its only error source).")
    ap.add_argument("--out-fig", type=Path,
                    default=Path("figures/scheme_sweep/wft_vs_fv.png"))
    ap.add_argument("--out-csv", type=Path,
                    default=Path("figures/scheme_sweep/wft_vs_fv.csv"))
    args = ap.parse_args()

    P.set_pressure_form(args.pressure_form)
    x_min, x_max = 0.0, 1.0
    dx = (x_max - x_min) / args.nx
    xm = x_min + (np.arange(args.nx) + 0.5) * dx
    t = np.linspace(0.0, args.t_max, args.nt)

    fields = ["rho", "v", "w"]
    fig, axes = plt.subplots(len(CASES), 3, figsize=(15, 3.2 * len(CASES)),
                             squeeze=False)
    rows = []

    for r, (cname, c) in enumerate(CASES.items()):
        rL, vL, rR, vR = c["rL"], c["vL"], c["rR"], c["vR"]
        wL = float(vL + P.pressure(np.array(rL)))
        wR = float(vR + P.pressure(np.array(rR)))
        rA, wA, vA = solve_riemann_arz_xt(rL, wL, rR, wR, xm, t, x0=args.x0)
        gt = {"rho": rA[-1], "v": vA[-1], "w": wA[-1]}

        for fi, fld in enumerate(fields):
            axes[r][fi].plot(xm, gt[fld], "k-", lw=2.6, label="exact GT", zorder=6)

        for scheme, sp in SCHEMES.items():
            out = _run(scheme, rL, vL, rR, vR, args.x0, xm, t,
                       x_min, x_max, args.t_max, args.nt, sp["cfl"], args.rare_delta)
            if out is None:
                rows.append(dict(case=cname, scheme=scheme, L1_rho="CRASH",
                                 L1_v="CRASH", L1_w="CRASH"))
                continue
            rT, vT, wT = out
            cur = {"rho": rT, "v": vT, "w": wT}
            rows.append(dict(
                case=cname, scheme=scheme,
                L1_rho=float(np.abs(cur["rho"] - gt["rho"]).mean()),
                L1_v=float(np.abs(cur["v"] - gt["v"]).mean()),
                L1_w=float(np.abs(cur["w"] - gt["w"]).mean()),
            ))
            for fi, fld in enumerate(fields):
                axes[r][fi].plot(xm, cur[fld], color=sp["color"], ls=sp["ls"],
                                 lw=sp["lw"], alpha=0.9, label=scheme)

        for fi, fld in enumerate(fields):
            ax = axes[r][fi]
            ax.set_title(f"{cname}  —  {fld}", fontsize=10)
            ax.grid(alpha=0.25)
            if r == 0 and fi == 0:
                ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"ARZ schemes vs exact Riemann GT  (p={args.pressure_form}, "
        f"nx={args.nx}, t={args.t_max}, WFT rare_delta={args.rare_delta})",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=130)
    print(f"[fig] wrote {args.out_fig}")

    with args.out_csv.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["case", "scheme", "L1_rho", "L1_v", "L1_w"])
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"[csv] wrote {args.out_csv}")

    # Console table.
    print(f"\n=== final-time L1 vs exact GT (lower=better)  p={args.pressure_form} ===")
    print(f"{'case':26s} {'scheme':7s} {'L1(rho)':>12s} {'L1(v)':>12s} {'L1(w)':>12s}")
    for row in rows:
        def fmt(v):
            return f"{v:12.4e}" if isinstance(v, float) else f"{v:>12s}"
        print(f"{row['case']:26s} {row['scheme']:7s} "
              f"{fmt(row['L1_rho'])} {fmt(row['L1_v'])} {fmt(row['L1_w'])}")


if __name__ == "__main__":
    main()
