"""Resolution convergence: scheme error vs grid size, per wave-type.

Datasets live at modest nx (~64-256). This sweep asks: at the resolutions we
actually generate data, which scheme wins, and how does the gap scale with nx?
High-order WENO5 should pull ahead on smooth (rarefaction) parts as nx grows;
on discontinuities all schemes are limited by the O(dx) kink. We measure L1 vs
the exact analytic GT at each nx for a fixed CFL per scheme.

Usage:
    .../python.exe hyperbolic_pde/arz/sweep_resolution.py \
        --nxs 32,64,128,256,512 --pressure-form rho \
        --out figures/scheme_sweep/resolution.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz.riemann_arz import solve_riemann_arz_xt
from hyperbolic_pde.arz.datagen_arz import _sample_riemann_stratified_cell
from hyperbolic_pde.arz.eval_vs_numerical_arz import solve_arz_weno5


def _strength_bins(rho_min, rho_max, n_bins):
    span = rho_max - rho_min
    edges = np.linspace(0.0, span, n_bins + 1)
    edges[0] = min(0.02 * span, 0.5 * edges[1])
    return [(float(edges[b]), float(edges[b + 1])) for b in range(n_bins)]


def _run(scheme, rho0, w0, x_min, x_max, t_max, nt, cfl):
    if scheme in ("hll", "godunov"):
        _, _, rH, wH = Ref.solve_arz_reference(
            rho0, w0, x_min, x_max, t_max, nt_out=nt, tau=1e9, cfl=cfl,
            boundary="ghost", refine=1, flux_scheme=scheme)
        return rH, wH - P.pressure(rH)
    if scheme == "weno5":
        rW, wW = solve_arz_weno5(rho0, w0, x_min, x_max, t_max, nt_out=nt,
                                 tau=1e9, cfl=cfl, boundary="ghost")
        return rW, wW - P.pressure(rW)
    raise ValueError(scheme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", type=str, default="hll,godunov,weno5")
    ap.add_argument("--cfl-fv", type=float, default=0.4, help="CFL for hll/godunov")
    ap.add_argument("--cfl-weno", type=float, default=0.4, help="CFL for weno5")
    ap.add_argument("--nxs", type=str, default="32,64,128,256,512")
    ap.add_argument("--n-strength-bins", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nt", type=int, default=64)
    ap.add_argument("--x-min", type=float, default=0.0)
    ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--t-max", type=float, default=0.3)
    ap.add_argument("--rho-min", type=float, default=0.1)
    ap.add_argument("--rho-max", type=float, default=0.9)
    ap.add_argument("--v-min", type=float, default=0.0)
    ap.add_argument("--v-max", type=float, default=1.0)
    ap.add_argument("--pressure-form", type=str, default="rho", choices=("rho", "rho+rho2"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("figures/scheme_sweep/resolution.csv"))
    args = ap.parse_args()

    P.set_pressure_form(args.pressure_form)
    bins = _strength_bins(args.rho_min, args.rho_max, args.n_strength_bins)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    nxs = [int(n) for n in args.nxs.split(",") if n.strip()]

    acc = {}  # (scheme,nx,wtype,field) -> list L1
    for nx in nxs:
        dx = (args.x_max - args.x_min) / nx
        xm = (args.x_min + (np.arange(nx) + 0.5) * dx).astype(np.float64)
        t = np.linspace(0.0, args.t_max, args.nt)
        for is_shock in (True, False):
            wtype = "shock" if is_shock else "rare"
            for jb in bins:
                for cb in bins:
                    bin_seed = (args.seed + (1 if is_shock else 0) * 100003
                                + int(jb[0] * 1e4) * 31 + int(cb[0] * 1e4))
                    rng = np.random.default_rng(bin_seed)
                    for _ in range(args.seeds):
                        rL, vL, rR, vR, x0 = _sample_riemann_stratified_cell(
                            xm, rng, args.rho_min, args.rho_max, args.v_min, args.v_max,
                            one_wave_is_shock=is_shock,
                            one_wave_jump_range=jb, contact_jump_range=cb)
                        wL = float(vL + P.pressure(np.array(float(rL))))
                        wR = float(vR + P.pressure(np.array(float(rR))))
                        rA, wA, vA = solve_riemann_arz_xt(rL, wL, rR, wR, xm, t, x0=x0)
                        rho0 = np.where(xm <= x0, rL, rR).astype(float)
                        v0 = np.where(xm <= x0, vL, vR).astype(float)
                        w0 = v0 + P.pressure(rho0)
                        for scheme in schemes:
                            cfl = args.cfl_weno if scheme == "weno5" else args.cfl_fv
                            try:
                                with np.errstate(over="raise", invalid="raise", divide="raise"):
                                    rS, vS = _run(scheme, rho0, w0, args.x_min, args.x_max,
                                                  args.t_max, args.nt, cfl)
                                if not (np.isfinite(rS).all() and np.isfinite(vS).all()):
                                    raise FloatingPointError
                            except (FloatingPointError, RuntimeError):
                                continue
                            acc.setdefault((scheme, nx, wtype, "rho"), []).append(
                                float(np.abs(rS[-1] - rA[-1]).mean()))
                            acc.setdefault((scheme, nx, wtype, "v"), []).append(
                                float(np.abs(vS[-1] - vA[-1]).mean()))
        print(f"[res] done nx={nx}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (scheme, nx, wtype, field), vals in sorted(acc.items()):
        rows.append(dict(scheme=scheme, nx=nx, wave_type=wtype, field=field,
                         mean_L1=float(np.mean(vals)), n=len(vals)))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[res] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
