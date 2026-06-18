"""Overshoot / oscillation probe for ARZ numerical schemes.

Mean-L1 error rewards smooth accuracy but HIDES the failure mode that matters
most for a high-order scheme used as DATAGEN GROUND TRUTH: Gibbs-type over/
undershoots near discontinuities. A WENO5 GT that rings near a contact teaches
the network non-physical local structure even if its L1 is low.

This probe measures, per scheme x cfl, against the exact analytic GT:
  * max_overshoot : max amount the scheme exceeds the GT's [min,max] envelope
                    (per field) -- i.e. new extrema the true solution never has.
                    For a monotone/TVD scheme this is ~0. WENO5 can be >0.
  * tv_excess     : total-variation(scheme) / total-variation(GT) - 1 at final
                    time. >0 means spurious wiggles added; ~0 is clean.
  * L1            : for reference.

Across the stratification grid, split by wave-type, both rho and v.

Usage:
    .../python.exe hyperbolic_pde/arz/probe_overshoot.py \
        --schemes hll,godunov,weno5 --cfls 0.1,0.4,0.9 --pressure-form rho \
        --out figures/scheme_sweep/overshoot.csv
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


def _run_scheme(scheme, rho0, w0, x_min, x_max, t_max, nt, cfl):
    if scheme in ("hll", "godunov"):
        _, _, rH, wH = Ref.solve_arz_reference(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=1e9, cfl=cfl, boundary="ghost", refine=1, flux_scheme=scheme)
        return rH, wH - P.pressure(rH)
    if scheme == "weno5":
        rW, wW = solve_arz_weno5(rho0, w0, x_min, x_max, t_max, nt_out=nt,
                                 tau=1e9, cfl=cfl, boundary="ghost")
        return rW, wW - P.pressure(rW)
    raise ValueError(scheme)


def _tv(a):
    return float(np.abs(np.diff(a)).sum())


def probe(args):
    P.set_pressure_form(args.pressure_form)
    bins = _strength_bins(args.rho_min, args.rho_max, args.n_strength_bins)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    cfls = [float(c) for c in args.cfls.split(",") if c.strip()]
    dx = (args.x_max - args.x_min) / args.nx
    xm = (args.x_min + (np.arange(args.nx) + 0.5) * dx).astype(np.float64)
    t = np.linspace(0.0, args.t_max, args.nt)

    acc = {}  # (scheme,cfl,wtype,field,metric) -> list
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
                        for cfl in cfls:
                            try:
                                with np.errstate(over="raise", invalid="raise", divide="raise"):
                                    rS, vS = _run_scheme(scheme, rho0, w0, args.x_min,
                                                         args.x_max, args.t_max, args.nt, cfl)
                                if not (np.isfinite(rS).all() and np.isfinite(vS).all()):
                                    raise FloatingPointError
                            except (FloatingPointError, RuntimeError):
                                continue
                            for fld, S, A in (("rho", rS[-1], rA[-1]), ("v", vS[-1], vA[-1])):
                                gmin, gmax = A.min(), A.max()
                                over = max(0.0, float(S.max() - gmax))
                                under = max(0.0, float(gmin - S.min()))
                                overshoot = max(over, under)
                                tv_excess = _tv(S) / max(_tv(A), 1e-12) - 1.0
                                l1 = float(np.abs(S - A).mean())
                                acc.setdefault((scheme, cfl, wtype, fld, "overshoot"), []).append(overshoot)
                                acc.setdefault((scheme, cfl, wtype, fld, "tv_excess"), []).append(tv_excess)
                                acc.setdefault((scheme, cfl, wtype, fld, "L1"), []).append(l1)
        print(f"[probe]   done wave-type={wtype}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    keyset = sorted({(s, c, wt, fl) for (s, c, wt, fl, _m) in acc})
    for (s, c, wt, fl) in keyset:
        row = dict(scheme=s, cfl=c, wave_type=wt, field=fl)
        for m in ("overshoot", "tv_excess", "L1"):
            vals = np.array(acc.get((s, c, wt, fl, m), [np.nan]))
            row[f"mean_{m}"] = float(np.nanmean(vals))
            row[f"max_{m}"] = float(np.nanmax(vals))
        rows.append(row)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[probe] wrote {out} ({len(rows)} rows)")

    print("\n=== Overshoot / TV-excess (lower=cleaner GT), avg over wave-types & fields ===")
    print(f"{'scheme':9s} {'cfl':>5s} | {'mean_oversh':>11s} {'max_oversh':>10s} "
          f"{'mean_tvX':>9s} {'max_tvX':>9s} | {'mean_L1':>9s}")
    for s in schemes:
        for c in cfls:
            def g(metric, agg):
                vals = [v for (ss, cc, wt, fl, m), lst in acc.items()
                        if ss == s and cc == c and m == metric for v in lst]
                if not vals:
                    return float("nan")
                return float(np.nanmean(vals)) if agg == "mean" else float(np.nanmax(vals))
            print(f"{s:9s} {c:5.2f} | {g('overshoot','mean'):11.3e} {g('overshoot','max'):10.3e} "
                  f"{g('tv_excess','mean'):9.3e} {g('tv_excess','max'):9.3e} | {g('L1','mean'):9.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", type=str, default="hll,godunov,weno5")
    ap.add_argument("--cfls", type=str, default="0.1,0.4,0.9")
    ap.add_argument("--n-strength-bins", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nx", type=int, default=128)
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
    ap.add_argument("--out", type=Path, default=Path("figures/scheme_sweep/overshoot.csv"))
    args = ap.parse_args()
    probe(args)


if __name__ == "__main__":
    main()
