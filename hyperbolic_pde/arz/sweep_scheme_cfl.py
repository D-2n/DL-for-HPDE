"""Numerical-scheme x CFL sweep for ARZ ground-truth generation.

Compares finite-volume schemes against the EXACT analytic Riemann solution
(`solve_riemann_arz_xt`, the dataset's gold single-jump GT) across the datagen
stratification grid:

    (1-wave type) x (1-wave strength bin) x (contact strength bin)

Schemes:
  * hll      : 1st-order FV, HLL flux            (reference_arz, flux_scheme='hll')
  * hllc     : 1st-order FV, HLLC flux (sharp 2-contact)  (flux_scheme='hllc')
  * muscl    : 2nd-order MUSCL-Hancock + HLLC     (reference_arz.solve_arz_muscl)
  * godunov  : 1st-order FV, exact-Riemann flux  (reference_arz, flux_scheme='godunov')
  * weno5    : 5th-order WENO + SSP-RK3 (global LF) (eval_vs_numerical_arz.solve_arz_weno5)

For each (scheme, cfl) we report mean L1 error on rho and v vs the analytic GT,
split by wave-type (shock / rarefaction), averaged over `--seeds` IC draws per
bin, plus wall-clock cost. WENO5 instabilities (overflow / NaN) are caught and
recorded as failures rather than crashing the sweep.

Output: a tidy CSV (one row per scheme x cfl x wave-type x field) and a console
summary table.

Usage (headless):
    .../python.exe hyperbolic_pde/arz/sweep_scheme_cfl.py \
        --pressure-form rho --out figures/scheme_sweep/sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz.riemann_arz import solve_riemann_arz_xt
from hyperbolic_pde.arz.datagen_arz import _sample_riemann_stratified_cell
from hyperbolic_pde.arz.eval_vs_numerical_arz import solve_arz_weno5
from hyperbolic_pde.arz.reference_arz import solve_arz_muscl


def _strength_bins(rho_min, rho_max, n_bins):
    span = rho_max - rho_min
    edges = np.linspace(0.0, span, n_bins + 1)
    edges[0] = min(0.02 * span, 0.5 * edges[1])
    return [(float(edges[b]), float(edges[b + 1])) for b in range(n_bins)]


def _analytic_gt(rL, vL, rR, vR, x0, xm, t):
    wL = float(vL + P.pressure(np.array(float(rL))))
    wR = float(vR + P.pressure(np.array(float(rR))))
    rA, wA, vA = solve_riemann_arz_xt(rL, wL, rR, wR, xm, t, x0=x0)
    return rA, wA, vA


def _run_scheme(scheme, rho0, w0, x_min, x_max, t_max, nt, cfl):
    """Return (rho_hist, v_hist) for the requested scheme, or raise on failure."""
    if scheme in ("hll", "godunov", "hllc"):
        _, _, rH, wH = Ref.solve_arz_reference(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=1e9, cfl=cfl, boundary="ghost", refine=1, flux_scheme=scheme,
        )
        vH = wH - P.pressure(rH)
        return rH, vH
    if scheme == "weno5":
        rW, wW = solve_arz_weno5(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=1e9, cfl=cfl, boundary="ghost",
        )
        vW = wW - P.pressure(rW)
        return rW, vW
    if scheme == "muscl":
        rM, wM = solve_arz_muscl(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=1e9, cfl=cfl, boundary="ghost",
        )
        vM = wM - P.pressure(rM)
        return rM, vM
    raise ValueError(scheme)


def sweep(args):
    P.set_pressure_form(args.pressure_form)
    bins = _strength_bins(args.rho_min, args.rho_max, args.n_strength_bins)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    cfls = [float(c) for c in args.cfls.split(",") if c.strip()]

    dx = (args.x_max - args.x_min) / args.nx
    xm = (args.x_min + (np.arange(args.nx) + 0.5) * dx).astype(np.float64)
    t = np.linspace(0.0, args.t_max, args.nt)

    # Accumulators: key=(scheme,cfl,wavetype,field) -> list of per-sample L1 errs.
    err = {}
    fails = {}        # (scheme,cfl) -> count of failed solves
    cost = {}         # (scheme,cfl) -> total solve seconds
    n_solves = {}     # (scheme,cfl) -> count of attempted solves

    print(f"[sweep] schemes={schemes} cfls={cfls} grid=2x{args.n_strength_bins}x"
          f"{args.n_strength_bins} seeds={args.seeds} nx={args.nx} nt={args.nt} "
          f"t_max={args.t_max} p={args.pressure_form}")

    for is_shock in (True, False):
        wtype = "shock" if is_shock else "rare"
        for jb in bins:
            for cb in bins:
                # Pre-sample the ICs for this bin (shared across all scheme/cfl
                # so every scheme sees the SAME problems).
                rng = np.random.default_rng(args.seed)
                # advance rng deterministically per (wtype,jb,cb): re-seed by hashing
                # the bin coords into the stream via repeated draws is fragile;
                # instead derive a per-bin seed.
                bin_seed = (args.seed
                            + (1 if is_shock else 0) * 100003
                            + int(jb[0] * 1e4) * 31
                            + int(cb[0] * 1e4))
                rng = np.random.default_rng(bin_seed)
                ics = []
                for _ in range(args.seeds):
                    rL, vL, rR, vR, x0 = _sample_riemann_stratified_cell(
                        xm, rng, args.rho_min, args.rho_max, args.v_min, args.v_max,
                        one_wave_is_shock=is_shock,
                        one_wave_jump_range=jb, contact_jump_range=cb,
                    )
                    ics.append((rL, vL, rR, vR, x0))

                # Analytic GT per IC (scheme-independent).
                gts = []
                for (rL, vL, rR, vR, x0) in ics:
                    rA, wA, vA = _analytic_gt(rL, vL, rR, vR, x0, xm, t)
                    gts.append((rA, vA))

                for scheme in schemes:
                    for cfl in cfls:
                        key_sc = (scheme, cfl)
                        cost.setdefault(key_sc, 0.0)
                        n_solves.setdefault(key_sc, 0)
                        fails.setdefault(key_sc, 0)
                        for (rL, vL, rR, vR, x0), (rA, vA) in zip(ics, gts):
                            rho0 = np.where(xm <= x0, rL, rR).astype(float)
                            v0 = np.where(xm <= x0, vL, vR).astype(float)
                            w0 = v0 + P.pressure(rho0)
                            n_solves[key_sc] += 1
                            t0 = time.perf_counter()
                            try:
                                with np.errstate(over="raise", invalid="raise", divide="raise"):
                                    rS, vS = _run_scheme(
                                        scheme, rho0, w0, args.x_min, args.x_max,
                                        args.t_max, args.nt, cfl,
                                    )
                                if not (np.isfinite(rS).all() and np.isfinite(vS).all()):
                                    raise FloatingPointError("non-finite output")
                            except (FloatingPointError, RuntimeError, FloatingPointError) as e:
                                fails[key_sc] += 1
                                cost[key_sc] += time.perf_counter() - t0
                                continue
                            cost[key_sc] += time.perf_counter() - t0
                            # final-time L1 error (mean abs over space).
                            e_rho = float(np.abs(rS[-1] - rA[-1]).mean())
                            e_v = float(np.abs(vS[-1] - vA[-1]).mean())
                            err.setdefault((scheme, cfl, wtype, "rho"), []).append(e_rho)
                            err.setdefault((scheme, cfl, wtype, "v"), []).append(e_v)
        print(f"[sweep]   done wave-type={wtype}", flush=True)

    # Write CSV.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (scheme, cfl, wtype, field), vals in sorted(err.items()):
        arr = np.array(vals)
        rows.append(dict(
            scheme=scheme, cfl=cfl, wave_type=wtype, field=field,
            mean_L1=float(arr.mean()), std_L1=float(arr.std()),
            median_L1=float(np.median(arr)), n=len(arr),
            fails=fails[(scheme, cfl)], n_solves=n_solves[(scheme, cfl)],
            sec_per_solve=cost[(scheme, cfl)] / max(n_solves[(scheme, cfl)], 1),
        ))
    with out.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"[sweep] wrote {out}  ({len(rows)} rows)")

    # Console summary: pivot by (scheme,cfl) -> overall mean L1 across both fields/wave-types.
    print("\n=== Summary: mean L1 error (lower=better), avg over wave-types ===")
    print(f"{'scheme':9s} {'cfl':>5s} | {'rho_shock':>10s} {'rho_rare':>9s} "
          f"{'v_shock':>9s} {'v_rare':>9s} | {'overall':>8s} {'fails':>6s} {'s/solve':>8s}")
    summary = []
    for scheme in schemes:
        for cfl in cfls:
            def g(wt, fld):
                vals = err.get((scheme, cfl, wt, fld), [])
                return float(np.mean(vals)) if vals else float("nan")
            rs, rr = g("shock", "rho"), g("rare", "rho")
            vs, vr = g("shock", "v"), g("rare", "v")
            overall = np.nanmean([rs, rr, vs, vr])
            spc = cost[(scheme, cfl)] / max(n_solves[(scheme, cfl)], 1)
            fl = fails[(scheme, cfl)]
            summary.append((scheme, cfl, rs, rr, vs, vr, overall, fl, spc))
            print(f"{scheme:9s} {cfl:5.2f} | {rs:10.3e} {rr:9.3e} {vs:9.3e} {vr:9.3e} "
                  f"| {overall:8.3e} {fl:6d} {spc:8.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="ARZ scheme x CFL sweep vs analytic GT.")
    ap.add_argument("--schemes", type=str, default="hll,hllc,muscl,weno5,godunov")
    ap.add_argument("--cfls", type=str, default="0.1,0.2,0.4,0.6,0.9")
    ap.add_argument("--n-strength-bins", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3, help="IC draws per bin")
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
    ap.add_argument("--out", type=Path, default=Path("figures/scheme_sweep/sweep.csv"))
    args = ap.parse_args()
    sweep(args)


if __name__ == "__main__":
    main()
