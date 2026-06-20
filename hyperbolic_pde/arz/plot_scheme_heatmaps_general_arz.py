"""Space-time (x-t) heatmaps on GENERAL multi-segment ARZ data.

Beyond the single-interface Riemann family: multi-segment piecewise-constant ICs
(several discontinuities that interact over time -- fronts collide, shocks merge,
contacts cross). There is NO closed-form analytic GT for these, but WFT IS exact
for piecewise-constant data (exact wave speeds + exact Riemann collision
resolution), so **WFT is the reference** here and the FV schemes (muscl / weno5 /
hll) are measured against it.

Each sampled IC gives one row of x-t heatmaps: columns = [wft (=GT), muscl,
weno5, hll]; one figure per field (rho, v, w). An --error variant plots
|FV - WFT|. ICs are drawn like the datagen sampler (stratified adjacent jumps in
rho and v), with a configurable number of segments and a fixed seed for
reproducibility.

Usage:
    python hyperbolic_pde/arz/plot_scheme_heatmaps_general_arz.py \
        --pressure-form rho --segments 4 --n-samples 4 --seed 0 \
        --out-dir figures/scheme_sweep/heatmaps_general
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
from hyperbolic_pde.arz.wft_arz import solve_arz_wft_xt


CMAPS = {"rho": "viridis", "v": "cividis", "w": "magma"}
LABELS = {"rho": r"$\rho$", "v": r"$v$", "w": r"$w$"}
_DEFAULT_CFL = {"muscl": 0.4, "weno5": 0.2, "hll": 0.4, "hllc": 0.4}

# Jump-strength band for the stratified random walk (mirrors fvm._stratified_pair
# intent: visible, well-spread adjacent jumps rather than near-zero ones).
_JUMP_MIN, _JUMP_MAX = 0.15, 0.6


def _walk(n, lo, hi, rng):
    """Random walk of n segment values in [lo, hi] with controlled adjacent
    jumps (reflected at the bounds so jump magnitude is preserved)."""
    vals = [rng.uniform(lo, hi)]
    for _ in range(n - 1):
        du = rng.uniform(_JUMP_MIN, min(_JUMP_MAX, hi - lo))
        nxt = vals[-1] + (du if rng.random() < 0.5 else -du)
        # reflect into [lo, hi]
        if nxt < lo:
            nxt = lo + (lo - nxt)
        if nxt > hi:
            nxt = hi - (nxt - hi)
        nxt = float(np.clip(nxt, lo, hi))
        vals.append(nxt)
    return vals


def _sample_ic(num_segments, x_min, x_max, rng,
               rho_lo=0.15, rho_hi=0.85, v_lo=0.05, v_hi=0.95):
    """Return (states, boundaries) for a multi-segment piecewise-constant IC.

    states : list[(rho, v)] of length num_segments
    boundaries : list[float] of length num_segments-1, strictly increasing,
                 inside the open domain.
    """
    rho_vals = _walk(num_segments, rho_lo, rho_hi, rng)
    v_vals = _walk(num_segments, v_lo, v_hi, rng)
    states = list(zip(rho_vals, v_vals))
    if num_segments == 1:
        return states, []
    # interior cut points, sorted, kept off the very edges
    span = x_max - x_min
    cuts = np.sort(rng.uniform(x_min + 0.08 * span, x_max - 0.08 * span,
                               size=num_segments - 1))
    # nudge apart if too close
    cuts = np.maximum.accumulate(cuts + 1e-9 * np.arange(num_segments - 1))
    return states, [float(c) for c in cuts]


def _ic_to_cells(states, boundaries, xm):
    """Project (states, boundaries) onto cell midpoints -> (rho0, v0)."""
    rho0 = np.empty_like(xm); v0 = np.empty_like(xm)
    seg_idx = np.searchsorted(boundaries, xm, side="right")
    for k in range(xm.size):
        r, v = states[int(seg_idx[k])]
        rho0[k] = r; v0[k] = v
    return rho0, v0


def _run_fv(scheme, rho0, w0, x_min, x_max, t_max, nt, cfl):
    """Full-field (rho, v, w) for an FV scheme, or None on failure."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if scheme == "muscl":
                rH, wH = R.solve_arz_muscl(rho0, w0, x_min, x_max, t_max, nt,
                                           tau=1e9, cfl=cfl, boundary="ghost")
            elif scheme == "weno5":
                rH, wH = R.solve_arz_weno5(rho0, w0, x_min, x_max, t_max, nt,
                                           tau=1e9, cfl=cfl, boundary="ghost")
            elif scheme in ("hll", "hllc"):
                _, _, rH, wH = R.solve_arz_reference(
                    rho0, w0, x_min, x_max, t_max, nt_out=nt, tau=1e9,
                    cfl=cfl, boundary="ghost", refine=1, flux_scheme=scheme)
            else:
                raise ValueError(scheme)
            if not (np.isfinite(rH).all() and np.isfinite(wH).all()):
                raise FloatingPointError("non-finite")
        vH = wH - P.pressure(rH)
        return rH, vH, wH
    except (FloatingPointError, RuntimeError, ZeroDivisionError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pressure-form", default="rho", choices=("rho", "rho+rho2"))
    ap.add_argument("--schemes", default="muscl,weno5,hll",
                    help="FV schemes to compare against WFT (the GT column)")
    ap.add_argument("--fields", default="rho,v,w")
    ap.add_argument("--segments", type=int, default=4,
                    help="number of piecewise-constant segments per IC")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--t-max", type=float, default=0.4)
    ap.add_argument("--rare-delta", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--error", action="store_true",
                    help="plot |FV - WFT| instead of the raw field")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("figures/scheme_sweep/heatmaps_general"))
    args = ap.parse_args()

    P.set_pressure_form(args.pressure_form)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    x_min, x_max = 0.0, 1.0
    dx = (x_max - x_min) / args.nx
    xm = x_min + (np.arange(args.nx) + 0.5) * dx
    t = np.linspace(0.0, args.t_max, args.nt)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Sample ICs and solve once per (sample, scheme). WFT is the reference.
    samples = []
    ref = {}     # sample_idx -> dict(field -> (nt,nx))  [WFT]
    fv = {}      # (sample_idx, scheme) -> dict or None
    for si in range(args.n_samples):
        states, boundaries = _sample_ic(args.segments, x_min, x_max, rng)
        samples.append((states, boundaries))
        rW, wW, vW = solve_arz_wft_xt(states, boundaries, xm, t,
                                      rare_delta=args.rare_delta)
        ref[si] = {"rho": rW, "v": vW, "w": wW}
        rho0, v0 = _ic_to_cells(states, boundaries, xm)
        w0 = v0 + P.pressure(rho0)
        for scheme in schemes:
            out = _run_fv(scheme, rho0, w0, x_min, x_max, args.t_max, args.nt,
                          _DEFAULT_CFL.get(scheme, 0.4))
            fv[(si, scheme)] = (None if out is None
                                else {"rho": out[0], "v": out[1], "w": out[2]})

    cols = (["wft (GT)"] if not args.error else []) + schemes
    extent = [x_min, x_max, 0.0, args.t_max]

    for fld in fields:
        nrow, ncol = args.n_samples, len(cols)
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.7 * nrow),
                                 squeeze=False)
        for si in range(args.n_samples):
            gfield = ref[si][fld]
            if args.error:
                errs = {s: (None if fv[(si, s)] is None
                            else np.abs(fv[(si, s)][fld] - gfield))
                        for s in schemes}
                finite = [e for e in errs.values() if e is not None]
                vmax = max((float(e.max()) for e in finite), default=1.0) or 1.0
                for ci, scheme in enumerate(schemes):
                    ax = axes[si][ci]
                    e = errs[scheme]
                    if e is None:
                        ax.text(0.5, 0.5, "CRASH", ha="center", va="center",
                                transform=ax.transAxes, color="red")
                        ax.set_xticks([]); ax.set_yticks([])
                    else:
                        im = ax.imshow(e, origin="lower", aspect="auto",
                                       extent=extent, cmap="magma",
                                       vmin=0.0, vmax=vmax)
                    if si == 0:
                        ax.set_title(f"|{scheme} - WFT|", fontsize=10)
            else:
                vmin = float(gfield.min()); vmax = float(gfield.max())
                for scheme in schemes:
                    cur = fv[(si, scheme)]
                    if cur is not None:
                        vmin = min(vmin, float(cur[fld].min()))
                        vmax = max(vmax, float(cur[fld].max()))
                if vmax - vmin < 1e-9:
                    vmax = vmin + 1e-9
                for ci, col in enumerate(cols):
                    ax = axes[si][ci]
                    field = gfield if col == "wft (GT)" else (
                        None if fv[(si, col)] is None else fv[(si, col)][fld])
                    if field is None:
                        ax.text(0.5, 0.5, "CRASH", ha="center", va="center",
                                transform=ax.transAxes, color="red")
                        ax.set_xticks([]); ax.set_yticks([])
                    else:
                        im = ax.imshow(field, origin="lower", aspect="auto",
                                       extent=extent, cmap=CMAPS[fld],
                                       vmin=vmin, vmax=vmax)
                    if si == 0:
                        ax.set_title(col, fontsize=10)
            axes[si][0].set_ylabel(f"sample {si}\n(seg={args.segments})\n\nt",
                                   fontsize=8)
            for ci in range(ncol):
                axes[si][ci].set_xlabel("x", fontsize=8)
            try:
                fig.colorbar(im, ax=axes[si].tolist(), fraction=0.025, pad=0.01)
            except Exception:
                pass

        mode = "error vs WFT" if args.error else "field (WFT=GT)"
        fig.suptitle(
            f"ARZ general multi-segment x-t heatmaps — {LABELS[fld]} ({mode})  "
            f"p={args.pressure_form}, segments={args.segments}, "
            f"nx={args.nx}, nt={args.nt}, t_max={args.t_max}", fontsize=12)
        out = args.out_dir / (
            f"general_heatmap_{fld}_{'error' if args.error else 'field'}_"
            f"seg{args.segments}_{args.pressure_form.replace('+','')}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[heatmap] wrote {out}")


if __name__ == "__main__":
    main()
