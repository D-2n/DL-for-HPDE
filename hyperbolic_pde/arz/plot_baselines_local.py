"""Local sanity plot: exact vs Godunov vs WENO5 for ARZ, on a few dataset samples.

Purpose: eyeball whether the two numerical baselines (the HLL/Godunov reference
solver and the component-wise WENO5 solver) reproduce the exact solution before
trusting them on the cluster. No model, no checkpoint -- baselines only.

Runs on CPU against any ARZ dataset .npz (e.g. an exact-Riemann set). For each
chosen sample it overlays, per channel (rho, v), at a few time slices:
  * exact   -- dataset GT (for exact-Riemann sets this is the exact solver at
               cell midpoints); for Riemann ICs we ALSO recompute the exact
               solution from the L/R states and check it matches the stored GT.
  * godunov -- reference_arz.solve_arz_reference (HLL + Strang relaxation)
  * weno5   -- eval_vs_numerical_arz.solve_arz_weno5

Usage (Windows, anaconda python):
  & C:\\Users\\dimit\\anaconda3\\python.exe -m hyperbolic_pde.arz.plot_baselines_local ^
      --data path\\to\\arz_riemann_mark2_prho.npz --samples 0,1,2 --out figs_baselines

If --samples is omitted, picks a few spread across the dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless; just write PNGs
import matplotlib.pyplot as plt

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz.eval_vs_numerical_arz import solve_arz_weno5
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.riemann_arz import solve_riemann_arz_xt


def _domain_edges(x: np.ndarray) -> tuple[float, float, float]:
    """x are CELL MIDPOINTS; recover true domain edges (matches eval script)."""
    x = np.asarray(x, dtype=np.float64)
    dx = float(x[1] - x[0])
    return float(x[0] - 0.5 * dx), float(x[-1] + 0.5 * dx), dx


def _is_riemann_ic(rho0: np.ndarray) -> bool:
    """A 2-segment Riemann IC has exactly one jump in rho0."""
    return int(np.count_nonzero(np.abs(np.diff(rho0)) > 1e-9)) == 1


def _riemann_lr(rho0, w0, x):
    """Return (rho_L, w_L, rho_R, w_R, x0) for a clean 2-segment Riemann IC, else None."""
    rho0 = np.asarray(rho0, np.float64)
    w0 = np.asarray(w0, np.float64)
    jumps = np.where(np.abs(np.diff(rho0)) > 1e-9)[0]
    if jumps.size != 1:
        return None
    j = int(jumps[0])
    x0 = 0.5 * (x[j] + x[j + 1])  # interface between the two cells
    return float(rho0[0]), float(w0[0]), float(rho0[-1]), float(w0[-1]), x0


def _exact_from_ic(rho0, w0, x, t):
    """Exact Riemann solution sampled on the dataset grid x (nt, nx), or None."""
    lr = _riemann_lr(rho0, w0, x)
    if lr is None:
        return None
    rho_L, w_L, rho_R, w_R, x0 = lr
    return solve_riemann_arz_xt(rho_L, w_L, rho_R, w_R,
                                np.asarray(x, np.float64),
                                np.asarray(t, np.float64), x0=x0)


def _exact_hires(rho0, w0, x, t, nx_hi=1000):
    """Smooth sub-grid exact solution: evaluate the self-similar profile on a
    hi-res x grid (like the ARZ notebook). Razor-sharp shocks/contacts because
    it is NOT reduced to the coarse cell grid. Returns (x_hi, rho, w, v) or None.
    """
    lr = _riemann_lr(rho0, w0, x)
    if lr is None:
        return None
    rho_L, w_L, rho_R, w_R, x0 = lr
    x_hi = np.linspace(float(x.min()), float(x.max()), nx_hi)
    rho, w, v = solve_riemann_arz_xt(rho_L, w_L, rho_R, w_R,
                                     x_hi, np.asarray(t, np.float64), x0=x0)
    return x_hi, rho, w, v


def _run_godunov(rho0, w0, x, t_max, nt, tau, boundary="ghost", refine=1):
    x_min, x_max, _ = _domain_edges(x)
    _, _, rho_h, w_h = Ref.solve_arz_reference(
        rho0, w0, x_min, x_max, t_max, nt_out=nt,
        tau=tau, cfl=0.4, boundary=boundary, refine=refine,
    )
    return rho_h, w_h


def _run_weno5(rho0, w0, x, t_max, nt, tau, boundary="ghost"):
    x_min, x_max, _ = _domain_edges(x)
    try:
        rho_h, w_h = solve_arz_weno5(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=tau, cfl=0.2, boundary=boundary,
        )
        if not (np.all(np.isfinite(rho_h)) and np.all(np.isfinite(w_h))):
            print("    [weno5] produced non-finite values (overflow) -> plotted as NaN")
        return rho_h, w_h
    except Exception as e:  # mirror eval's per-sample tolerance
        print(f"    [weno5] FAILED: {type(e).__name__}: {e}")
        return None, None


def _v_from(rho, w):
    return w - P.pressure(np.asarray(rho, np.float64))


def _solve_sample(bundle, idx, godunov_refine=1, nx_hi=1000):
    """Run both baselines + recover the exact for one sample.

    Returns a dict of (nt, nx) arrays per channel/method plus metadata.
    """
    x = np.asarray(bundle.x, np.float64)
    t = np.asarray(bundle.t, np.float64)
    nt = t.size
    tau = float(bundle.tau)
    baseline_tau = tau if np.isfinite(tau) else 1e6  # solvers can't take inf

    rho0 = bundle.rho0[idx].astype(np.float64)
    w0 = bundle.w0[idx].astype(np.float64)

    rho_gt = bundle.rho[idx].astype(np.float64)
    v_gt = bundle.v[idx].astype(np.float64)

    ex = _exact_from_ic(rho0, w0, x, t)
    if ex is not None:
        rho_ex, _, v_ex = ex
        print(f"    exact recompute vs stored GT: max|drho| = "
              f"{float(np.max(np.abs(rho_ex - rho_gt))):.2e}")
    else:
        rho_ex = v_ex = None

    # Smooth sub-grid exact (hi-res, notebook-style). Riemann ICs only.
    hi = _exact_hires(rho0, w0, x, t, nx_hi=nx_hi)
    if hi is not None:
        x_hi, rho_hi, _, v_hi = hi
    else:
        x_hi = rho_hi = v_hi = None

    print(f"    running godunov (tau={baseline_tau:g}, refine={godunov_refine}) ...")
    rho_g, w_g = _run_godunov(rho0, w0, x, float(t.max()), nt, baseline_tau,
                              refine=godunov_refine)
    v_g = _v_from(rho_g, w_g)

    print(f"    running weno5 (tau={baseline_tau:g}) ...")
    rho_w, w_w = _run_weno5(rho0, w0, x, float(t.max()), nt, baseline_tau)
    v_w = _v_from(rho_w, w_w) if rho_w is not None else None

    # Quantitative read: MAE vs the GT, per method per channel.
    for nm, ra, va in [("godunov", rho_g, v_g), ("weno5", rho_w, v_w)]:
        rmae = None if ra is None else float(np.mean(np.abs(ra - rho_gt)))
        vmae = None if va is None else float(np.mean(np.abs(va - v_gt)))
        rs = "n/a" if rmae is None else f"{rmae:.3e}"
        vs = "n/a" if vmae is None else f"{vmae:.3e}"
        print(f"    MAE {nm:8s}  rho={rs}  v={vs}")

    return dict(
        x=x, t=t, tau=tau, p_form=bundle.p_form,
        fam=str(bundle.ic_type[idx]), seg=int(bundle.num_segments[idx]),
        x_hi=x_hi,
        rho=dict(exact=rho_gt, godunov=rho_g, weno5=rho_w, exact_hi=rho_hi),
        v=dict(exact=v_gt, godunov=v_g, weno5=v_w, exact_hi=v_hi),
    )


def plot_sample_heat(d, idx, out_dir):
    """Space-time color plots (the usual view): pcolormesh over (x, t).

    Per channel block (rho, then v):
      solution row: exact (hi-res, smooth) | godunov | weno5 | exact (grid)
      error row:                 -          | godunov | weno5 |     -
    The hi-res exact is sampled sub-grid (NX_HI x-points) so shocks/contacts are
    razor-sharp -- the "true" answer, not reduced to the coarse cell grid.
    """
    x, t = d["x"], d["t"]
    x_hi = d.get("x_hi")
    cols = ["exact_hi", "godunov", "weno5", "exact"]
    titles = ["exact (hi-res)", "godunov", "weno5", "exact (grid)"]
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), squeeze=False)

    for blk, ch in enumerate(("rho", "v")):
        fields = d[ch]
        exact = fields["exact"]
        finite = [a for a in fields.values() if a is not None]
        vmin = min(float(np.nanmin(a)) for a in finite)
        vmax = max(float(np.nanmax(a)) for a in finite)
        errs = [np.abs(fields[m] - exact) for m in ("godunov", "weno5")
                if fields[m] is not None]
        err_vmax = max((float(np.nanmax(e)) for e in errs), default=1e-6) or 1e-6

        row_f, row_e = 2 * blk, 2 * blk + 1
        for c, m in enumerate(cols):
            arr = fields[m]
            xaxis = x_hi if (m == "exact_hi" and x_hi is not None) else x
            ax = axes[row_f][c]
            if arr is None:
                msg = "n/a\n(not Riemann)" if m == "exact_hi" else f"{m}\nFAILED"
                ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                im = ax.pcolormesh(xaxis, t, arr, shading="auto", cmap="jet",
                                   vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, label=ch if c == len(cols) - 1 else None)
            ax.set_title(f"{ch}  {titles[c]}")
            if c == 0:
                ax.set_ylabel("t")

            # Error row only meaningful for the two numerical baselines.
            axe = axes[row_e][c]
            if m in ("godunov", "weno5") and arr is not None:
                im = axe.pcolormesh(x, t, np.abs(arr - exact), shading="auto",
                                    cmap="magma", vmin=0, vmax=err_vmax)
                fig.colorbar(im, ax=axe, label="|err|" if c == len(cols) - 1 else None)
                axe.set_title(f"{ch} err  {titles[c]}")
                axe.set_xlabel("x")
                if c == 0:
                    axe.set_ylabel("t")
            else:
                axe.axis("off")

    fig.suptitle(f"sample {idx}  |  ic={d['fam']}  segs={d['seg']}  |  "
                 f"tau={d['tau']:g}  p_form={d['p_form']}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = out_dir / f"baselines_heat_sample{idx}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"    wrote {out}")


def plot_sample_lines(d, idx, t_frac, out_dir):
    """1D line slices at a few times. Black = smooth hi-res exact (sub-grid);
    faint dots = grid-sampled exact; baselines on the cell grid."""
    x, t = d["x"], d["t"]
    x_hi = d.get("x_hi")
    nt = t.size
    ks = sorted({int(round(f * (nt - 1))) for f in t_frac})
    fig, axes = plt.subplots(2, len(ks), figsize=(4.2 * len(ks), 6.4), squeeze=False)
    for c, k in enumerate(ks):
        for r, ch in enumerate(("rho", "v")):
            f = d[ch]
            ax = axes[r][c]
            if f.get("exact_hi") is not None and x_hi is not None:
                ax.plot(x_hi, f["exact_hi"][k], "k-", lw=2.0, label="exact (hi-res)")
                ax.plot(x, f["exact"][k], "k.", ms=3, alpha=0.4, label="exact (grid)")
            else:
                ax.plot(x, f["exact"][k], "k-", lw=2.0, label="exact (grid)")
            ax.plot(x, f["godunov"][k], color="tab:green", ls="--", lw=1.4,
                    marker="o", ms=2.5, label="godunov")
            if f["weno5"] is not None:
                ax.plot(x, f["weno5"][k], color="tab:orange", ls="-.", lw=1.4,
                        marker="s", ms=2.5, label="weno5")
            ax.set_title(f"{ch}  t={t[k]:.3f}")
            ax.grid(alpha=0.3)
            if r == 1:
                ax.set_xlabel("x")
            if c == 0:
                ax.set_ylabel(ch)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"sample {idx}  |  ic={d['fam']}  segs={d['seg']}  |  "
                 f"tau={d['tau']:g}  p_form={d['p_form']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = out_dir / f"baselines_lines_sample{idx}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"    wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="ARZ dataset .npz")
    ap.add_argument("--samples", type=str, default=None,
                    help="comma-separated sample indices; default spreads a few")
    ap.add_argument("--n", type=int, default=4,
                    help="how many samples if --samples not given (default 4)")
    ap.add_argument("--t-frac", type=str, default="0.0,0.33,0.66,1.0",
                    help="time fractions in [0,1] for the line-slice view")
    ap.add_argument("--mode", choices=["heat", "lines", "both"], default="heat",
                    help="heat = space-time color plots (default); lines = 1D slices; both")
    ap.add_argument("--godunov-refine", type=int, default=1,
                    help="spatial refinement for the Godunov/HLL reference solver "
                         "(1 = on the cell grid, what the cluster eval uses; "
                         ">1 = solve fine then block-mean, sharper). Default 1.")
    ap.add_argument("--nx-hi", type=int, default=1000,
                    help="x-resolution for the smooth sub-grid exact (default 1000)")
    ap.add_argument("--out", type=Path, default=Path("figs_baselines"))
    args = ap.parse_args()

    bundle = load_arz_dataset(args.data)
    # CRITICAL: set the pressure closure from the dataset before any solver call,
    # or v / eigenvalues / fluxes use the wrong p(rho).
    P.set_pressure_form(bundle.p_form)

    N = bundle.rho.shape[0]
    if args.samples:
        idxs = [int(s) for s in args.samples.split(",") if s.strip() != ""]
    else:
        idxs = list(np.linspace(0, N - 1, min(args.n, N)).astype(int))
    t_frac = [float(s) for s in args.t_frac.split(",")]

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"dataset: {args.data}  N={N}  nt={bundle.t.size}  nx={bundle.x.size}  "
          f"tau={bundle.tau:g}  p_form={bundle.p_form}")
    print(f"plotting samples: {idxs}")
    for idx in idxs:
        print(f"sample {idx}:")
        d = _solve_sample(bundle, idx, godunov_refine=args.godunov_refine,
                          nx_hi=args.nx_hi)
        if args.mode in ("heat", "both"):
            plot_sample_heat(d, idx, args.out)
        if args.mode in ("lines", "both"):
            plot_sample_lines(d, idx, t_frac, args.out)
    print(f"\nDone. Figures in {args.out.resolve()}")


if __name__ == "__main__":
    main()
