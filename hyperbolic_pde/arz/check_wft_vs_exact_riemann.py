"""Verify WFT dataset values against the EXACT analytic ARZ Riemann solution.

For every `riemann_stratified` sample (single interface), this:
  1. recovers (rho_L, w_L, rho_R, w_R, x0) from the stored IC (same detection as
     datagen: the largest adjacent jump in rho0/w0),
  2. computes the EXACT self-similar solution on the same (x, t) grid via
     riemann_arz.solve_riemann_arz_xt,
  3. compares it cell-by-cell against the dataset's stored values,
  4. classifies the 1-wave (shock / rarefaction / degenerate-contact) and,
     SPECIFICALLY for SHOCK cases, checks that every cell is L, R, or the exact
     star state rho_* -- i.e. that a shock has NO spurious intermediate value
     beyond the single grid-straddle cell.

The question this answers: the "little smear" between two states in the WFT
output -- is it a real intermediate (star state rho_* / rarefaction fan, which
IS exact physics), or a numerical artifact?

  * If max |WFT - exact| ~ 1e-6, the WFT data IS the exact solution -> the
    intermediate cells are real rho_* / fan states, NOT smear.
  * If a SHOCK sample has a cell whose value is none of {L, R, rho_*} beyond one
    straddle cell, that IS a spurious intermediate -> a real artifact.

Usage (CLEPS):
    python -m hyperbolic_pde.arz.check_wft_vs_exact_riemann \
        --data /home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho_clean.npz \
        --pressure-form rho \
        --n 40 \
        --out /home/dzdrale/scratch/results/wft_vs_exact/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import riemann_arz as Rie


def _recover_riemann_ic(rho0, w0, x_min, x_max):
    """Replicate datagen's single-interface detection (datagen_arz.py)."""
    nx = rho0.size
    dx = (x_max - x_min) / nx
    x_mid = np.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, nx, dtype=np.float64)
    d = np.abs(np.diff(rho0)) + np.abs(np.diff(w0))
    j = int(np.argmax(d))
    x0 = 0.5 * (x_mid[j] + x_mid[j + 1])
    return (float(rho0[j]), float(w0[j]),
            float(rho0[j + 1]), float(w0[j + 1]), x0, x_mid)


def _classify_1wave(rho_L, w_L, rho_R, w_R):
    """Return (wave_type, rho_star, v_star, shock_speed_or_None, contact_speed)."""
    v_L = w_L - P.pressure(rho_L)
    v_R = w_R - P.pressure(rho_R)
    rho_star = float(Rie._rho_star_from_w_minus_v(w_L - v_R))
    v_star = v_R
    lam1_L = v_L - rho_L * P.dpressure(rho_L)
    lam1_star = v_star - rho_star * P.dpressure(rho_star)
    contact_speed = v_star
    if lam1_L > lam1_star + 1e-12:
        denom = rho_star - rho_L
        s = lam1_L if abs(denom) < 1e-14 else (rho_star * v_star - rho_L * v_L) / denom
        return "shock", rho_star, v_star, float(s), float(contact_speed)
    elif lam1_L < lam1_star - 1e-12:
        return "rarefaction", rho_star, v_star, None, float(contact_speed)
    else:
        return "contact-only", rho_star, v_star, None, float(contact_speed)


def check(data_path: Path, pressure_form: str, n_samples: int, out_dir: Path,
          tol: float, plot_worst: int) -> None:
    P.set_pressure_form(pressure_form)
    print(f"[check] pressure_form = {P.get_pressure_form()}")

    d = np.load(data_path, allow_pickle=True)
    stored_pform = str(d["p_form"]) if "p_form" in d else "unknown"
    print(f"[check] dataset p_form = {stored_pform}")
    if stored_pform != pressure_form:
        print(f"[check] WARNING: dataset p_form '{stored_pform}' != requested '{pressure_form}'")

    N = d["rho"].shape[0]
    nt, nx = d["rho"].shape[1], d["rho"].shape[2]
    x = d["x"] if "x" in d else np.linspace(-1, 1, nx)
    t = d["t"] if "t" in d else np.linspace(0, 1, nt)
    x_min, x_max = float(x[0] - 0.5 * (x[1] - x[0])), float(x[-1] + 0.5 * (x[1] - x[0]))
    ic_types = d["ic_type"] if "ic_type" in d else np.array(["riemann_stratified"] * N)

    riemann_idx = np.where(ic_types == "riemann_stratified")[0]
    if riemann_idx.size == 0:
        print("[check] no riemann_stratified samples in this dataset; nothing to check.")
        return
    rng = np.random.default_rng(0)
    chosen = rng.choice(riemann_idx, size=min(n_samples, riemann_idx.size), replace=False)
    chosen = sorted(chosen)
    out_dir.mkdir(parents=True, exist_ok=True)

    dx = (x_max - x_min) / nx
    results = []   # (idx, wave_type, max_abs_err, n_bad_shock_cells)

    for idx in chosen:
        rho_ds = d["rho"][idx].astype(np.float64)
        w_ds = d["w"][idx].astype(np.float64)
        rho0 = d["rho0"][idx].astype(np.float64) if "rho0" in d else rho_ds[0]
        w0 = d["w0"][idx].astype(np.float64) if "w0" in d else w_ds[0]

        rho_L, w_L, rho_R, w_R, x0, x_mid = _recover_riemann_ic(rho0, w0, x_min, x_max)

        # exact analytic solution on the same grid
        rho_ex, w_ex, v_ex = Rie.solve_riemann_arz_xt(rho_L, w_L, rho_R, w_R, x_mid, t, x0=x0)

        err_rho = np.abs(rho_ds - rho_ex)
        err_w = np.abs(w_ds - w_ex)
        max_err = float(max(err_rho.max(), err_w.max()))

        wave_type, rho_star, v_star, s_shock, s_contact = _classify_1wave(rho_L, w_L, rho_R, w_R)

        # Shock-specific test: at the final time, every cell should be ~L, ~R, or
        # ~rho_star (the contact carries L*=star -> R). A cell that is none of
        # these (beyond ONE straddle cell per discontinuity) is a spurious smear.
        n_bad_shock = 0
        if wave_type == "shock":
            rT = rho_ds[-1]
            known = np.array([rho_L, rho_star, rho_R])
            dist = np.min(np.abs(rT[:, None] - known[None, :]), axis=1)
            # cells not within tol of any known plateau value
            bad = dist > max(tol, 1e-3 * (abs(rho_R - rho_L) + 1e-9))
            n_bad_shock = int(bad.sum())

        results.append((idx, wave_type, max_err, n_bad_shock, rho_L, rho_star, rho_R))

    # -------------------- summary --------------------
    results.sort(key=lambda r: -r[2])
    print(f"\n[check] {len(results)} riemann_stratified samples checked against EXACT solver")
    print(f"  {'idx':>5s} {'wave':>12s} {'max|WFT-exact|':>15s} {'shock_intermed_cells':>20s}")
    for idx, wt, me, nb, *_ in results[:min(len(results), 25)]:
        flag = "  <-- SMEAR?" if (me > tol or nb > 2) else ""
        print(f"  {idx:>5d} {wt:>12s} {me:>15.3e} {nb:>20d}{flag}")

    all_err = np.array([r[2] for r in results])
    shock_bad = np.array([r[3] for r in results if r[1] == "shock"])
    print(f"\n[check] max|WFT-exact| over all checked: {all_err.max():.3e}")
    print(f"[check] mean|WFT-exact| over all checked: {all_err.mean():.3e}")
    if all_err.max() <= tol:
        print(f"[check] => WFT data MATCHES the exact analytic solution to <{tol:.0e}.")
        print(f"        The intermediate values are real rho_* / rarefaction states, NOT smear.")
    else:
        nbad = int((all_err > tol).sum())
        print(f"[check] => {nbad} sample(s) exceed tol={tol:.0e}: genuine discrepancy from exact. INVESTIGATE.")
    if shock_bad.size:
        print(f"[check] shock samples: max intermediate (non-L/R/star) cells = {int(shock_bad.max())} "
              f"(0-2 = grid straddle, expected; >2 = spurious smear)")

    # -------------------- plot the worst few --------------------
    worst = results[:plot_worst]
    for idx, wt, me, nb, rho_L, rho_star, rho_R in worst:
        rho_ds = d["rho"][idx].astype(np.float64)
        w0 = d["w0"][idx].astype(np.float64) if "w0" in d else d["w"][idx, 0].astype(np.float64)
        rho0 = d["rho0"][idx].astype(np.float64) if "rho0" in d else rho_ds[0]
        rL, wL, rR, wR, x0, x_mid = _recover_riemann_ic(rho0, w0, x_min, x_max)
        rho_ex, w_ex, v_ex = Rie.solve_riemann_arz_xt(rL, wL, rR, wR, x_mid, t, x0=x0)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].plot(x_mid, rho_ds[-1], "o-", ms=3, label="WFT (dataset)")
        axes[0].plot(x_mid, rho_ex[-1], "--", lw=1.2, label="exact analytic")
        for yval, lbl in [(rL, "L"), (rho_star, "ρ*"), (rR, "R")]:
            axes[0].axhline(yval, color="gray", ls=":", lw=0.6)
        axes[0].set_title(f"rho final-time | {wt} | max err {me:.2e}")
        axes[0].legend(fontsize=8); axes[0].set_xlabel("x")

        # zoom on the discontinuity region
        diff = np.abs(np.diff(rho_ds[-1]))
        j = int(np.argmax(diff)) if diff.size else 0
        lo, hi = max(0, j - 10), min(nx - 1, j + 11)
        axes[1].plot(np.arange(lo, hi + 1), rho_ds[-1][lo:hi + 1], "o-", ms=4, label="WFT")
        axes[1].plot(np.arange(lo, hi + 1), rho_ex[-1][lo:hi + 1], "x--", ms=5, label="exact")
        for yval in (rL, rho_star, rR):
            axes[1].axhline(yval, color="gray", ls=":", lw=0.6)
        axes[1].set_title("zoom on transition (gray = L/ρ*/R plateaus)")
        axes[1].legend(fontsize=8); axes[1].set_xlabel("cell index")

        im = axes[2].imshow(np.abs(rho_ds - rho_ex), aspect="auto", origin="lower",
                            extent=[x_mid[0], x_mid[-1], t[0], t[-1]], cmap="Reds")
        axes[2].set_title("|WFT - exact| space-time"); axes[2].set_xlabel("x"); axes[2].set_ylabel("t")
        plt.colorbar(im, ax=axes[2])

        fig.suptitle(f"sample {idx} | L=({rL:.3f},{wL:.3f}) R=({rR:.3f},{wR:.3f}) "
                     f"ρ*={rho_star:.3f} | wave={wt}", fontsize=10)
        plt.tight_layout()
        fp = out_dir / f"wft_vs_exact_{idx:04d}.png"
        plt.savefig(fp, dpi=110); plt.close(fig)
        print(f"  plotted sample {idx} -> {fp.name}")

    print(f"\n[check] figures -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--pressure-form", default="rho", choices=["rho", "rho+rho2"])
    parser.add_argument("--n", type=int, default=40, help="riemann samples to check")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="max|WFT-exact| below which WFT == exact (no smear)")
    parser.add_argument("--plot-worst", type=int, default=6,
                        help="how many worst-error samples to plot")
    args = parser.parse_args()
    check(args.data, args.pressure_form, args.n, args.out, args.tol, args.plot_worst)


if __name__ == "__main__":
    main()
