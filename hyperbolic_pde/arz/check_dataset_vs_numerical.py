"""Compare dataset GT against numerically re-solved HLL and Godunov solutions.

For each sample: load (rho0, w0) from the npz, re-run HLL and Godunov from
scratch using the CURRENT physics_arz pressure form, then plot GT vs both
numerical solutions side by side.

This is the canonical check for pressure-form mismatches: if the GT was
generated with rho+rho2 but the model/eval uses rho, the numerical re-solve
(which uses the current P module) will disagree with the stored GT.

Usage (CLEPS):
    python -m hyperbolic_pde.arz.check_dataset_vs_numerical \
        --data /home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho.npz \
        --pressure-form rho \
        --n 6 \
        --out /home/dzdrale/scratch/results/dataset_check/

Usage (local):
    python -m hyperbolic_pde.arz.check_dataset_vs_numerical \
        --data data/arz_mixed_wft_prho.npz \
        --pressure-form rho \
        --n 6 \
        --out results/dataset_check/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.reference_arz import solve_arz_reference


def _resolve(rho_hist: np.ndarray, w_hist: np.ndarray, flux: str,
             x_min: float, x_max: float, t_max: float,
             rho0: np.ndarray, w0: np.ndarray, nt: int) -> tuple:
    _, _, rho_num, w_num = solve_arz_reference(
        rho0, w0, x_min=x_min, x_max=x_max, t_max=t_max,
        nt_out=nt, tau=1e12, cfl=0.4, boundary="ghost",
        refine=1, flux_scheme=flux,
    )
    v_num = w_num - P.pressure(rho_num)
    return rho_num, w_num, v_num


def check(data_path: Path, pressure_form: str, n_samples: int, out_dir: Path,
          ic_filter: str | None) -> None:
    P.set_pressure_form(pressure_form)
    print(f"[check] pressure_form = {P.get_pressure_form()}")

    d = np.load(data_path, allow_pickle=True)
    stored_pform = str(d["p_form"]) if "p_form" in d else "unknown"
    print(f"[check] dataset p_form = {stored_pform}")
    if stored_pform != pressure_form:
        print(f"[check] WARNING: dataset p_form '{stored_pform}' != requested '{pressure_form}'")

    N_total = d["rho"].shape[0]
    nt, nx = d["rho"].shape[1], d["rho"].shape[2]
    x = d["x"] if "x" in d else np.linspace(-1, 1, nx)
    t = d["t"] if "t" in d else np.linspace(0, 1, nt)
    x_min, x_max, t_max = float(x[0]), float(x[-1]), float(t[-1])
    ic_types = d["ic_type"] if "ic_type" in d else np.array(["?"] * N_total)

    indices = np.arange(N_total)
    if ic_filter:
        indices = indices[ic_types[indices] == ic_filter]
    rng = np.random.default_rng(0)
    indices = rng.choice(indices, size=min(n_samples, len(indices)), replace=False)
    indices = sorted(indices)

    out_dir.mkdir(parents=True, exist_ok=True)

    errors_hll = []
    errors_god = []

    for idx in indices:
        rho0 = d["rho0"][idx].astype(np.float64) if "rho0" in d else d["rho"][idx, 0].astype(np.float64)
        w0   = d["w0"][idx].astype(np.float64)   if "w0"   in d else None
        if w0 is None:
            v0 = d["v0"][idx].astype(np.float64) if "v0" in d else None
            if v0 is None:
                raise ValueError("npz must contain w0 or v0")
            w0 = v0 + P.pressure(rho0)

        rho_gt = d["rho"][idx]
        w_gt   = d["w"][idx]
        v_gt   = d["v"][idx] if "v" in d else w_gt - P.pressure(rho_gt)

        rho_hll, w_hll, v_hll = _resolve(rho_gt, w_gt, "hll",
                                          x_min, x_max, t_max, rho0, w0, nt)
        rho_god, w_god, v_god = _resolve(rho_gt, w_gt, "godunov",
                                          x_min, x_max, t_max, rho0, w0, nt)

        err_hll = float(np.mean(np.abs(rho_gt - rho_hll)))
        err_god = float(np.mean(np.abs(rho_gt - rho_god)))
        errors_hll.append(err_hll)
        errors_god.append(err_god)
        ic_label = str(ic_types[idx])

        # --- Figure: final-time profiles ---
        fig, axes = plt.subplots(3, 3, figsize=(14, 9))
        fields = [("rho", rho_gt[-1], rho_hll[-1], rho_god[-1]),
                  ("w",   w_gt[-1],   w_hll[-1],   w_god[-1]),
                  ("v",   v_gt[-1],   v_hll[-1],   v_god[-1])]
        for row, (fname, gt_f, hll_f, god_f) in enumerate(fields):
            axes[row, 0].plot(x, gt_f,  label="GT (dataset)")
            axes[row, 0].plot(x, hll_f, label=f"HLL (err={np.mean(np.abs(gt_f-hll_f)):.3e})", ls="--")
            axes[row, 0].plot(x, god_f, label=f"Godunov (err={np.mean(np.abs(gt_f-god_f)):.3e})", ls=":")
            axes[row, 0].set_title(f"{fname} final-time"); axes[row, 0].legend(fontsize=7)

            im1 = axes[row, 1].imshow(np.abs(gt_f.reshape(1,-1) - rho_gt * 0 +
                                             (d["rho"][idx] if fname=="rho" else
                                              d["w"][idx]   if fname=="w"   else v_gt)),
                                      aspect="auto", origin="lower")
            axes[row, 1].set_title(f"{fname} GT heatmap")
            plt.colorbar(im1, ax=axes[row, 1])

            diff = (rho_hll if fname=="rho" else w_hll if fname=="w" else v_hll) - \
                   (d["rho"][idx] if fname=="rho" else d["w"][idx] if fname=="w" else v_gt)
            im2 = axes[row, 2].imshow(np.abs(diff), aspect="auto", origin="lower", cmap="Reds")
            axes[row, 2].set_title(f"|HLL - GT| heatmap")
            plt.colorbar(im2, ax=axes[row, 2])

        fig.suptitle(f"Sample {idx} | {ic_label} | p_form={P.get_pressure_form()} | "
                     f"rho MAE: HLL={err_hll:.3e} God={err_god:.3e}", fontsize=9)
        plt.tight_layout()
        fig_path = out_dir / f"sample_{idx:04d}.png"
        plt.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"  [{idx}] {ic_label}  HLL MAE={err_hll:.3e}  Godunov MAE={err_god:.3e}  -> {fig_path.name}")

    print(f"\n[check] mean rho MAE over {len(indices)} samples:")
    print(f"  HLL     = {np.mean(errors_hll):.4e}")
    print(f"  Godunov = {np.mean(errors_god):.4e}")
    print(f"[check] figures saved to {out_dir}")

    # CSV summary
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("sample_idx,ic_type,hll_mae_rho,godunov_mae_rho\n")
        for idx, eh, eg in zip(indices, errors_hll, errors_god):
            f.write(f"{idx},{ic_types[idx]},{eh:.6e},{eg:.6e}\n")
    print(f"[check] CSV -> {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--pressure-form", default="rho", choices=["rho", "rho+rho2"])
    parser.add_argument("--n", type=int, default=6, help="Number of samples to check")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ic-filter", default=None,
                        help="Only check samples of this IC type "
                             "(riemann_stratified | piecewise_constant_stratified | piecewise_sine)")
    args = parser.parse_args()
    check(args.data, args.pressure_form, args.n, args.out, args.ic_filter)


if __name__ == "__main__":
    main()
