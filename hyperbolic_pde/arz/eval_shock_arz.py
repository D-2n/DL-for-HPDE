"""ARZ shock-band eval (plan §8b).

The discontinuity detector is run on the ground-truth fields, split by wave
type:

  * 1-shock band   : |dv| large with w approximately continuous.
                     Sign by Lax test: lambda1(U_L) < lambda1(U_R).
  * 2-contact band : |drho| large with v approximately continuous.

Both detectors use a TV-multiplier pass on the appropriate signed-jump
indicator, followed by a +/- b cell dilation. Reports MAE on the full field
and inside each band (and combined).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.model_arz import HypNO_ARZ, load_hypno_arz_from_checkpoint


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def _tv(field: np.ndarray) -> float:
    """Total variation along axis -1."""
    return float(np.abs(np.diff(field, axis=-1)).sum())


def _dilate_1d(mask: np.ndarray, half: int) -> np.ndarray:
    """+/- half cell dilation along the last axis."""
    if half <= 0:
        return mask
    out = mask.copy()
    for k in range(1, half + 1):
        out[..., k:]    |= mask[..., :-k]
        out[..., :-k]   |= mask[..., k:]
    return out


def detect_1shock_band(
    rho: np.ndarray, w: np.ndarray, v: np.ndarray,
    tau_shock: float, half: int, tv_mult: float,
) -> np.ndarray:
    """Boolean mask of cells in a 1-shock band, computed on GT fields.

    Criterion (per timeslice):
      jump_v = -dv (decreasing v across a Lax 1-shock for concave LWR-like family)
      indicator = relu(-dv); mark cells where indicator > max(tau_shock, tv_mult * TV(v)/nx).
      Then +/- half dilation. Sign of the jump is consistent with lambda1_L < lambda1_R.
    """
    nt, nx = rho.shape
    out = np.zeros_like(rho, dtype=bool)
    dp_rho = 1.0 + 2.0 * rho
    lam1 = v - rho * dp_rho
    for k in range(nt):
        dv = np.diff(v[k])                              # nx-1
        dlam1 = np.diff(lam1[k])                        # nx-1, < 0 implies Lax-1 shock
        # Signed indicator: penalise drops in lambda1 (Lax) and drops in v.
        ind = np.maximum(-dlam1, 0.0) + np.maximum(-dv, 0.0)
        thr = max(tau_shock, tv_mult * _tv(v[k]) / max(nx, 1))
        # Map interface mask back to cells (cell i is "in" if interface i-1/2 or i+1/2 flagged).
        cell = np.zeros(nx, dtype=bool)
        flagged = ind > thr
        cell[1:]  |= flagged
        cell[:-1] |= flagged
        out[k] = cell
    return _dilate_1d(out, half)


def detect_contact_band(
    rho: np.ndarray, w: np.ndarray, v: np.ndarray,
    tau_shock: float, half: int, tv_mult: float,
) -> np.ndarray:
    """Boolean mask of cells in a 2-contact band: |drho| large with v approx continuous."""
    nt, nx = rho.shape
    out = np.zeros_like(rho, dtype=bool)
    for k in range(nt):
        drho = np.diff(rho[k])
        dv   = np.diff(v[k])
        ind  = np.abs(drho) - 0.5 * np.abs(dv)          # prefer drho-only jumps
        thr  = max(tau_shock, tv_mult * _tv(rho[k]) / max(nx, 1))
        cell = np.zeros(nx, dtype=bool)
        flagged = ind > thr
        cell[1:]  |= flagged
        cell[:-1] |= flagged
        out[k] = cell
    return _dilate_1d(out, half)


# --------------------------------------------------------------------------- #
# Eval driver
# --------------------------------------------------------------------------- #
def _mae(arr_pred, arr_gt, mask=None):
    if mask is None:
        return float(np.mean(np.abs(arr_pred - arr_gt)))
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(arr_pred[mask] - arr_gt[mask])))


def _run_baseline(name, rho0, w0, bundle, tau, boundary):
    nt = bundle.t.shape[0]
    x_min = float(bundle.x.min()); x_max = float(bundle.x.max())
    t_max = float(bundle.t.max())
    if name == "godunov":
        _, _, rho_h, w_h = Ref.solve_arz_reference(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=tau, cfl=0.4, boundary=boundary, refine=1,
        )
        return rho_h, w_h
    if name == "weno5":
        from hyperbolic_pde.arz.eval_vs_numerical_arz import solve_arz_weno5
        return solve_arz_weno5(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=tau, cfl=0.2, boundary=boundary,
        )
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Checkpoint file (legacy dict OR bare state_dict from the new trainer).")
    ap.add_argument("--config", type=Path, default=None,
                    help="config.yaml describing the architecture (required for "
                         "bare-state_dict checkpoints if it can't be auto-located "
                         "next to the ckpt).")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--baselines", type=str, default="weno5,godunov")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--tau-shock", type=float, default=0.06)
    ap.add_argument("--band-halfwidth", type=int, default=2)
    ap.add_argument("--tv-mult", type=float, default=1.5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--tau", type=float, default=None,
                    help="Override the relaxation tau used by the baselines "
                         "(defaults to the dataset's bundle.tau).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", type=Path, default=None,
                    help="If set, write per-sample band-overlay plots into "
                         "this directory.")
    ap.add_argument("--n-plots", type=int, default=5)
    args = ap.parse_args()

    bundle = load_arz_dataset(args.data)
    tau = float(args.tau) if args.tau is not None else float(bundle.tau)
    N = bundle.rho.shape[0]
    take = min(args.samples, N)

    model, _ck_tau = load_hypno_arz_from_checkpoint(
        args.ckpt, device=args.device, config_path=args.config,
    )
    print(f"[eval_shock_arz] loaded {args.ckpt}  tau_eval={tau}")

    x = torch.tensor(bundle.x, dtype=torch.float32, device=args.device)
    t = torch.tensor(bundle.t, dtype=torch.float32, device=args.device)

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    methods = ["model"] + baselines

    plot_dir = None
    if args.figures is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_dir = Path(args.figures)
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval_shock_arz] writing figures to {plot_dir}")

    # rows: family, num_segments, method, region, channel, mean, std, n
    rows = []
    # accumulate per (fam,seg,method,region,channel) -> list of MAE
    acc: dict = {}

    for i in range(take):
        fam = str(bundle.ic_type[i])
        seg = int(bundle.num_segments[i])
        rho_gt = bundle.rho[i].astype(np.float64)
        w_gt   = bundle.w[i].astype(np.float64)
        v_gt   = w_gt - (rho_gt + rho_gt * rho_gt)
        rho0 = bundle.rho0[i].astype(np.float64)
        w0   = bundle.w0[i].astype(np.float64)

        # Bands on GT.
        mask_1shock = detect_1shock_band(rho_gt, w_gt, v_gt,
                                         args.tau_shock, args.band_halfwidth, args.tv_mult)
        mask_contact = detect_contact_band(rho_gt, w_gt, v_gt,
                                           args.tau_shock, args.band_halfwidth, args.tv_mult)
        mask_combined = mask_1shock | mask_contact

        # Predictions per method.
        preds = {}
        with torch.no_grad():
            rho_p, w_p, _ = model(
                torch.tensor(rho0[None], dtype=torch.float32, device=args.device),
                torch.tensor(w0[None],   dtype=torch.float32, device=args.device),
                x, t,
            )
        preds["model"] = (rho_p[0].cpu().numpy(), w_p[0].cpu().numpy())
        for m in baselines:
            try:
                preds[m] = _run_baseline(m, rho0, w0, bundle, tau, args.boundary)
            except Exception as e:
                print(f"  [warn] {m} sample {i}: {e}", flush=True)

        for m, (rho_b, w_b) in preds.items():
            v_b = w_b - (rho_b + rho_b * rho_b)
            for ch_name, arr_b, arr_gt in (
                ("rho", rho_b, rho_gt), ("w", w_b, w_gt), ("v", v_b, v_gt),
            ):
                regions = {
                    "full":     None,
                    "1shock":   mask_1shock,
                    "contact":  mask_contact,
                    "combined": mask_combined,
                }
                for r_name, mask in regions.items():
                    val = _mae(arr_b, arr_gt, mask)
                    if not np.isfinite(val):
                        continue
                    acc.setdefault((fam, seg, m, r_name, ch_name), []).append(val)

        # Per-sample shock-band plot (rho channel: shows GT, model error,
        # and the detected band overlay).
        if plot_dir is not None and i < args.n_plots:
            x_np = bundle.x.astype(np.float64)
            t_np = bundle.t.astype(np.float64)
            n_meth = 1 + len(preds)  # GT + each method
            fig, axes = plt.subplots(2, n_meth, figsize=(4 * n_meth, 8),
                                     constrained_layout=True)
            # Row 0 col 0: GT rho with 1-shock band hatched and contact band dotted.
            vmin, vmax = float(rho_gt.min()), float(rho_gt.max())
            im = axes[0, 0].pcolormesh(x_np, t_np, rho_gt, shading="auto",
                                       cmap="jet", vmin=vmin, vmax=vmax)
            axes[0, 0].contour(x_np, t_np, mask_1shock.astype(float),
                               levels=[0.5], colors="white", linewidths=0.7)
            axes[0, 0].contour(x_np, t_np, mask_contact.astype(float),
                               levels=[0.5], colors="black", linewidths=0.7,
                               linestyles="dashed")
            axes[0, 0].set_title("GT rho  (white=1-shock, black=contact)")
            axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("t")
            fig.colorbar(im, ax=axes[0, 0])

            # Row 0 cols 1..: each method's rho prediction.
            method_order = list(preds.keys())
            err_vmax = None
            for c, m in enumerate(method_order, start=1):
                arr = preds[m][0]
                im = axes[0, c].pcolormesh(x_np, t_np, arr, shading="auto",
                                           cmap="jet", vmin=vmin, vmax=vmax)
                axes[0, c].set_title(f"{m} rho")
                axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
                fig.colorbar(im, ax=axes[0, c])
                err = np.abs(arr - rho_gt)
                err_vmax = err.max() if err_vmax is None else max(err_vmax, err.max())

            # Row 1 col 0: GT rho restricted to combined band (highlight where
            # the bands are).
            band_view = np.where(mask_combined, rho_gt, np.nan)
            im = axes[1, 0].pcolormesh(x_np, t_np, band_view, shading="auto",
                                       cmap="jet", vmin=vmin, vmax=vmax)
            axes[1, 0].set_title("GT inside combined band")
            axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("t")
            fig.colorbar(im, ax=axes[1, 0])
            # Row 1 cols 1..: |method - GT| with the band outlined.
            for c, m in enumerate(method_order, start=1):
                err = np.abs(preds[m][0] - rho_gt)
                im = axes[1, c].pcolormesh(x_np, t_np, err, shading="auto",
                                           cmap="magma", vmin=0, vmax=err_vmax)
                axes[1, c].contour(x_np, t_np, mask_1shock.astype(float),
                                   levels=[0.5], colors="cyan", linewidths=0.6)
                axes[1, c].contour(x_np, t_np, mask_contact.astype(float),
                                   levels=[0.5], colors="lime", linewidths=0.6,
                                   linestyles="dashed")
                axes[1, c].set_title(f"|{m} - GT|")
                axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
                fig.colorbar(im, ax=axes[1, c])

            mae_full = {m: float(np.mean(np.abs(preds[m][0] - rho_gt))) for m in method_order}
            mae_1sh  = {m: _mae(preds[m][0], rho_gt, mask_1shock) for m in method_order}
            mae_con  = {m: _mae(preds[m][0], rho_gt, mask_contact) for m in method_order}
            mae_str = "  ".join(
                f"{m}: full={mae_full[m]:.2e}  1sh={mae_1sh[m]:.2e}  con={mae_con[m]:.2e}"
                for m in method_order
            )
            fig.suptitle(
                f"Sample {i}  fam={fam} seg={seg}  rho MAE -- {mae_str}"
            )
            fig.savefig(plot_dir / f"shock_sample_{i}_rho.png", dpi=150)
            plt.close(fig)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["family", "num_segments", "method", "region", "channel",
                     "mean", "std", "n"])
        for key, vals in sorted(acc.items()):
            arr = np.array(vals, dtype=np.float64)
            wr.writerow([*key, float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)])
    print(f"\n[eval_shock_arz] wrote {args.out}  (N={take} samples)")

    # Pretty print key combined-region rows.
    print("\n  region=combined, channel=rho")
    for key, vals in sorted(acc.items()):
        fam, seg, m, region, ch = key
        if region != "combined" or ch != "rho":
            continue
        arr = np.array(vals)
        print(f"    {fam} seg={seg}  {m:8s}  {arr.mean():.3e} +/- {arr.std():.1e}")


if __name__ == "__main__":
    main()
