"""ARZ eval: model vs numerical baselines (plan §8a).

Baselines:
  * godunov  : the existing reference solver (HLL + Strang relaxation).
  * weno5    : component-wise WENO5 (global LF) on (rho, y) + SSP-RK3,
               with Strang-split exact relaxation step.

Per-channel MAE on (rho, w, v), mean +/- std stratified by
(ic_type, num_segments). CSV output mirrors the LWR table.
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
from hyperbolic_pde.data.fvm import _apply_ghost_bc, _weno5_reconstruct_left


# --------------------------------------------------------------------------- #
# WENO5 systemized
# --------------------------------------------------------------------------- #
def _weno5_arz_rhs(rho, y, dx, boundary):
    """Component-wise WENO5 + global LF on (rho, y)."""
    # Use the spectral radius across cells as a global LF coefficient.
    rho_safe = np.maximum(rho, P.RHO_MIN)
    w = y / rho_safe
    v = w - P.pressure(rho)
    lam1 = v - rho * P.dpressure(rho)
    lam2 = v
    alpha = max(float(np.max(np.maximum(np.abs(lam1), np.abs(lam2)))), 1e-8)

    F1 = rho * v
    F2 = y * v

    def _component_rhs(u, f):
        ng = 3
        u_ext = _apply_ghost_bc(u, boundary, ng)
        f_ext = _apply_ghost_bc(f, boundary, ng)
        fp = 0.5 * (f_ext + alpha * u_ext)
        fm = 0.5 * (f_ext - alpha * u_ext)
        fp_s = fp[0:-1]
        fm_s = fm[1:][::-1]
        fhat_p = _weno5_reconstruct_left(fp_s)
        fhat_m = _weno5_reconstruct_left(fm_s)[::-1]
        fhat = fhat_p + fhat_m
        return -(fhat[1:] - fhat[:-1]) / dx

    return _component_rhs(rho, F1), _component_rhs(y, F2)


def _weno5_arz_step(rho, y, dx, dt, boundary):
    """Single SSP-RK3 WENO step on (rho, y)."""
    def L(rho_, y_):
        return _weno5_arz_rhs(rho_, y_, dx, boundary)

    Lr1, Ly1 = L(rho, y)
    rho1 = rho + dt * Lr1
    y1   = y   + dt * Ly1
    rho1 = np.maximum(rho1, P.RHO_MIN)

    Lr2, Ly2 = L(rho1, y1)
    rho2 = 0.75 * rho + 0.25 * (rho1 + dt * Lr2)
    y2   = 0.75 * y   + 0.25 * (y1   + dt * Ly2)
    rho2 = np.maximum(rho2, P.RHO_MIN)

    Lr3, Ly3 = L(rho2, y2)
    rho_n = (1.0/3.0) * rho + (2.0/3.0) * (rho2 + dt * Lr3)
    y_n   = (1.0/3.0) * y   + (2.0/3.0) * (y2   + dt * Ly3)
    return np.maximum(rho_n, P.RHO_MIN), y_n


def solve_arz_weno5(rho0, w0, x_min, x_max, t_max, nt_out, tau, cfl=0.2, boundary="ghost"):
    """Strang-split WENO5(arz) + exact source. No grid refinement here -- the
    point of WENO is to ride the train grid."""
    rho0 = np.asarray(rho0, dtype=np.float64)
    w0   = np.asarray(w0,   dtype=np.float64)
    nx = rho0.size
    # Finite-volume cell width: dx = L/nx (nx cells), matching the dataset's
    # midpoint grid. Using L/(nx-1) here gave the wrong propagation speed and a
    # half-cell spatial offset vs the ground truth. [x_min, x_max] is the FULL
    # domain extent (true edges), not the midpoint min/max.
    dx = (x_max - x_min) / nx
    t_out = np.linspace(0.0, t_max, nt_out, dtype=np.float64)
    rho = np.maximum(rho0.copy(), P.RHO_MIN)
    y = rho * w0
    rho_hist = np.empty((nt_out, nx), dtype=np.float32)
    w_hist   = np.empty((nt_out, nx), dtype=np.float32)
    rho_hist[0] = rho.astype(np.float32)
    w_hist[0]   = (y / np.maximum(rho, P.RHO_MIN)).astype(np.float32)

    t = 0.0
    k = 1
    while k < nt_out:
        w_now = y / np.maximum(rho, P.RHO_MIN)
        lam_max = float(P.spectral_radius(rho, w_now).max())
        lam_max = max(lam_max, 1e-6)
        dt = cfl * dx / lam_max
        dt = min(dt, t_out[k] - t)
        if dt <= 0:
            break
        # Strang.
        y = P.y_eq(rho) + (y - P.y_eq(rho)) * np.exp(-0.5 * dt / tau)
        rho, y = _weno5_arz_step(rho, y, dx, dt, boundary)
        y = P.y_eq(rho) + (y - P.y_eq(rho)) * np.exp(-0.5 * dt / tau)
        t += dt
        while k < nt_out and t >= t_out[k] - 1e-12:
            rho_hist[k] = rho.astype(np.float32)
            w_hist[k]   = (y / np.maximum(rho, P.RHO_MIN)).astype(np.float32)
            k += 1
    return rho_hist, w_hist


# --------------------------------------------------------------------------- #
# Eval driver
# --------------------------------------------------------------------------- #
def _run_baseline(name, rho0, w0, bundle, tau, boundary):
    nt = bundle.t.shape[0]
    # bundle.x are CELL MIDPOINTS (datagen: x_min + (i+0.5)*dx). Recover the
    # TRUE domain edges so the baseline solvers build the same midpoint grid as
    # the dataset/model rather than a linspace-endpoints grid half a cell off.
    xb = np.asarray(bundle.x, dtype=np.float64)
    dx = float(xb[1] - xb[0])
    x_min = float(xb[0] - 0.5 * dx)
    x_max = float(xb[-1] + 0.5 * dx)
    t_max = float(bundle.t.max())
    if name in ("godunov", "hll"):
        # 'godunov' = exact-Riemann flux (sharp contact, the LWR-equivalent).
        # 'hll'     = the old HLL flux (contact-smearing) -- kept selectable so
        #             the two can be compared. Previously 'godunov' silently ran
        #             HLL; that was the mislabel behind the over-smeared baseline.
        _, _, rho_h, w_h = Ref.solve_arz_reference(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=tau, cfl=0.4, boundary=boundary, refine=1,
            flux_scheme=("godunov" if name == "godunov" else "hll"),
        )
        return rho_h, w_h
    if name == "weno5":
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
    ap.add_argument("--samples", type=int, default=20,
                    help="Cap on number of eval samples (use all if dataset smaller).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--tau", type=float, default=None,
                    help="Override the relaxation tau used by the baselines "
                         "(defaults to the dataset's bundle.tau).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", type=Path, default=None,
                    help="If set, write per-sample comparison plots into this "
                         "directory (mirrors LWR eval_vs_numerical.py).")
    ap.add_argument("--n-plots", type=int, default=5,
                    help="Number of per-sample plots to save (default 5).")
    ap.add_argument("--model-variant", type=str, default="auto",
                    choices=["auto", "mark1", "mark1_router", "mark2", "mark2_1"],
                    help="Model class. 'auto' reads it off --model-section "
                         "(mark2_1 > mark2 > mark1_router > mark1).")
    ap.add_argument("--model-section", type=str, default="hypno_arz_mark2",
                    help="Config section for mark2 reconstruction.")
    args = ap.parse_args()

    bundle = load_arz_dataset(args.data)
    tau = float(args.tau) if args.tau is not None else float(bundle.tau)
    N = bundle.rho.shape[0]
    take = min(args.samples, N)

    # Match the closure used by the dataset (and presumably the model).
    from hyperbolic_pde.arz.physics_arz import set_pressure_form, get_pressure_form
    set_pressure_form(bundle.p_form)
    print(f"[eval_vs_numerical_arz] pressure_form={get_pressure_form()}")

    # Resolve variant: auto -> mark2_1 > mark2 > mark1 by section name.
    # (mark2_1 must be checked FIRST -- "mark2_1" contains "mark2".)
    variant = args.model_variant
    if variant == "auto":
        if "mark2_1" in args.model_section:
            variant = "mark2_1"
        elif "mark2" in args.model_section:
            variant = "mark2"
        elif "mark1_router" in args.model_section:
            variant = "mark1_router"
        else:
            variant = "mark1"

    # Load model. mark2 / mark2_1 = (rho,v) decoder; forward still returns
    # (rho, w, u_hats) with w recovered, so the eval path below is unchanged.
    if variant == "mark2_1":
        from hyperbolic_pde.arz.model_arz_mark2_1 import load_hypno_arz_mark2_1_from_checkpoint
        model, _ck_tau = load_hypno_arz_mark2_1_from_checkpoint(
            args.ckpt, device=args.device, config_path=args.config,
            model_section=args.model_section,
        )
    elif variant == "mark2":
        from hyperbolic_pde.arz.model_arz_mark2 import load_hypno_arz_mark2_from_checkpoint
        model, _ck_tau = load_hypno_arz_mark2_from_checkpoint(
            args.ckpt, device=args.device, config_path=args.config,
            model_section=args.model_section,
        )
    elif variant == "mark1_router":
        # Mark1 frame: forward returns (rho, w, u_hats), so the eval path below
        # is identical to plain mark1 -- only the loader/class differ.
        from hyperbolic_pde.arz.model_arz_mark1_router import (
            load_hypno_arz_mark1_router_from_checkpoint,
        )
        model, _ck_tau = load_hypno_arz_mark1_router_from_checkpoint(
            args.ckpt, device=args.device, config_path=args.config,
            model_section=args.model_section,
        )
    else:
        model, _ck_tau = load_hypno_arz_from_checkpoint(
            args.ckpt, device=args.device, config_path=args.config,
        )
    print(f"[eval_vs_numerical_arz] loaded {args.ckpt}  variant={variant}  tau_eval={tau}")

    x = torch.tensor(bundle.x, dtype=torch.float32, device=args.device)
    t = torch.tensor(bundle.t, dtype=torch.float32, device=args.device)

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    methods_in_order = ["model"] + baselines
    results = []  # rows: family, num_segments, method, channel, mean, std, n

    # Plotting setup (mirrors LWR eval_vs_numerical.py)
    plot_dir = None
    per_t_errors: dict = {}    # method -> list of [nt] arrays per (channel, sample)
    if args.figures is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_dir = Path(args.figures)
        plot_dir.mkdir(parents=True, exist_ok=True)
        for m in methods_in_order:
            per_t_errors[m] = {"rho": [], "w": [], "v": []}
        print(f"[eval_vs_numerical_arz] writing figures to {plot_dir}")

    # Iterate samples, compute MAE per (family, segments, method, channel).
    buckets: dict = {}
    for i in range(take):
        fam = str(bundle.ic_type[i])
        seg = int(bundle.num_segments[i])
        key = (fam, seg)
        if key not in buckets:
            buckets[key] = {m: {"rho": [], "w": [], "v": []}
                            for m in (["model"] + baselines)}

        rho_gt = bundle.rho[i].astype(np.float64)
        w_gt   = bundle.w[i].astype(np.float64)
        v_gt   = w_gt - P.pressure(rho_gt)
        rho0   = bundle.rho0[i].astype(np.float64)
        w0     = bundle.w0[i].astype(np.float64)

        # Model.
        rho0_t = torch.tensor(rho0[None], dtype=torch.float32, device=args.device)
        w0_t   = torch.tensor(w0[None],   dtype=torch.float32, device=args.device)
        with torch.no_grad():
            if variant in ("mark2", "mark2_1"):
                # Get v DIRECTLY from the decoder (the model's actual output),
                # NOT v = w - p(rho). The latter just inverts the w = v + p(rho)
                # recovery, so it would hide any inconsistency in the decoded
                # (rho, v) pair -- exactly what we want to see (e.g. a spurious
                # w-band means v and p(rho) don't cancel across the 1-wave).
                rho_pt, v_pt, w_pt, _ = model.forward_primitive(rho0_t, w0_t, x, t)
                rho_p = rho_pt[0].cpu().numpy()
                v_p   = v_pt[0].cpu().numpy()
                w_p   = w_pt[0].cpu().numpy()
            else:
                rho_pt, w_pt, _ = model(rho0_t, w0_t, x, t)
                rho_p = rho_pt[0].cpu().numpy(); w_p = w_pt[0].cpu().numpy()
                v_p = w_p - P.pressure(rho_p)   # mark1 has no direct v output
        buckets[key]["model"]["rho"].append(float(np.mean(np.abs(rho_p - rho_gt))))
        buckets[key]["model"]["w"  ].append(float(np.mean(np.abs(w_p   - w_gt  ))))
        buckets[key]["model"]["v"  ].append(float(np.mean(np.abs(v_p   - v_gt  ))))

        # Stash predictions for plotting (only kept when --figures is set).
        # Each entry is (rho, w, v). For the model, v is the DIRECT decoder
        # output; for baselines it is the recovery w - p(rho) (a numerical
        # solver has no separate v field).
        sample_preds = {"model": (rho_p, w_p, v_p)} if plot_dir is not None else None

        # For exact-Riemann datasets (tau=inf) run baselines as homogeneous ARZ.
        baseline_tau = tau if np.isfinite(tau) else 1e6
        for m in baselines:
            try:
                rho_b, w_b = _run_baseline(m, rho0, w0, bundle, baseline_tau, args.boundary)
            except Exception as e:
                print(f"  [warn] {m} failed on sample {i}: {e}", flush=True)
                continue
            v_b = w_b - P.pressure(rho_b)
            buckets[key][m]["rho"].append(float(np.mean(np.abs(rho_b - rho_gt))))
            buckets[key][m]["w"  ].append(float(np.mean(np.abs(w_b   - w_gt  ))))
            buckets[key][m]["v"  ].append(float(np.mean(np.abs(v_b   - v_gt  ))))
            if sample_preds is not None:
                sample_preds[m] = (rho_b, w_b, v_b)

        # Per-sample plot + per-t error accumulation.
        if plot_dir is not None:
            x_np = bundle.x.astype(np.float64)
            t_np = bundle.t.astype(np.float64)
            # Per-t error (mean over x) for the global error-vs-t plot.
            for m, (rho_m, w_m, v_m) in sample_preds.items():
                per_t_errors[m]["rho"].append(np.mean(np.abs(rho_m - rho_gt), axis=1))
                per_t_errors[m]["w"  ].append(np.mean(np.abs(w_m   - w_gt  ), axis=1))
                per_t_errors[m]["v"  ].append(np.mean(np.abs(v_m   - v_gt  ), axis=1))
            if i < args.n_plots:
                for ch_name, gt_arr, pred_dict in (
                    ("rho", rho_gt, {m: sample_preds[m][0] for m in sample_preds}),
                    ("v",   v_gt,   {m: sample_preds[m][2] for m in sample_preds}),
                    ("w",   w_gt,   {m: sample_preds[m][1] for m in sample_preds}),
                ):
                    methods_present = ["GT"] + [m for m in methods_in_order if m in pred_dict]
                    n_cols = len(methods_present)
                    # 3 rows now: GT/preds (row 0), shared-scale errors (row 1),
                    # per-method-normalised errors for pattern visualisation (row 2).
                    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 11),
                                             constrained_layout=True)
                    vmin = float(gt_arr.min()); vmax = float(gt_arr.max())
                    # Compute per-method MAE up-front so it can go in panel titles.
                    mae_per_method = {
                        name: float(np.mean(np.abs(pred_dict[name] - gt_arr)))
                        for name in methods_present[1:]
                    }
                    # Row 0: GT + each method, with MAE annotated on the title.
                    err_vmax = None
                    for c, name in enumerate(methods_present):
                        arr = gt_arr if name == "GT" else pred_dict[name]
                        im = axes[0, c].pcolormesh(x_np, t_np, arr, shading="auto",
                                                   cmap="jet", vmin=vmin, vmax=vmax)
                        if name == "GT":
                            axes[0, c].set_title(name)
                        else:
                            axes[0, c].set_title(
                                f"{name}\nMAE={mae_per_method[name]:.2e}"
                            )
                        axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
                        fig.colorbar(im, ax=axes[0, c], label=ch_name)
                        if name != "GT":
                            err = np.abs(arr - gt_arr)
                            err_vmax = err.max() if err_vmax is None else max(err_vmax, err.max())
                    # Row 1: |method - GT| on a SHARED colour scale -- compare magnitudes.
                    axes[1, 0].axis("off")
                    axes[1, 0].text(
                        0.5, 0.5,
                        "shared err scale\n(compare magnitudes)",
                        ha="center", va="center",
                        transform=axes[1, 0].transAxes,
                        fontsize=10,
                    )
                    for c, name in enumerate(methods_present[1:], start=1):
                        err = np.abs(pred_dict[name] - gt_arr)
                        im = axes[1, c].pcolormesh(x_np, t_np, err, shading="auto",
                                                   cmap="magma", vmin=0, vmax=err_vmax)
                        axes[1, c].set_title(f"|{name} - GT|  (shared)")
                        axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
                        fig.colorbar(im, ax=axes[1, c])
                    # Row 2: |method - GT| with per-method scaling -- reveal pattern
                    # regardless of magnitude. Bright in this row does NOT mean bigger
                    # error in absolute terms; check row-1 colourbars or the panel
                    # title MAE for that.
                    axes[2, 0].axis("off")
                    axes[2, 0].text(
                        0.5, 0.5,
                        "per-method err scale\n(reveal pattern)",
                        ha="center", va="center",
                        transform=axes[2, 0].transAxes,
                        fontsize=10,
                    )
                    for c, name in enumerate(methods_present[1:], start=1):
                        err = np.abs(pred_dict[name] - gt_arr)
                        local_vmax = float(err.max()) if err.size else 0.0
                        im = axes[2, c].pcolormesh(x_np, t_np, err, shading="auto",
                                                   cmap="magma", vmin=0,
                                                   vmax=local_vmax if local_vmax > 0 else None)
                        axes[2, c].set_title(
                            f"|{name} - GT|  (own scale, max={local_vmax:.2e})"
                        )
                        axes[2, c].set_xlabel("x"); axes[2, c].set_ylabel("t")
                        fig.colorbar(im, ax=axes[2, c])
                    mae_strs = [f"{m}={mae_per_method[m]:.3e}"
                                for m in methods_in_order if m in mae_per_method]
                    fig.suptitle(
                        f"Sample {i}  channel={ch_name}  fam={fam} seg={seg}  "
                        f"MAE: " + "  ".join(mae_strs)
                    )
                    fig.savefig(plot_dir / f"compare_sample_{i}_{ch_name}.png", dpi=150)
                    plt.close(fig)

    # Emit CSV.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["family", "num_segments", "method", "channel", "mean", "std", "n"])
        for (fam, seg), bymethod in sorted(buckets.items()):
            for m, chans in bymethod.items():
                for ch in ("rho", "w", "v"):
                    arr = np.array(chans[ch], dtype=np.float64)
                    if arr.size == 0:
                        continue
                    wr.writerow([fam, seg, m, ch, float(arr.mean()),
                                 float(arr.std(ddof=0)), int(arr.size)])

    # Global error-vs-time plot (one panel per channel).
    if plot_dir is not None and per_t_errors:
        t_np = bundle.t.astype(np.float64)
        fig, axes = plt.subplots(1, 3, figsize=(20, 4.5), constrained_layout=True)
        colors = {"model": "tab:blue", "weno5": "tab:orange",
                  "godunov": "tab:green", "hll": "tab:red"}
        for ci, ch_name in enumerate(("rho", "v", "w")):
            for m in methods_in_order:
                arrs = per_t_errors[m][ch_name]
                if not arrs:
                    continue
                mean_curve = np.stack(arrs).mean(axis=0)
                axes[ci].plot(t_np, mean_curve, label=m,
                              color=colors.get(m, None))
            axes[ci].set_xlabel("t")
            axes[ci].set_ylabel(f"mean |{ch_name}_pred - {ch_name}_gt| over x")
            axes[ci].set_title(f"Error vs time ({ch_name})")
            axes[ci].set_yscale("log")
            axes[ci].legend(); axes[ci].grid(True, alpha=0.3)
        fig.savefig(plot_dir / "error_vs_time.png", dpi=150)
        plt.close(fig)
        print(f"[eval_vs_numerical_arz] saved error_vs_time.png and "
              f"{min(take, args.n_plots) * 3} per-sample plots in {plot_dir}")

    # Pretty print to stdout.
    print(f"\n[eval_vs_numerical_arz] wrote {args.out}  (N={take} samples)")
    for (fam, seg), bymethod in sorted(buckets.items()):
        print(f"  {fam} seg={seg}:")
        for m, chans in bymethod.items():
            cells = []
            for ch in ("rho", "w", "v"):
                arr = np.array(chans[ch])
                if arr.size:
                    cells.append(f"{ch} {arr.mean():.3e}+/-{arr.std():.1e}")
            print(f"    {m:8s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
