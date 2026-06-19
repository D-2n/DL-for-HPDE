"""ARZ mark-0 (orig) paper evaluation.

Evaluates a HypNO_ARZ_Orig checkpoint against FNO, Godunov, and WENO5 on
both a homogeneous multi-discontinuity dataset and an exact-Riemann dataset.
Ground truth is always shown. All three channels (rho, w, v) are reported.

Outputs per dataset (written under --out_dir/<homog|riemann>/):
  * figures/compare_<idx>_<channel>.png  -- 3-row: rho0 strip / fields / |err|
  * mae_vs_time_<channel>.png            -- error vs t, averaged over samples
  * summary.csv                          -- long-format MAE table
  * summary.txt                          -- plain-text scan

Usage:
    python -m hyperbolic_pde.arz.arz_orig_paper_eval \\
        --ckpt /path/to/checkpoint_epoch170.pt \\
        --homog-data /path/to/arz_stratified_homog_prho.npz \\
        --riemann-data /path/to/arz_riemann_exact_prho_strat.npz \\
        --fno-weights /path/to/fno_arz_prho.pt \\
        --config hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml \\
        --out_dir /path/to/results/mark0_paper_eval \\
        --n-samples 100 --n-plots 10
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.model_arz_orig import load_hypno_arz_orig_from_checkpoint
from hyperbolic_pde.arz.eval_vs_numerical_arz import solve_arz_weno5
from hyperbolic_pde.models.fno import FNO2d

CHANNELS = ["rho", "w", "v"]
COLORS = {
    "HypNO": "tab:blue",
    "FNO": "tab:purple",
    "godunov": "tab:green",
    "weno5": "tab:orange",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred - truth).mean())


def _run_numerical(name: str, rho0, w0, bundle, boundary: str) -> tuple[np.ndarray, np.ndarray]:
    nt = bundle.t.shape[0]
    xb = np.asarray(bundle.x, dtype=np.float64)
    dx = float(xb[1] - xb[0])
    x_min = float(xb[0] - 0.5 * dx)
    x_max = float(xb[-1] + 0.5 * dx)
    t_max = float(bundle.t.max())
    tau = float(bundle.tau)
    baseline_tau = tau if np.isfinite(tau) else 1e6
    if name == "godunov":
        _, _, rho_h, w_h = Ref.solve_arz_reference(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=baseline_tau, cfl=0.4, boundary=boundary, refine=1,
            flux_scheme="godunov",
        )
        return rho_h, w_h
    if name == "weno5":
        return solve_arz_weno5(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=baseline_tau, cfl=0.2, boundary=boundary,
        )
    raise ValueError(name)


def _make_fno_input(rho0_np, w0_np, x_np, t_np) -> torch.Tensor:
    x_t = torch.tensor(x_np, dtype=torch.float32)
    t_t = torch.tensor(t_np, dtype=torch.float32)
    X, T = torch.meshgrid(x_t, t_t, indexing="ij")
    nt = t_t.numel()
    rho0_g = torch.tensor(rho0_np, dtype=torch.float32).unsqueeze(1).repeat(1, nt)
    w0_g = torch.tensor(w0_np, dtype=torch.float32).unsqueeze(1).repeat(1, nt)
    return torch.stack([X, T, rho0_g, w0_g], dim=0).unsqueeze(0)  # (1, 4, nx, nt)


def _load_fno(weights: Path, cfg_section: dict, device: str) -> FNO2d:
    model = FNO2d(
        in_channels=4, out_channels=2,
        width=int(cfg_section["width"]),
        modes_x=int(cfg_section["modes_x"]),
        modes_t=int(cfg_section["modes_t"]),
        layers=int(cfg_section["layers"]),
    ).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Per-dataset eval
# --------------------------------------------------------------------------- #

def eval_dataset(
    tag: str,
    bundle,
    hypno_model,
    fno_model,           # None if not provided
    baselines: list[str],
    device: str,
    boundary: str,
    n_samples: int,
    n_plots: int,
    out_dir: Path,
    plt,
):
    """Run the full eval loop on one dataset bundle and write all outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    N = bundle.rho.shape[0]
    take = min(n_samples, N)

    x_np = bundle.x.astype(np.float64)
    t_np = bundle.t.astype(np.float64)
    x_t = torch.tensor(bundle.x, dtype=torch.float32, device=device)
    t_t = torch.tensor(bundle.t, dtype=torch.float32, device=device)

    active_methods: list[str] = ["HypNO"]
    if fno_model is not None:
        active_methods.append("FNO")
    active_methods.extend(baselines)

    # Accumulators: method -> channel -> list of per-sample MAE.
    acc: dict[str, dict[str, list[float]]] = {
        m: {ch: [] for ch in CHANNELS} for m in active_methods
    }
    # Per-time-step mean |err| accumulators for the error-vs-time plot.
    per_t: dict[str, dict[str, list[np.ndarray]]] = {
        m: {ch: [] for ch in CHANNELS} for m in active_methods
    }

    print(f"\n[{tag}] evaluating {take}/{N} samples  methods={active_methods}", flush=True)

    for i in range(take):
        rho_gt = bundle.rho[i].astype(np.float64)
        w_gt   = bundle.w[i].astype(np.float64)
        v_gt   = w_gt - P.pressure(rho_gt)
        rho0   = bundle.rho0[i].astype(np.float64)
        w0     = bundle.w0[i].astype(np.float64)
        gt     = {"rho": rho_gt, "w": w_gt, "v": v_gt}

        preds: dict[str, dict[str, np.ndarray]] = {}

        # HypNO-orig.
        with torch.no_grad():
            rho_p, w_p, _ = hypno_model(
                torch.tensor(rho0[None], dtype=torch.float32, device=device),
                torch.tensor(w0[None],   dtype=torch.float32, device=device),
                x_t, t_t,
            )
        rho_p = rho_p[0].cpu().numpy()
        w_p   = w_p[0].cpu().numpy()
        preds["HypNO"] = {"rho": rho_p, "w": w_p, "v": w_p - P.pressure(rho_p)}

        # FNO.
        if fno_model is not None:
            inp = _make_fno_input(rho0, w0, x_np, t_np).to(device)
            with torch.no_grad():
                out = fno_model(inp)
            rho_f = out[0, 0].cpu().numpy().T  # (nt, nx)
            w_f   = out[0, 1].cpu().numpy().T
            preds["FNO"] = {"rho": rho_f, "w": w_f, "v": w_f - P.pressure(rho_f)}

        # Numerical baselines.
        for bl in baselines:
            try:
                rho_b, w_b = _run_numerical(bl, rho0, w0, bundle, boundary)
            except Exception as e:
                print(f"  [warn] {bl} sample {i}: {e}", flush=True)
                continue
            preds[bl] = {"rho": rho_b, "w": w_b, "v": w_b - P.pressure(rho_b)}

        # Accumulate MAE + per-time errors.
        for m, pd in preds.items():
            for ch in CHANNELS:
                diff = np.abs(pd[ch] - gt[ch])
                acc[m][ch].append(float(diff.mean()))
                per_t[m][ch].append(diff.mean(axis=1))  # (nt,)

        # Qualitative figure.
        if i < n_plots:
            _save_compare_fig(
                i, rho0, gt, preds, x_np, t_np, active_methods, figs_dir, plt,
            )

        if (i + 1) % max(1, take // 10) == 0:
            print(f"  [{tag}] {i + 1}/{take}", flush=True)

    # Error-vs-time figure.
    _save_error_vs_time(per_t, t_np, active_methods, out_dir, plt)

    # Summary outputs.
    _write_csv(acc, active_methods, out_dir, tag)
    _write_txt(acc, active_methods, out_dir, tag)
    print(f"[{tag}] outputs written to {out_dir}")


# --------------------------------------------------------------------------- #
# Figure helpers
# --------------------------------------------------------------------------- #

def _save_compare_fig(idx, rho0, gt, preds, x_np, t_np, methods, figs_dir, plt):
    present = [m for m in methods if m in preds]
    for ch in CHANNELS:
        gt_arr = gt[ch]
        vmin, vmax = float(gt_arr.min()), float(gt_arr.max())
        # Column order: GT, then each method.
        col_names = ["GT"] + present
        col_arrs  = [gt_arr] + [preds[m][ch] for m in present]
        ncols = len(col_names)

        fig = plt.figure(figsize=(4 * ncols, 9), constrained_layout=True)
        gs  = fig.add_gridspec(3, ncols, height_ratios=[0.6, 3.0, 3.0])

        # Row 0: rho0 strip (shared x-axis).
        ax_ic = fig.add_subplot(gs[0, :])
        ax_ic.plot(x_np, rho0, color="black", linewidth=1.4)
        ax_ic.set_ylabel(r"$\rho_0(x)$")
        ax_ic.set_xlabel("x")
        ax_ic.set_xlim(float(x_np[0]), float(x_np[-1]))
        ax_ic.grid(True, alpha=0.3)

        # Row 1: space-time fields, shared colour scale.
        for c, (name, arr) in enumerate(zip(col_names, col_arrs)):
            ax = fig.add_subplot(gs[1, c])
            mae_str = "" if name == "GT" else f"\nMAE={_mae(arr, gt_arr):.2e}"
            im = ax.pcolormesh(x_np, t_np, arr, shading="auto",
                               cmap="jet", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}{mae_str}")
            ax.set_xlabel("x"); ax.set_ylabel("t")
            fig.colorbar(im, ax=ax, label=ch)

        # Row 2: |pred - GT|, GT column is blank, shared error scale.
        errs = [np.abs(preds[m][ch] - gt_arr) for m in present if m in preds]
        err_vmax = max(e.max() for e in errs) if errs else 1.0
        for c, name in enumerate(col_names):
            ax = fig.add_subplot(gs[2, c])
            if name == "GT":
                ax.axis("off")
                continue
            err = np.abs(preds[name][ch] - gt_arr)
            im = ax.pcolormesh(x_np, t_np, err, shading="auto",
                               cmap="magma", vmin=0, vmax=err_vmax)
            ax.set_title(f"|{name} − GT|")
            ax.set_xlabel("x"); ax.set_ylabel("t")
            fig.colorbar(im, ax=ax)

        mae_str = "  ".join(
            f"{m}={_mae(preds[m][ch], gt_arr):.3e}" for m in present if m in preds
        )
        fig.suptitle(f"sample {idx}  channel={ch}\nMAE: {mae_str}", fontsize=11)
        fig.savefig(figs_dir / f"compare_{idx:04d}_{ch}.png", dpi=150)
        plt.close(fig)


def _save_error_vs_time(per_t, t_np, methods, out_dir, plt):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    for ci, ch in enumerate(CHANNELS):
        ax = axes[ci]
        for m in methods:
            arrs = per_t[m][ch]
            if not arrs:
                continue
            curve = np.stack(arrs).mean(axis=0)
            ax.plot(t_np, curve, label=m, color=COLORS.get(m))
        ax.set_xlabel("t")
        ax.set_ylabel(f"mean |{ch} error| over x")
        ax.set_title(f"Error vs time ({ch})")
        ax.set_yscale("log")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "mae_vs_time.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Table helpers
# --------------------------------------------------------------------------- #

def _write_csv(acc, methods, out_dir, tag):
    path = out_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "method", "channel", "mae_mean", "mae_std", "n"])
        for m in methods:
            for ch in CHANNELS:
                vals = acc[m][ch]
                if not vals:
                    continue
                a = np.array(vals)
                wr.writerow([tag, m, ch, f"{a.mean():.6e}", f"{a.std():.6e}", len(vals)])
    print(f"  wrote {path}")


def _write_txt(acc, methods, out_dir, tag):
    path = out_dir / "summary.txt"
    lines = [f"ARZ orig paper eval -- dataset: {tag}\n"]
    col_w = max(len(m) for m in methods) + 2
    header = f"{'method':{col_w}}" + "".join(f"  {ch:>14}" for ch in CHANNELS)
    lines.append(header)
    lines.append("-" * len(header))
    for m in methods:
        row = f"{m:{col_w}}"
        for ch in CHANNELS:
            vals = acc[m][ch]
            if vals:
                a = np.array(vals)
                row += f"  {a.mean():.4e}±{a.std():.1e}"
            else:
                row += f"  {'--':>14}"
        lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ARZ mark-0 (orig) paper eval: HypNO vs FNO vs Godunov vs WENO5."
    )
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="HypNO_ARZ_Orig checkpoint (.pt).")
    ap.add_argument("--config", type=Path, default=None,
                    help="config.yaml for the model architecture. "
                         "Auto-located next to the checkpoint if omitted.")
    ap.add_argument("--model-section", type=str, default="hypno_arz_orig",
                    help="Config section for architecture kwargs (default hypno_arz_orig).")

    ap.add_argument("--homog-data", type=Path, default=None,
                    help="Homogeneous multi-disc dataset .npz "
                         "(e.g. arz_stratified_homog_prho.npz). "
                         "Skipped if not provided.")
    ap.add_argument("--riemann-data", type=Path, default=None,
                    help="Exact-Riemann dataset .npz "
                         "(e.g. arz_riemann_exact_prho_strat.npz). "
                         "Skipped if not provided.")

    ap.add_argument("--fno-weights", type=Path, default=None,
                    help="FNO checkpoint (.pt). FNO column omitted if not given.")
    ap.add_argument("--fno-config", type=Path, default=None,
                    help="config.yaml containing the fno_arz section. "
                         "Defaults to --config if omitted.")
    ap.add_argument("--fno-section", type=str, default="fno_arz",
                    help="Config section for FNO architecture (default fno_arz).")

    ap.add_argument("--baselines", type=str, default="godunov,weno5",
                    help="Comma-separated numerical baselines (godunov, weno5). "
                         "Use 'godunov' only for pure exact-Riemann data (WENO5 "
                         "can be unstable there).")
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-samples", type=int, default=100,
                    help="Max samples to evaluate per dataset.")
    ap.add_argument("--n-plots", type=int, default=10,
                    help="Number of qualitative figures to save per dataset.")
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="Root output directory. Subdirs homog/ and riemann/ "
                         "are created automatically.")
    args = ap.parse_args()

    if args.homog_data is None and args.riemann_data is None:
        ap.error("Provide at least one of --homog-data or --riemann-data.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Locate config for HypNO.
    config_path = args.config
    if config_path is None:
        for cand in (
            args.ckpt.parent / "config.yaml",
            args.ckpt.parent.parent / "config.yaml",
        ):
            if cand.exists():
                config_path = cand
                break

    # Load HypNO-orig.
    hypno_model, _ = load_hypno_arz_orig_from_checkpoint(
        args.ckpt, device=args.device, config_path=config_path,
        model_section=args.model_section,
    )
    print(f"[arz_orig_paper_eval] loaded HypNO-orig: {args.ckpt}")

    # Load FNO (optional).
    fno_model = None
    if args.fno_weights is not None:
        fno_cfg_path = args.fno_config or config_path
        if fno_cfg_path is None:
            ap.error("--fno-weights requires --fno-config (or --config) to find the fno section.")
        with open(fno_cfg_path, encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
        fno_sec = full_cfg.get(args.fno_section)
        if fno_sec is None:
            ap.error(f"Section '{args.fno_section}' not found in {fno_cfg_path}.")
        fno_model = _load_fno(args.fno_weights, fno_sec, args.device)
        print(f"[arz_orig_paper_eval] loaded FNO: {args.fno_weights}")

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]

    datasets: list[tuple[str, Path]] = []
    if args.homog_data is not None:
        datasets.append(("homog", args.homog_data))
    if args.riemann_data is not None:
        datasets.append(("riemann", args.riemann_data))

    for tag, data_path in datasets:
        bundle = load_arz_dataset(data_path)
        P.set_pressure_form(bundle.p_form)
        print(f"[arz_orig_paper_eval] {tag}: pressure_form={P.get_pressure_form()}  "
              f"N={bundle.rho.shape[0]}  nx={bundle.x.shape[0]}  "
              f"nt={bundle.t.shape[0]}  tau={bundle.tau}")

        eval_dataset(
            tag=tag,
            bundle=bundle,
            hypno_model=hypno_model,
            fno_model=fno_model,
            baselines=baselines,
            device=args.device,
            boundary=args.boundary,
            n_samples=args.n_samples,
            n_plots=args.n_plots,
            out_dir=args.out_dir / tag,
            plt=plt,
        )

    print(f"\n[arz_orig_paper_eval] done. Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
