"""Eval the 2-channel ARZ FNO baseline vs ground truth (and optional numerical
baselines), on the SAME held-out test split it was trained against.

Mirror of eval_vs_numerical_arz.py but for the FNO2d baseline (train_fno_arz.py):
the FNO predicts (rho, w) directly; v is recovered as v = w - p(rho). Ground truth
is the dataset's stored (rho, w) history (exact/FV solver from datagen). Optionally
also runs WENO5 / Godunov for context.

Held-out test split: the SAME split_indices(train_fraction, seed) as
train_fno_arz.py / train_hypno_arz.py, so the evaluated samples were NOT seen in
training. Per-channel rho/w/v MAE, stratified by (ic_type, num_segments), CSV +
per-sample figures.

Usage (local):
    python -m hyperbolic_pde.arz.eval_fno_arz_vs_numerical \
        --config hyperbolic_pde/configs/hyperbolic_pde_arz_local.yaml \
        --data-section arz_general_local --fno-section fno_arz \
        --samples 60 --baselines godunov \
        --out hyperbolic_pde/arz/runs/fno_arz_local/eval.csv \
        --figures hyperbolic_pde/arz/runs/fno_arz_local/eval_figs
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

from hyperbolic_pde.utils.runtime import apply_runtime_overrides
from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.eval_vs_numerical_arz import solve_arz_weno5
from hyperbolic_pde.models.competitive_architectures.fno import FNO2d


# --------------------------------------------------------------------------- #
# config + split helpers (identical to train_fno_arz.py)
# --------------------------------------------------------------------------- #
def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> dict:
    base_path = ROOT / "configs" / "hyperbolic_pde_arz.yaml"
    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path.resolve() == base_path.resolve():
        return cfg
    with path.open("r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    return _deep_update(cfg, override or {})


def split_indices(n: int, train_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_fraction * n)
    return idx[:n_train], idx[n_train:]


def make_fno_input(rho0_np, w0_np, x_np, t_np) -> torch.Tensor:
    """Build the (1, 4, nx, nt) FNO input, mirroring ArzFNODataset.__getitem__."""
    x_t = torch.tensor(x_np, dtype=torch.float32)
    t_t = torch.tensor(t_np, dtype=torch.float32)
    rho0_t = torch.tensor(rho0_np, dtype=torch.float32)
    w0_t = torch.tensor(w0_np, dtype=torch.float32)
    X, T = torch.meshgrid(x_t, t_t, indexing="ij")
    nt = t_t.numel()
    rho0_grid = rho0_t.unsqueeze(1).repeat(1, nt)
    w0_grid = w0_t.unsqueeze(1).repeat(1, nt)
    inp = torch.stack([X, T, rho0_grid, w0_grid], dim=0).unsqueeze(0)  # (1, 4, nx, nt)
    return inp


def _run_baseline(name, rho0, w0, bundle, tau, boundary):
    """Same baseline runners as eval_vs_numerical_arz._run_baseline."""
    nt = bundle.t.shape[0]
    xb = np.asarray(bundle.x, dtype=np.float64)
    dx = float(xb[1] - xb[0])
    x_min = float(xb[0] - 0.5 * dx)
    x_max = float(xb[-1] + 0.5 * dx)
    t_max = float(bundle.t.max())
    if name in ("godunov", "hll"):
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval ARZ FNO baseline vs ground truth.")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs" / "hyperbolic_pde_arz_local.yaml")
    ap.add_argument("--data-section", type=str, default="arz_general_local")
    ap.add_argument("--fno-section", type=str, default="fno_arz")
    ap.add_argument("--weights", type=Path, default=None,
                    help="FNO checkpoint (.pt). Defaults to the fno-section save_path.")
    ap.add_argument("--baselines", type=str, default="",
                    help="Comma-separated extra numerical baselines for context "
                         "(e.g. 'godunov' or 'weno5,godunov'). Empty = FNO vs GT only.")
    ap.add_argument("--samples", type=int, default=60,
                    help="Cap on number of held-out test samples to evaluate.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", type=Path, default=None)
    ap.add_argument("--n-plots", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_runtime_overrides(cfg)
    data_cfg = cfg[args.data_section]
    fno_cfg = cfg[args.fno_section]
    seed = int(cfg.get("seed", 42))

    bundle = load_arz_dataset(Path(data_cfg["path"]))
    P.set_pressure_form(str(data_cfg.get("pressure_form", bundle.p_form)) if
                        str(data_cfg.get("pressure_form", bundle.p_form)) == bundle.p_form
                        else bundle.p_form)
    print(f"[eval_fno_arz] pressure_form={P.get_pressure_form()}  "
          f"N={bundle.rho.shape[0]}  nx={bundle.x.shape[0]}  nt={bundle.t.shape[0]}  "
          f"tau={bundle.tau}")

    # Held-out TEST split (same cap + split as training).
    N = bundle.rho.shape[0]
    if "num_samples" in data_cfg:
        N = min(N, int(data_cfg["num_samples"]))
    _, test_idx = split_indices(N, float(data_cfg["train_fraction"]), seed)
    test_idx = test_idx[: args.samples]
    print(f"[eval_fno_arz] evaluating {len(test_idx)} held-out test samples")

    # Load FNO.
    model = FNO2d(
        in_channels=4, out_channels=2,
        width=int(fno_cfg["width"]),
        modes_x=int(fno_cfg["modes_x"]),
        modes_t=int(fno_cfg["modes_t"]),
        layers=int(fno_cfg["layers"]),
    ).to(args.device)
    weights = args.weights if args.weights is not None else Path(fno_cfg["save_path"])
    state = torch.load(weights, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[eval_fno_arz] loaded FNO weights from {weights}")

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    methods_in_order = ["FNO"] + baselines

    x_np = bundle.x.astype(np.float64)
    t_np = bundle.t.astype(np.float64)
    tau = float(bundle.tau)
    baseline_tau = tau if np.isfinite(tau) else 1e6

    plot_dir = None
    per_t_errors: dict = {}
    if args.figures is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_dir = Path(args.figures)
        plot_dir.mkdir(parents=True, exist_ok=True)
        for m in methods_in_order:
            per_t_errors[m] = {"rho": [], "w": [], "v": []}
        print(f"[eval_fno_arz] writing figures to {plot_dir}")

    buckets: dict = {}
    for plot_i, idx in enumerate(test_idx):
        fam = str(bundle.ic_type[idx])
        seg = int(bundle.num_segments[idx])
        key = (fam, seg)
        if key not in buckets:
            buckets[key] = {m: {"rho": [], "w": [], "v": []} for m in methods_in_order}

        rho_gt = bundle.rho[idx].astype(np.float64)
        w_gt = bundle.w[idx].astype(np.float64)
        v_gt = w_gt - P.pressure(rho_gt)
        rho0 = bundle.rho0[idx].astype(np.float64)
        w0 = bundle.w0[idx].astype(np.float64)

        # FNO: predict (rho, w) -> (1, 2, nx, nt); transpose each channel to (nt, nx).
        inp = make_fno_input(rho0, w0, x_np, t_np).to(args.device)
        with torch.no_grad():
            pred = model(inp)
        rho_f = pred[0, 0].cpu().numpy().T
        w_f = pred[0, 1].cpu().numpy().T
        v_f = w_f - P.pressure(rho_f)
        buckets[key]["FNO"]["rho"].append(float(np.mean(np.abs(rho_f - rho_gt))))
        buckets[key]["FNO"]["w"].append(float(np.mean(np.abs(w_f - w_gt))))
        buckets[key]["FNO"]["v"].append(float(np.mean(np.abs(v_f - v_gt))))

        sample_preds = {"FNO": (rho_f, w_f, v_f)} if plot_dir is not None else None

        for m in baselines:
            try:
                rho_b, w_b = _run_baseline(m, rho0, w0, bundle, baseline_tau, args.boundary)
            except Exception as e:
                print(f"  [warn] {m} failed on sample {idx}: {e}", flush=True)
                continue
            v_b = w_b - P.pressure(rho_b)
            buckets[key][m]["rho"].append(float(np.mean(np.abs(rho_b - rho_gt))))
            buckets[key][m]["w"].append(float(np.mean(np.abs(w_b - w_gt))))
            buckets[key][m]["v"].append(float(np.mean(np.abs(v_b - v_gt))))
            if sample_preds is not None:
                sample_preds[m] = (rho_b, w_b, v_b)

        if plot_dir is not None:
            for m, (rm, wm, vm) in sample_preds.items():
                per_t_errors[m]["rho"].append(np.mean(np.abs(rm - rho_gt), axis=1))
                per_t_errors[m]["w"].append(np.mean(np.abs(wm - w_gt), axis=1))
                per_t_errors[m]["v"].append(np.mean(np.abs(vm - v_gt), axis=1))
            if plot_i < args.n_plots:
                for ch_name, gt_arr, pred_dict in (
                    ("rho", rho_gt, {m: sample_preds[m][0] for m in sample_preds}),
                    ("v", v_gt, {m: sample_preds[m][2] for m in sample_preds}),
                    ("w", w_gt, {m: sample_preds[m][1] for m in sample_preds}),
                ):
                    methods_present = ["GT"] + [m for m in methods_in_order if m in pred_dict]
                    n_cols = len(methods_present)
                    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8),
                                             constrained_layout=True, squeeze=False)
                    vmin = float(gt_arr.min()); vmax = float(gt_arr.max())
                    mae_per_method = {
                        name: float(np.mean(np.abs(pred_dict[name] - gt_arr)))
                        for name in methods_present[1:]
                    }
                    err_vmax = None
                    for c, name in enumerate(methods_present):
                        arr = gt_arr if name == "GT" else pred_dict[name]
                        im = axes[0, c].pcolormesh(x_np, t_np, arr, shading="auto",
                                                   cmap="jet", vmin=vmin, vmax=vmax)
                        title = name if name == "GT" else f"{name}\nMAE={mae_per_method[name]:.2e}"
                        axes[0, c].set_title(title)
                        axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
                        fig.colorbar(im, ax=axes[0, c], label=ch_name)
                        if name != "GT":
                            err = np.abs(arr - gt_arr)
                            err_vmax = err.max() if err_vmax is None else max(err_vmax, err.max())
                    axes[1, 0].axis("off")
                    for c, name in enumerate(methods_present[1:], start=1):
                        err = np.abs(pred_dict[name] - gt_arr)
                        im = axes[1, c].pcolormesh(x_np, t_np, err, shading="auto",
                                                   cmap="magma", vmin=0, vmax=err_vmax)
                        axes[1, c].set_title(f"|{name} - GT|")
                        axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
                        fig.colorbar(im, ax=axes[1, c])
                    mae_strs = [f"{m}={mae_per_method[m]:.3e}"
                                for m in methods_in_order if m in mae_per_method]
                    fig.suptitle(f"Sample {idx}  channel={ch_name}  fam={fam} seg={seg}  "
                                 f"MAE: " + "  ".join(mae_strs))
                    fig.savefig(plot_dir / f"compare_sample_{idx}_{ch_name}.png", dpi=150)
                    plt.close(fig)

        if (plot_i + 1) % max(1, len(test_idx) // 10) == 0:
            print(f"  {plot_i + 1}/{len(test_idx)}", flush=True)

    # CSV.
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

    # Error-vs-time plot.
    if plot_dir is not None and per_t_errors:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(20, 4.5), constrained_layout=True)
        colors = {"FNO": "tab:blue", "weno5": "tab:orange",
                  "godunov": "tab:green", "hll": "tab:red"}
        for ci, ch_name in enumerate(("rho", "v", "w")):
            for m in methods_in_order:
                arrs = per_t_errors[m][ch_name]
                if not arrs:
                    continue
                mean_curve = np.stack(arrs).mean(axis=0)
                axes[ci].plot(t_np, mean_curve, label=m, color=colors.get(m, None))
            axes[ci].set_xlabel("t")
            axes[ci].set_ylabel(f"mean |{ch_name}_pred - {ch_name}_gt| over x")
            axes[ci].set_title(f"Error vs time ({ch_name})")
            axes[ci].set_yscale("log")
            axes[ci].legend(); axes[ci].grid(True, alpha=0.3)
        fig.savefig(plot_dir / "error_vs_time.png", dpi=150)
        plt.close(fig)

    # Overall (pooled across all samples) per-channel MAE for the headline number.
    print(f"\n[eval_fno_arz] wrote {args.out}  (N={len(test_idx)} samples)")
    pooled = {m: {"rho": [], "w": [], "v": []} for m in methods_in_order}
    for bymethod in buckets.values():
        for m, chans in bymethod.items():
            for ch in ("rho", "w", "v"):
                pooled[m][ch].extend(chans[ch])
    print("  OVERALL MAE (pooled):")
    for m in methods_in_order:
        cells = []
        for ch in ("rho", "w", "v"):
            arr = np.array(pooled[m][ch])
            if arr.size:
                cells.append(f"{ch} {arr.mean():.3e}")
        print(f"    {m:8s}  " + "  ".join(cells))
    print("  by (family, segments):")
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
