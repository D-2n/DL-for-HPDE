"""Compare HypNO-ST3 vs WENO5 vs Godunov against Lax-Hopf exact solution."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from hyperbolic_pde.utils.runtime import apply_runtime_overrides, resolve_config_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset, solve_conservation_fvm
from hyperbolic_pde.data.lax_hopf import solve_lax_hopf
from hyperbolic_pde.models.hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> dict:
    base_path = ROOT / "configs" / "hyperbolic_pde.yaml"
    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path.resolve() == base_path.resolve():
        return cfg
    with path.open("r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    return _deep_update(cfg, override or {})


def split_indices(n: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_fraction * n)
    return idx[:n_train], idx[n_train:]


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred - truth).mean())


def mse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(((pred - truth) ** 2).mean())


def rel_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(((pred - truth) ** 2).sum() / ((truth ** 2).sum() + 1e-12)))


def per_t_mae(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(pred - truth).mean(axis=1)


def find_shock_positions(u_row: np.ndarray, x: np.ndarray, level: float = 0.5) -> np.ndarray:
    """Find continuous x-positions where u_row crosses `level` from low to high or
    high to low. Linear interpolation between bracketing cells gives sub-cell
    resolution.

    Returns an array of crossing positions (may be empty).
    """
    diffs = (u_row - level)
    # Bracketing pairs: sign changes between consecutive cells (strict, so exact
    # zeros at cell centers don't double-count).
    sign = np.sign(diffs)
    # Treat exact zero as +eps so we don't get spurious crossings at flat regions.
    sign = np.where(sign == 0, 1.0, sign)
    crossings = np.where(sign[:-1] != sign[1:])[0]
    if crossings.size == 0:
        return np.empty(0, dtype=np.float64)
    # Linear interpolation: u_left + alpha*(u_right - u_left) = level
    u_l = u_row[crossings]
    u_r = u_row[crossings + 1]
    x_l = x[crossings]
    x_r = x[crossings + 1]
    denom = (u_r - u_l)
    # |denom| is guaranteed > 0 because sign changed.
    alpha = (level - u_l) / denom
    return x_l + alpha * (x_r - x_l)


def shock_position_error(
    pred: np.ndarray, truth: np.ndarray, x: np.ndarray, level: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """For each timestep, match each GT shock crossing to the nearest predicted
    crossing and report |x_pred - x_gt|. Returns (per_t_mean_err, per_t_max_err)
    arrays of shape [nt]. NaN at timesteps with no GT crossings.
    """
    nt = pred.shape[0]
    mean_err = np.full(nt, np.nan, dtype=np.float64)
    max_err = np.full(nt, np.nan, dtype=np.float64)
    for k in range(nt):
        gt_x = find_shock_positions(truth[k], x, level)
        pr_x = find_shock_positions(pred[k], x, level)
        if gt_x.size == 0:
            continue
        if pr_x.size == 0:
            # GT has shocks, prediction missed all of them — flag with grid extent.
            err = np.full(gt_x.size, x[-1] - x[0])
        else:
            # Greedy nearest match: for each GT crossing find closest pred crossing.
            err = np.array([np.min(np.abs(pr_x - g)) for g in gt_x])
        mean_err[k] = err.mean()
        max_err[k] = err.max()
    return mean_err, max_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(resolve_config_path(ROOT / "configs")))
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=70, help="Number of test samples to evaluate")
    parser.add_argument("--n_plots", type=int, default=5, help="Number of samples to plot")
    parser.add_argument(
        "--diagnose_uhat", action="store_true",
        help="Save per-layer u_hat (state_probe) diagnostic alongside compare_sample plots.",
    )
    parser.add_argument(
        "--only_idx", type=str, default=None,
        help="Comma-separated dataset indices to evaluate (e.g. '1547,1753'). "
             "Bypasses --n_samples and the train/test split, runs only these.",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Override the dataset path from config.yaml (handy when the run was "
             "trained on a cluster path that doesn't exist locally).",
    )
    #parser.add_argument("--no-dg", action="store_true", help="Skip DG(1) solver")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"Using config from {run_dir / 'config.yaml'}")
    else:
        latest_path = Path("hyperbolic_pde/runs/hypno_st3/latest_run.txt")
        if latest_path.exists():
            run_dir = Path(latest_path.read_text(encoding="utf-8").strip().splitlines()[-1].strip())
            with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            print(f"Using latest run: {run_dir}")
        else:
            cfg = load_config(Path(args.config))
            run_dir = None
    cfg = apply_runtime_overrides(cfg)

    data_cfg = cfg["data"]
    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st")))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    data_path = Path(args.data_path) if args.data_path else Path(data_cfg["path"])
    print(f"Loading dataset from {data_path}")
    dataset = load_dataset(data_path)
    if args.only_idx:
        test_idx = np.array([int(s) for s in args.only_idx.split(",") if s.strip()])
        print(f"Using --only_idx override: indices={test_idx.tolist()}")
        # When only_idx is set, plot every requested sample.
        args.n_plots = max(args.n_plots, len(test_idx))
    else:
        _, test_idx = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
        test_idx = test_idx[:args.n_samples]  # noqa: already underscore

    x_np = dataset.x
    t_np = dataset.t
    x_min = float(x_np[0] - 0.5 * (x_np[1] - x_np[0]))
    x_max = float(x_np[-1] + 0.5 * (x_np[1] - x_np[0]))
    t_max = float(t_np[-1])
    nt = len(t_np)
    cfl = float(data_cfg.get("cfl", 0.3))
    boundary = str(data_cfg.get("boundary", "ghost"))

    # load HypNO-ST3
    _dhn = model_cfg.get("d_hidden_nonadj", None)
    d_hidden_nonadj = int(_dhn) if _dhn is not None else None
    model = HypNO_ST3(
        stencil_k_x=int(model_cfg.get("stencil_k_x", 3)),
        stencil_k_t=int(model_cfg.get("stencil_k_t", 2)),
        d_latent=int(model_cfg.get("d_latent", 128)),
        d_hidden=int(model_cfg.get("d_hidden", 128)),
        n_layers=int(model_cfg.get("n_layers", 6)),
        activation=str(model_cfg.get("activation", "gelu")),
        shock_delta=float(model_cfg.get("shock_delta", 0.01)),
        shock_threshold=float(model_cfg.get("shock_threshold", 0.1)),
        causal_temporal=bool(model_cfg.get("causal_temporal", True)),
        radius_x=None,
        radius_t=None,
        shock_mode=str(model_cfg.get("shock_mode", "physics")),
        weno_eps=float(model_cfg.get("weno_eps", 1e-6)),
        weno_p=float(model_cfg.get("weno_p", 2.0)),
        unified_mp=bool(model_cfg.get("unified_mp", False)),
        readout=str(model_cfg.get("readout", "relu")),
        encoder_scaling=str(model_cfg.get("encoder_scaling", "classic")),
        encoder_type=str(model_cfg.get("encoder_type", "gnn")),
        skip=bool(model_cfg.get("skip", False)),
        use_char_cone=bool(model_cfg.get("use_char_cone", False)),
        detector_path=None,
        detector_cfg={},
        d_hidden_nonadj=d_hidden_nonadj,
        include_flux=bool(model_cfg.get("include_flux", True)),
        mask_same_t_nonadj=bool(model_cfg.get("mask_same_t_nonadj", True)),
        temporal_gate_type=str(model_cfg.get("temporal_gate_type", "cfl")),
    ).to(device)

    if run_dir and (run_dir / "model_final.pt").exists():
        weights_path = run_dir / "model_final.pt"
    else:
        weights_path = Path(model_cfg["save_path"])
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded HypNO-ST3 from {weights_path}")

    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)
    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"

    # accumulate metrics
    solvers_to_run = ["HypNO-ST3", "WENO5", "Godunov"] #+ ([] if args.no_dg else ["DG(1)"])
    metrics: dict[str, list[float]] = {k: [] for k in solvers_to_run}
    per_t_errors: dict[str, list[np.ndarray]] = {k: [] for k in solvers_to_run}

    plot_dir = (run_dir / "plots_vs_numerical") if run_dir else Path("hyperbolic_pde/runs/plots/vs_numerical")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for plot_i, idx in enumerate(test_idx):
        u0_np = dataset.u0[idx]
        print(f"Sample {plot_i + 1}/{len(test_idx)} (dataset idx {idx}) ...")

        # Lax-Hopf exact solution (ground truth)
        print(f"  running Lax-Hopf...", flush=True)
        _, _, lh = solve_lax_hopf(u0_np, x_min, x_max, t_max, nt, boundary=boundary)

        # WENO5
        print(f"  running WENO5...", flush=True)
        _, _, weno = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="weno5"
        )

        # Godunov
        print(f"  running Godunov...", flush=True)
        _, _, godunov = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="godunov"
        )

        # DG(1)
        dg = None
        '''
        if not args.no_dg:
            print(f"  running DG...", flush=True)
            try:
                _, _, dg = solve_conservation_fvm(
                    u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="dg"
                )
            except RuntimeError as e:
                print(f"  DG diverged: {e} — skipping", flush=True)
                dg = np.full((nt, len(x_np)), float("nan"))
        else:
            dg = None
        '''
        # HypNO-ST3
        print(f"  running HypNO...", flush=True)
        u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device).unsqueeze(0)
        if not skip_ef:
            ef_adj, ef_nonadj = precompute_lwr_edge_features_v3(u0_t, x_grid, stencil_k_x)
            ef_adj = ef_adj.to(device)
            ef_nonadj = ef_nonadj.to(device) if ef_nonadj.numel() > 0 else None
        else:
            ef_adj, ef_nonadj = None, None
        with torch.no_grad():
            pred_t, _, _, u_hats_t = model(
                u0_t, x_grid, t_grid,
                edge_feats_adj=ef_adj, edge_feats_nonadj=ef_nonadj,
                diagnose_lifting=args.diagnose_uhat,
            )
        hypno_np = pred_t[0].cpu().numpy()
        # u_hats_t layout depends on diagnose_lifting:
        #   diagnose_lifting=False -> [MP_0, MP_1, ...]                 (one per MP layer)
        #   diagnose_lifting=True  -> [post_node, post_full, MP_0, ...] (lifting probes prepended)
        u_hats_np = [uh[0].cpu().numpy() for uh in u_hats_t] if u_hats_t else []

        pairs = [("HypNO-ST3", hypno_np), ("WENO5", weno), ("Godunov", godunov)]
        #if not args.no_dg:
        if False:
            pairs.append(("DG(1)", dg))
        for name, pred in pairs:
            metrics[name].append(mae(pred, lh))
            per_t_errors[name].append(per_t_mae(pred, lh))

        if plot_i < args.n_plots:
            vmin, vmax = float(lh.min()), float(lh.max())
            solvers = [("Lax-Hopf (GT)", lh), ("HypNO-ST3", hypno_np), ("WENO5", weno), ("Godunov", godunov)]
            #if not args.no_dg:
            if False:
                solvers.append(("DG(1)", dg))
            fig, axes = plt.subplots(2, len(solvers), figsize=(4 * len(solvers), 8), constrained_layout=True)

            for c, (name, sol) in enumerate(solvers):
                im_sol = axes[0, c].pcolormesh(x_np, t_np, sol, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
                axes[0, c].set_title(name)
                axes[0, c].set_xlabel("x")
                axes[0, c].set_ylabel("t")
                fig.colorbar(im_sol, ax=axes[0, c], label="u")

            # absolute error maps (row 1)
            err_vmax = None
            for c, (name, sol) in enumerate(solvers[1:], start=1):
                err = np.abs(sol - lh)
                if err_vmax is None:
                    err_vmax = err.max()
                im = axes[1, c].pcolormesh(x_np, t_np, err, shading="auto", cmap="magma", vmin=0, vmax=err_vmax)
                axes[1, c].set_title(f"|{name} - GT|")
                axes[1, c].set_xlabel("x")
                axes[1, c].set_ylabel("t")
                fig.colorbar(im, ax=axes[1, c])
            axes[1, 0].axis("off")

            fig.suptitle(
                f"Sample {idx} — MAE: HypNO={metrics['HypNO-ST3'][-1]:.3e}  "
                f"WENO5={metrics['WENO5'][-1]:.3e}  Godunov={metrics['Godunov'][-1]:.3e}  "
                #f"DG(1)={metrics['DG(1)'][-1]:.3e}"
            )
            fig.savefig(plot_dir / f"compare_sample_{idx}.png", dpi=150)
            plt.close(fig)

            # Continuous shock-position error: aliasing test.
            # If the salt-and-pepper at single cells is just sub-cell shock
            # mis-positioning, the *continuous* shock position should be much
            # more accurate than the cell-wise value error suggests.
            try:
                hypno_mean, hypno_max = shock_position_error(hypno_np, lh, x_np)
                weno_mean, weno_max = shock_position_error(weno, lh, x_np)
                godu_mean, godu_max = shock_position_error(godunov, lh, x_np)
                dx_grid = x_np[1] - x_np[0]
                fig_s, axes_s = plt.subplots(
                    1, 2, figsize=(11, 4), constrained_layout=True,
                )
                # Mean per-t shock-position error in units of dx
                axes_s[0].plot(t_np, hypno_mean / dx_grid, label="HypNO-ST3", color="tab:blue")
                axes_s[0].plot(t_np, weno_mean / dx_grid, label="WENO5", color="tab:orange")
                axes_s[0].plot(t_np, godu_mean / dx_grid, label="Godunov", color="tab:green")
                axes_s[0].axhline(0.5, color="k", ls="--", lw=0.7, label="half-cell")
                axes_s[0].set_xlabel("t")
                axes_s[0].set_ylabel("mean shock-position error / dx")
                axes_s[0].set_title("Continuous shock-position error (mean)")
                axes_s[0].legend()
                axes_s[0].grid(True, alpha=0.3)
                # Max
                axes_s[1].plot(t_np, hypno_max / dx_grid, label="HypNO-ST3", color="tab:blue")
                axes_s[1].plot(t_np, weno_max / dx_grid, label="WENO5", color="tab:orange")
                axes_s[1].plot(t_np, godu_max / dx_grid, label="Godunov", color="tab:green")
                axes_s[1].axhline(0.5, color="k", ls="--", lw=0.7, label="half-cell")
                axes_s[1].set_xlabel("t")
                axes_s[1].set_ylabel("max shock-position error / dx")
                axes_s[1].set_title("Continuous shock-position error (max)")
                axes_s[1].legend()
                axes_s[1].grid(True, alpha=0.3)
                fig_s.suptitle(
                    f"Sample {idx} — Continuous shock localization "
                    f"(below half-cell ⇒ pixel error is sub-cell aliasing)"
                )
                fig_s.savefig(plot_dir / f"compare_sample_{idx}_shockpos.png", dpi=150)
                plt.close(fig_s)

                # Print summary line
                def _nm(a):
                    a = a[~np.isnan(a)]
                    return float(a.mean()) if a.size else float("nan")
                print(
                    f"  shock-pos err (mean over t, units=dx): "
                    f"HypNO={_nm(hypno_mean) / dx_grid:.3f}  "
                    f"WENO={_nm(weno_mean) / dx_grid:.3f}  "
                    f"Godunov={_nm(godu_mean) / dx_grid:.3f}"
                )
            except Exception as e:
                print(f"  (shock-pos diagnostic skipped: {e})")

            if args.diagnose_uhat and u_hats_np:
                # u_hats_np layout (set in HypNO_ST3.forward):
                #   [0] post_node    — lifting node_mlp output, probed by MP-0's state_probe
                #   [1] post_full    — full lifting (node + edge + combine), same probe
                #   [2..]            — after each MP layer, that layer's own state_probe
                # Both pre-MP probes use MP-0's state_probe and are biased; the
                # spatial *pattern* (where the model has committed vs. where it's
                # still smooth) is what matters for diagnostics.
                n_cols = len(u_hats_np)
                col_titles = ["post_node", "post_full"] + [
                    f"MP_{k}" for k in range(n_cols - 2)
                ]
                err_vmax_uh = max(np.abs(uh - lh).max() for uh in u_hats_np)
                fig_d, axes_d = plt.subplots(
                    2, n_cols,
                    figsize=(3.2 * n_cols, 6.5),
                    constrained_layout=True,
                )
                if n_cols == 1:
                    axes_d = axes_d.reshape(2, 1)
                for li, uh in enumerate(u_hats_np):
                    title = col_titles[li]
                    im_top = axes_d[0, li].pcolormesh(
                        x_np, t_np, uh, shading="auto", cmap="jet", vmin=vmin, vmax=vmax,
                    )
                    axes_d[0, li].set_title(f"u_hat {title}")
                    axes_d[0, li].set_xlabel("x")
                    axes_d[0, li].set_ylabel("t")
                    fig_d.colorbar(im_top, ax=axes_d[0, li])

                    im_bot = axes_d[1, li].pcolormesh(
                        x_np, t_np, np.abs(uh - lh),
                        shading="auto", cmap="magma", vmin=0, vmax=err_vmax_uh,
                    )
                    axes_d[1, li].set_title(f"|u_hat {title} - GT|")
                    axes_d[1, li].set_xlabel("x")
                    axes_d[1, li].set_ylabel("t")
                    fig_d.colorbar(im_bot, ax=axes_d[1, li])
                fig_d.suptitle(
                    f"Sample {idx} — state_probe diagnostic (lifting + per-MP-layer)"
                )
                fig_d.savefig(plot_dir / f"compare_sample_{idx}_uhat.png", dpi=150)
                plt.close(fig_d)

    # summary metrics
    print("\n=== Summary (MAE vs Lax-Hopf) ===")
    for name in solvers_to_run:
        vals = metrics[name]
        print(f"  {name:12s}: mean={np.mean(vals):.4e}  std={np.std(vals):.4e}  min={np.min(vals):.4e}  max={np.max(vals):.4e}")

    # error vs time plot
    fig_t, ax_t = plt.subplots(figsize=(8, 4), constrained_layout=True)
    colors = {"HypNO-ST3": "tab:blue", "WENO5": "tab:orange", "Godunov": "tab:green", "DG(1)": "tab:red"}
    for name in solvers_to_run:
        mean_curve = np.stack(per_t_errors[name]).mean(axis=0)
        ax_t.plot(t_np, mean_curve, label=name, color=colors[name])
    ax_t.set_xlabel("t")
    ax_t.set_ylabel("MAE vs Lax-Hopf")
    ax_t.set_title("Error vs time")
    ax_t.set_yscale("log")
    ax_t.legend()
    ax_t.grid(True, alpha=0.3)
    fig_t.savefig(plot_dir / "error_vs_time.png", dpi=150)
    plt.close(fig_t)

    # save metrics to file
    metrics_path = plot_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("MAE vs Lax-Hopf exact solution\n")
        f.write(f"N samples: {len(test_idx)}\n\n")
        for name in solvers_to_run:
            vals = metrics[name]
            f.write(f"{name}:\n")
            f.write(f"  mean = {np.mean(vals):.6e}\n")
            f.write(f"  std  = {np.std(vals):.6e}\n")
            f.write(f"  min  = {np.min(vals):.6e}\n")
            f.write(f"  max  = {np.max(vals):.6e}\n\n")

    print(f"\nSaved plots to {plot_dir}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
