"""Compare HypNO-ST3-CharCone vs WENO5 vs Godunov against Lax-Hopf exact solution."""
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
from hyperbolic_pde.models.legacy.hypno_st3_charcone import HypNO_ST3, precompute_lwr_edge_features_v3


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(resolve_config_path(ROOT / "configs")))
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=10, help="Number of test samples to evaluate")
    parser.add_argument("--n_plots", type=int, default=3, help="Number of samples to plot")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"Using config from {run_dir / 'config.yaml'}")
    else:
        latest_path = Path("hyperbolic_pde/runs/hypno_st3_charcone/latest_run.txt")
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
    model_cfg = cfg.get("hypno_st3_charcone", cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st"))))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    dataset = load_dataset(Path(data_cfg["path"]))
    _, test_idx = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
    test_idx = test_idx[:args.n_samples]

    x_np = dataset.x
    t_np = dataset.t
    x_min = float(x_np[0] - 0.5 * (x_np[1] - x_np[0]))
    x_max = float(x_np[-1] + 0.5 * (x_np[1] - x_np[0]))
    t_max = float(t_np[-1])
    nt = len(t_np)
    cfl = float(data_cfg.get("cfl", 0.3))
    boundary = str(data_cfg.get("boundary", "ghost"))

    # load HypNO-ST3-CharCone
    _dhn = model_cfg.get("d_hidden_nonadj", None)
    d_hidden_nonadj = int(_dhn) if _dhn is not None else None
    model = HypNO_ST3(
        stencil_k_x=int(model_cfg.get("stencil_k_x", 3)),
        stencil_k_t=int(model_cfg.get("stencil_k_t", 2)),
        d_latent=int(model_cfg.get("d_latent", 128)),
        d_hidden=int(model_cfg.get("d_hidden", 128)),
        n_layers=int(model_cfg.get("n_layers", 6)),
        activation=str(model_cfg.get("activation", "gelu")),
        causal_temporal=bool(model_cfg.get("causal_temporal", True)),
        radius_x=None,
        radius_t=None,
        readout=str(model_cfg.get("readout", "gelu")),
        encoder_scaling=str(model_cfg.get("encoder_scaling", "physics")),
        encoder_type=str(model_cfg.get("encoder_type", "gnn")),
        skip=bool(model_cfg.get("skip", True)),
        use_char_cone=bool(model_cfg.get("use_char_cone", False)),
        d_hidden_nonadj=d_hidden_nonadj,
        use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
        mask_same_t_nonadj=bool(model_cfg.get("mask_same_t_nonadj", True)),
        temporal_gate_type=str(model_cfg.get("temporal_gate_type", "cfl")),
        detector_path=None,
        detector_cfg={},
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
    print(f"Loaded HypNO-ST3-CharCone from {weights_path}")

    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)
    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"

    solvers_to_run = ["HypNO-ST3-CC", "WENO5", "Godunov"]
    metrics: dict[str, list[float]] = {k: [] for k in solvers_to_run}
    per_t_errors: dict[str, list[np.ndarray]] = {k: [] for k in solvers_to_run}

    plot_dir = (run_dir / "plots_vs_numerical") if run_dir else Path("hyperbolic_pde/runs/plots/vs_numerical_charcone")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for plot_i, idx in enumerate(test_idx):
        u0_np = dataset.u0[idx]
        print(f"Sample {plot_i + 1}/{len(test_idx)} (dataset idx {idx}) ...")

        print(f"  running Lax-Hopf...", flush=True)
        _, _, lh = solve_lax_hopf(u0_np, x_min, x_max, t_max, nt, boundary=boundary)

        print(f"  running WENO5...", flush=True)
        _, _, weno = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="weno5"
        )

        print(f"  running Godunov...", flush=True)
        _, _, godunov = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="godunov"
        )

        print(f"  running HypNO-CC...", flush=True)
        u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device).unsqueeze(0)
        if not skip_ef:
            ef_adj, ef_nonadj = precompute_lwr_edge_features_v3(u0_t, x_grid, stencil_k_x)
            ef_adj = ef_adj.to(device)
            ef_nonadj = ef_nonadj.to(device) if ef_nonadj.numel() > 0 else None
        else:
            ef_adj, ef_nonadj = None, None
        with torch.no_grad():
            pred_t, _, _, _ = model(u0_t, x_grid, t_grid, edge_feats_adj=ef_adj, edge_feats_nonadj=ef_nonadj)
        hypno_np = pred_t[0].cpu().numpy()

        pairs = [("HypNO-ST3-CC", hypno_np), ("WENO5", weno), ("Godunov", godunov)]
        for name, pred in pairs:
            metrics[name].append(mae(pred, lh))
            per_t_errors[name].append(per_t_mae(pred, lh))

        if plot_i < args.n_plots:
            vmin, vmax = float(lh.min()), float(lh.max())
            solvers = [("Lax-Hopf (GT)", lh), ("HypNO-ST3-CC", hypno_np), ("WENO5", weno), ("Godunov", godunov)]
            fig, axes = plt.subplots(2, len(solvers), figsize=(4 * len(solvers), 8), constrained_layout=True)

            for c, (name, sol) in enumerate(solvers):
                axes[0, c].pcolormesh(x_np, t_np, sol, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
                axes[0, c].set_title(name)
                axes[0, c].set_xlabel("x")
                axes[0, c].set_ylabel("t")

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
                f"Sample {idx} — MAE: HypNO-CC={metrics['HypNO-ST3-CC'][-1]:.3e}  "
                f"WENO5={metrics['WENO5'][-1]:.3e}  Godunov={metrics['Godunov'][-1]:.3e}"
            )
            fig.savefig(plot_dir / f"compare_sample_{idx}.png", dpi=150)
            plt.close(fig)

    print("\n=== Summary (MAE vs Lax-Hopf) ===")
    for name in solvers_to_run:
        vals = metrics[name]
        print(f"  {name:14s}: mean={np.mean(vals):.4e}  std={np.std(vals):.4e}  min={np.min(vals):.4e}  max={np.max(vals):.4e}")

    fig_t, ax_t = plt.subplots(figsize=(8, 4), constrained_layout=True)
    colors = {"HypNO-ST3-CC": "tab:blue", "WENO5": "tab:orange", "Godunov": "tab:green"}
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
