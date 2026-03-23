from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset, solve_conservation_fvm
from hyperbolic_pde.cfl import annotate_cfl, print_cfl_report
from hyperbolic_pde.models.fluxgnn import FluxGNN1D


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


def mse_mae_rel(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    err = pred - truth
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.linalg.norm(truth))
    rel = float(np.linalg.norm(err)) if denom == 0.0 else float(np.linalg.norm(err) / denom)
    return mse, mae, rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FluxGNN time-jump prediction.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--t_jump",
        type=float,
        default=2.0,
        help="Target time for the jump evaluation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate/plot.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    flux_cfg = cfg["fluxgnn"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    test_path = Path(data_cfg.get("test_path", "hyperbolic_pde/data/hyperbolic_test_superres.npz"))
    if test_path.exists():
        data = np.load(test_path)
        x = data["x_1x"]
        t_ref = data["t_1x"]
        u0_all = data["u0_1x"]
    else:
        dataset = load_dataset(Path(data_cfg["path"]))
        x = dataset.x
        t_ref = dataset.t
        u0_all = dataset.u0

    dx = float(x[1] - x[0])
    dt_ref = float(t_ref[1] - t_ref[0])
    n_steps = int(round(args.t_jump / dt_ref)) + 1
    t_pred = dt_ref * (n_steps - 1)
    t_eval = np.linspace(0.0, t_pred, n_steps, dtype=np.float32)
    if abs(t_pred - args.t_jump) > 1e-6:
        print(f"[FluxGNN Jump] Using t={t_pred:.6f} (closest to requested {args.t_jump:.6f}).")

    num_samples = args.num_samples
    if num_samples is None:
        num_samples = int(data_cfg.get("compare_samples", 5))
    num_samples = max(1, min(num_samples, u0_all.shape[0]))

    model = FluxGNN1D(
        hidden=int(flux_cfg["hidden"]),
        layers=int(flux_cfg["layers"]),
        activation=str(flux_cfg.get("activation", "gelu")),
        latent_dim=flux_cfg.get("latent_dim"),
        flux_hidden=flux_cfg.get("flux_hidden"),
        use_base_flux=bool(flux_cfg.get("use_base_flux", False)),
        base_flux_weight=float(flux_cfg.get("base_flux_weight", 0.0)),
        flux_scale=float(flux_cfg.get("flux_scale", 0.25)),
    ).to(device)
    model.load_state_dict(torch.load(Path(flux_cfg["save_path"]), map_location=device))
    model.eval()

    plot_dir = Path("hyperbolic_pde/runs/plots/fluxgnn_time_jump")
    plot_dir.mkdir(parents=True, exist_ok=True)

    boundary = str(data_cfg.get("boundary", "ghost"))
    cfl = float(data_cfg.get("cfl", 0.1))
    x_min = float(data_cfg.get("x_min", -1.0))
    x_max = float(data_cfg.get("x_max", 1.0))
    cfl_metrics = print_cfl_report(data_cfg, x, t_eval, t_final=t_pred)

    metrics = {"mse": [], "mae": [], "rel_l2": []}

    with torch.no_grad():
        for i in range(num_samples):
            u0 = u0_all[i]
            u0_t = torch.tensor(u0, dtype=torch.float32, device=device).unsqueeze(0)
            pred_full = model(u0_t, dt_ref, dx, n_steps, boundary)[0]
            pred = pred_full[-1].detach().cpu().numpy()

            x_truth, t_truth, truth_hist = solve_conservation_fvm(
                u0=u0,
                x_min=x_min,
                x_max=x_max,
                t_max=t_pred,
                nt_out=n_steps,
                cfl=cfl,
                boundary=boundary,
            )
            truth = truth_hist[-1]

            mse, mae, rel = mse_mae_rel(pred, truth)
            metrics["mse"].append(mse)
            metrics["mae"].append(mae)
            metrics["rel_l2"].append(rel)

            fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
            annotate_cfl(fig, cfl_metrics)
            axes[0].plot(x, u0, color="black", linewidth=1.5)
            axes[0].set_title("Initial condition")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("u")

            axes[1].plot(x, truth, label="Godunov (truth)", color="tab:blue", linewidth=1.5)
            axes[1].plot(x, pred, label="FluxGNN (pred)", color="tab:orange", linewidth=1.5, linestyle="--")
            axes[1].set_title(f"Time jump at t={t_pred:.3f}")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("u")
            axes[1].legend()

            axes[2].plot(x, np.abs(pred - truth), color="tab:red", linewidth=1.5)
            axes[2].set_title("Absolute error")
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("|error|")

            out_path = plot_dir / f"fluxgnn_time_jump_{i}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            # Heatmap-style plots (prediction vs truth vs error)
            pred_hist_np = pred_full.detach().cpu().numpy()
            err_hist = np.abs(pred_hist_np - truth_hist)
            vmin = float(np.min(truth_hist))
            vmax = float(np.max(truth_hist))

            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
            annotate_cfl(fig, cfl_metrics)
            im0 = axes[0].pcolormesh(x_truth, t_truth, pred_hist_np, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
            axes[0].set_title("FluxGNN prediction")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("t")
            fig.colorbar(im0, ax=axes[0])

            im1 = axes[1].pcolormesh(x_truth, t_truth, truth_hist, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
            axes[1].set_title("Godunov FVM truth")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("t")
            fig.colorbar(im1, ax=axes[1])

            im2 = axes[2].pcolormesh(x_truth, t_truth, err_hist, shading="auto", cmap="magma")
            axes[2].set_title("Absolute error")
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("t")
            fig.colorbar(im2, ax=axes[2])

            out_path = plot_dir / f"fluxgnn_time_jump_heatmap_{i}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

    print(f"[FluxGNN Jump] Mean MSE: {np.mean(metrics['mse']):.6e}")
    print(f"[FluxGNN Jump] Mean MAE: {np.mean(metrics['mae']):.6e}")
    print(f"[FluxGNN Jump] Mean Rel L2: {np.mean(metrics['rel_l2']):.6e}")
    print(f"[FluxGNN Jump] Saved {num_samples} plots to {plot_dir}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    annotate_cfl(fig, cfl_metrics)
    labels = ["MSE", "MAE", "Rel L2"]
    keys = ["mse", "mae", "rel_l2"]
    for i, key in enumerate(keys):
        axes[i].boxplot(metrics[key], labels=[key], showfliers=False)
        axes[i].set_yscale("log")
        axes[i].set_title(labels[i])
    fig.savefig(plot_dir / "fluxgnn_time_jump_metrics_boxplot.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
