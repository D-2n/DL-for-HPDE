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

from hyperbolic_pde.models.fluxgnn import FluxGNN1D
from hyperbolic_pde.cfl import annotate_cfl, print_cfl_report


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


def total_variation_map(u: np.ndarray) -> np.ndarray:
    tv = np.zeros_like(u)
    tv[:-1, :] += np.abs(u[1:, :] - u[:-1, :])
    tv[:, :-1] += np.abs(u[:, 1:] - u[:, :-1])
    return tv


def mse_mae_rel(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float, float]:
    err = pred - truth
    mse = torch.mean(err.pow(2)).item()
    mae = torch.mean(err.abs()).item()
    denom = torch.norm(truth).item()
    rel = torch.norm(err).item() if denom == 0.0 else torch.norm(err).item() / denom
    return mse, mae, rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FluxGNN on 1x and 2x test data.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    flux_cfg = cfg["fluxgnn"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    test_path = Path(data_cfg.get("test_path", "hyperbolic_pde/data/hyperbolic_test_superres.npz"))
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}. Run generate_test_superres.py first."
        )
    data = np.load(test_path)

    x1 = data["x_1x"]
    t1 = data["t_1x"]
    u1 = data["u_1x"]
    u0_1 = data["u0_1x"]
    x2 = data["x_2x"]
    t2 = data["t_2x"]
    u2 = data["u_2x"]
    u0_2 = data["u0_2x"]
    cfl_1x = print_cfl_report(data_cfg, x1, t1, label="1x")
    cfl_2x = print_cfl_report(data_cfg, x2, t2, label="2x")

    model = FluxGNN1D(
        hidden=int(flux_cfg["hidden"]),
        layers=int(flux_cfg["layers"]),
        activation=str(flux_cfg.get("activation", "gelu")),
        latent_dim=flux_cfg.get("latent_dim"),
        flux_hidden=flux_cfg.get("flux_hidden"),
        use_base_flux=bool(flux_cfg.get("use_base_flux", True)),
        base_flux_weight=float(flux_cfg.get("base_flux_weight", 0.5)),
        flux_scale=float(flux_cfg.get("flux_scale", 0.25)),
    ).to(device)
    model.load_state_dict(torch.load(Path(flux_cfg["save_path"]), map_location=device))
    model.eval()

    plot_dir = Path("hyperbolic_pde/runs/plots/fluxgnn_superres")
    plot_dir.mkdir(parents=True, exist_ok=True)

    boundary = str(data_cfg.get("boundary", "ghost"))

    dx1 = float(x1[1] - x1[0])
    dt1 = float(t1[1] - t1[0])
    n_steps_1 = int(t1.shape[0])
    dx2 = float(x2[1] - x2[0])
    dt2 = float(t2[1] - t2[0])
    n_steps_2 = int(t2.shape[0])

    metrics_1 = {"mse": [], "mae": [], "rel_l2": []}
    metrics_2 = {"mse": [], "mae": [], "rel_l2": []}

    with torch.no_grad():
        for i in range(min(u1.shape[0], u2.shape[0])):
            u0_1_t = torch.tensor(u0_1[i], dtype=torch.float32, device=device).unsqueeze(0)
            u0_2_t = torch.tensor(u0_2[i], dtype=torch.float32, device=device).unsqueeze(0)
            pred_1 = model(u0_1_t, dt1, dx1, n_steps_1, boundary)[0]
            pred_2 = model(u0_2_t, dt2, dx2, n_steps_2, boundary)[0]
            truth_1 = torch.tensor(u1[i], dtype=torch.float32, device=device)
            truth_2 = torch.tensor(u2[i], dtype=torch.float32, device=device)

            mse, mae, rel = mse_mae_rel(pred_1, truth_1)
            metrics_1["mse"].append(mse)
            metrics_1["mae"].append(mae)
            metrics_1["rel_l2"].append(rel)

            mse, mae, rel = mse_mae_rel(pred_2, truth_2)
            metrics_2["mse"].append(mse)
            metrics_2["mae"].append(mae)
            metrics_2["rel_l2"].append(rel)

    mean_1 = {k: float(np.mean(v)) for k, v in metrics_1.items()}
    mean_2 = {k: float(np.mean(v)) for k, v in metrics_2.items()}

    print(f"[FluxGNN SuperRes] Test MSE 1x: {mean_1['mse']:.6e}")
    print(f"[FluxGNN SuperRes] Test MSE 2x: {mean_2['mse']:.6e}")

    # Metrics plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    annotate_cfl(fig, cfl_1x, prefix="1x", y=0.02)
    annotate_cfl(fig, cfl_2x, prefix="2x", y=0.005)
    metrics_order = ["mse", "mae", "rel_l2"]
    labels = ["MSE", "MAE", "Rel L2"]
    for i, key in enumerate(metrics_order):
        axes[i].bar(["1x", "2x"], [mean_1[key], mean_2[key]])
        axes[i].set_title(labels[i])
        axes[i].set_yscale("log")
        axes[i].set_ylabel(labels[i])
    fig.savefig(plot_dir / "fluxgnn_superres_metrics.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    annotate_cfl(fig, cfl_1x, prefix="1x", y=0.02)
    annotate_cfl(fig, cfl_2x, prefix="2x", y=0.005)
    for i, key in enumerate(metrics_order):
        axes[i].boxplot([metrics_1[key], metrics_2[key]], labels=["1x", "2x"], showfliers=False)
        axes[i].set_title(f"{labels[i]} distribution")
        axes[i].set_yscale("log")
    fig.savefig(plot_dir / "fluxgnn_superres_metrics_boxplot.png", dpi=150)
    plt.close(fig)

    # Sample plots
    max_plots = int(flux_cfg.get("eval_plots", 3))
    plots_made = 0
    with torch.no_grad():
        for idx in range(min(max_plots, u1.shape[0], u2.shape[0])):
            u0_1_t = torch.tensor(u0_1[idx], dtype=torch.float32, device=device).unsqueeze(0)
            u0_2_t = torch.tensor(u0_2[idx], dtype=torch.float32, device=device).unsqueeze(0)
            pred_1 = model(u0_1_t, dt1, dx1, n_steps_1, boundary)[0]
            pred_2 = model(u0_2_t, dt2, dx2, n_steps_2, boundary)[0]

            pred_1_np = pred_1.detach().cpu().numpy()
            pred_2_np = pred_2.detach().cpu().numpy()
            truth_1_np = u1[idx]
            truth_2_np = u2[idx]
            err_1 = np.abs(pred_1_np - truth_1_np)
            err_2 = np.abs(pred_2_np - truth_2_np)
            tv_1 = total_variation_map(pred_1_np)
            tv_2 = total_variation_map(pred_2_np)

            vmin_1 = float(np.min(truth_1_np))
            vmax_1 = float(np.max(truth_1_np))
            vmin_2 = float(np.min(truth_2_np))
            vmax_2 = float(np.max(truth_2_np))

            fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
            annotate_cfl(fig, cfl_1x, prefix="1x", y=0.02)
            annotate_cfl(fig, cfl_2x, prefix="2x", y=0.005)
            im00 = axes[0, 0].pcolormesh(
                x1, t1, pred_1_np.T, shading="auto", cmap="jet", vmin=vmin_1, vmax=vmax_1
            )
            axes[0, 0].set_title("FluxGNN prediction (1x)")
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("t")
            fig.colorbar(im00, ax=axes[0, 0])

            im01 = axes[0, 1].pcolormesh(
                x1, t1, truth_1_np.T, shading="auto", cmap="jet", vmin=vmin_1, vmax=vmax_1
            )
            axes[0, 1].set_title("Godunov FVM truth (1x)")
            axes[0, 1].set_xlabel("x")
            axes[0, 1].set_ylabel("t")
            fig.colorbar(im01, ax=axes[0, 1])

            im02 = axes[0, 2].pcolormesh(x1, t1, err_1.T, shading="auto", cmap="magma")
            axes[0, 2].set_title("Abs error (1x)")
            axes[0, 2].set_xlabel("x")
            axes[0, 2].set_ylabel("t")
            fig.colorbar(im02, ax=axes[0, 2])

            im03 = axes[0, 3].pcolormesh(x1, t1, tv_1.T, shading="auto", cmap="viridis")
            axes[0, 3].set_title("Total variation (1x)")
            axes[0, 3].set_xlabel("x")
            axes[0, 3].set_ylabel("t")
            fig.colorbar(im03, ax=axes[0, 3])

            im10 = axes[1, 0].pcolormesh(
                x2, t2, pred_2_np.T, shading="auto", cmap="jet", vmin=vmin_2, vmax=vmax_2
            )
            axes[1, 0].set_title("FluxGNN prediction (2x)")
            axes[1, 0].set_xlabel("x")
            axes[1, 0].set_ylabel("t")
            fig.colorbar(im10, ax=axes[1, 0])

            im11 = axes[1, 1].pcolormesh(
                x2, t2, truth_2_np.T, shading="auto", cmap="jet", vmin=vmin_2, vmax=vmax_2
            )
            axes[1, 1].set_title("Godunov FVM truth (2x)")
            axes[1, 1].set_xlabel("x")
            axes[1, 1].set_ylabel("t")
            fig.colorbar(im11, ax=axes[1, 1])

            im12 = axes[1, 2].pcolormesh(x2, t2, err_2.T, shading="auto", cmap="magma")
            axes[1, 2].set_title("Abs error (2x)")
            axes[1, 2].set_xlabel("x")
            axes[1, 2].set_ylabel("t")
            fig.colorbar(im12, ax=axes[1, 2])

            im13 = axes[1, 3].pcolormesh(x2, t2, tv_2.T, shading="auto", cmap="viridis")
            axes[1, 3].set_title("Total variation (2x)")
            axes[1, 3].set_xlabel("x")
            axes[1, 3].set_ylabel("t")
            fig.colorbar(im13, ax=axes[1, 3])

            out_path = plot_dir / f"fluxgnn_superres_{plots_made}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            plots_made += 1

    print(f"[FluxGNN SuperRes] Saved {plots_made} plots to {plot_dir}")


if __name__ == "__main__":
    main()
