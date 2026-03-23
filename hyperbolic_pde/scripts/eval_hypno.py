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

from hyperbolic_pde.data.fvm import load_dataset
from hyperbolic_pde.cfl import annotate_cfl, print_cfl_report
from hyperbolic_pde.models.hypno import HypNO


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


def total_variation_map(u: np.ndarray) -> np.ndarray:
    tv = np.zeros_like(u)
    tv[:-1, :] += np.abs(u[1:, :] - u[:-1, :])
    tv[:, :-1] += np.abs(u[:, 1:] - u[:, :-1])
    return tv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HypNO on test set.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    model_cfg = cfg["hypno"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    dataset = load_dataset(Path(data_cfg["path"]))
    cfl_metrics = print_cfl_report(data_cfg, dataset.x, dataset.t)
    _, test_idx = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))

    _rx = model_cfg.get("radius_x", None)
    _rt = model_cfg.get("radius_t", None)
    radius_x = float(_rx) if _rx is not None else None
    radius_t = float(_rt) if _rt is not None else None

    model = HypNO(
        stencil_k_x=int(model_cfg.get("stencil_k_x", 3)),
        stencil_k_t=int(model_cfg.get("stencil_k_t", 2)),
        d_latent=int(model_cfg.get("d_latent", 128)),
        d_hidden=int(model_cfg.get("d_hidden", 128)),
        d_time=int(model_cfg.get("d_time", 32)),
        n_layers=int(model_cfg.get("n_layers", 6)),
        activation=str(model_cfg.get("activation", "gelu")),
        radius_x=radius_x,
        radius_t=radius_t,
    ).to(device)
    model.load_state_dict(torch.load(Path(model_cfg["save_path"]), map_location=device))
    model.eval()

    x_grid = torch.tensor(dataset.x, dtype=torch.float32, device=device)
    t_grid = torch.tensor(dataset.t, dtype=torch.float32, device=device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[HypNO] {n_params:,} trainable parameters")

    plot_dir = Path(model_cfg.get("plot_dir", "hyperbolic_pde/runs/plots/hypno"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    nt = len(dataset.t)

    # evaluate: predict full trajectory in one shot
    u0_test = torch.tensor(dataset.u0[test_idx], dtype=torch.float32, device=device)
    u_test = torch.tensor(dataset.u[test_idx], dtype=torch.float32, device=device)
    n_test = u0_test.shape[0]

    # process in batches to avoid OOM
    batch_size = int(model_cfg.get("batch_size", 8))
    pred_chunks = []
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            u0_batch = u0_test[i : i + batch_size]
            pred_chunks.append(model(u0_batch, x_grid, t_grid))
    pred_full = torch.cat(pred_chunks, dim=0)                            # [N_test, nt, nx]

    # metrics
    mse = (pred_full - u_test).pow(2).mean().item()
    mae = (pred_full - u_test).abs().mean().item()
    rel_l2 = (
        (pred_full - u_test).pow(2).sum() / u_test.pow(2).sum().clamp(min=1e-12)
    ).sqrt().item()

    print(f"[HypNO] Test MSE:  {mse:.6e}")
    print(f"[HypNO] Test MAE:  {mae:.6e}")
    print(f"[HypNO] Test rL2:  {rel_l2:.6e}")

    # per-time-step error
    per_t_mse = (pred_full - u_test).pow(2).mean(dim=(0, 2))
    per_t_mae = (pred_full - u_test).abs().mean(dim=(0, 2))

    fig_err, ax_err = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax_err[0].plot(dataset.t, per_t_mse.cpu().numpy())
    ax_err[0].set_xlabel("t")
    ax_err[0].set_ylabel("MSE")
    ax_err[0].set_title("MSE vs time")
    ax_err[0].set_yscale("log")

    ax_err[1].plot(dataset.t, per_t_mae.cpu().numpy())
    ax_err[1].set_xlabel("t")
    ax_err[1].set_ylabel("MAE")
    ax_err[1].set_title("MAE vs time")
    ax_err[1].set_yscale("log")

    fig_err.savefig(plot_dir / "hypno_error_vs_time.png", dpi=150)
    plt.close(fig_err)

    # sample plots (4-panel layout)
    max_plots = int(model_cfg.get("eval_plots", 3))
    plots_made = 0
    for b in range(n_test):
        if plots_made >= max_plots:
            break
        pred_np = pred_full[b].cpu().numpy()
        truth_np = u_test[b].cpu().numpy()
        err_np = np.abs(pred_np - truth_np)
        tv_np = total_variation_map(pred_np)
        vmin = float(np.min(truth_np))
        vmax = float(np.max(truth_np))

        fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
        annotate_cfl(fig, cfl_metrics)

        im0 = axes[0].pcolormesh(
            dataset.x, dataset.t, pred_np, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
        )
        axes[0].set_title("HypNO prediction")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("t")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].pcolormesh(
            dataset.x, dataset.t, truth_np, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
        )
        axes[1].set_title("Godunov FVM truth")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("t")
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].pcolormesh(dataset.x, dataset.t, err_np, shading="auto", cmap="magma")
        axes[2].set_title("Absolute error")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("t")
        fig.colorbar(im2, ax=axes[2])

        im3 = axes[3].pcolormesh(dataset.x, dataset.t, tv_np, shading="auto", cmap="viridis")
        axes[3].set_title("Total variation")
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("t")
        fig.colorbar(im3, ax=axes[3])

        out_path = plot_dir / f"hypno_sample_{plots_made}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        plots_made += 1

    print(f"[HypNO] Saved {plots_made} plots + error curve to {plot_dir}")


if __name__ == "__main__":
    main()
