from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from hyperbolic_pde.utils.runtime import apply_runtime_overrides, resolve_config_path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset
from hyperbolic_pde.cfl import annotate_cfl, print_cfl_report
from hyperbolic_pde.models.hypfluxno import HypFluxNO


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
    parser = argparse.ArgumentParser(description="Evaluate HypFluxNO on test set.")
    parser.add_argument(
        "--config", type=str,
        default=str(resolve_config_path(ROOT / "configs")),
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Run directory to load model from and save plots to. "
             "If not given, reads from latest_run.txt.",
    )
    args = parser.parse_args()

    # resolve run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        latest_path = Path("hyperbolic_pde/runs/hypfluxno/latest_run.txt")
        if latest_path.exists():
            run_dir = Path(latest_path.read_text(encoding="utf-8").strip())
        else:
            run_dir = None

    # load config: prefer run dir's config copy, fall back to args.config
    if run_dir and (run_dir / "config.yaml").exists():
        with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"[HypFluxNO] Using config from {run_dir / 'config.yaml'}")
    else:
        cfg = load_config(Path(args.config))
    cfg = apply_runtime_overrides(cfg)

    data_cfg = cfg["data"]
    model_cfg = cfg["hypfluxno"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    dataset = load_dataset(Path(data_cfg["path"]))
    cfl_metrics = print_cfl_report(data_cfg, dataset.x, dataset.t)
    _, test_idx = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))

    model = HypFluxNO(
        stencil_k_t=int(model_cfg.get("stencil_k_t", 2)),
        stencil_k_x=int(model_cfg.get("stencil_k_x", 3)),
        d_latent=int(model_cfg.get("d_latent", 64)),
        d_hidden=int(model_cfg.get("d_hidden", 64)),
        d_time=int(model_cfg.get("d_time", 32)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        activation=str(model_cfg.get("activation", "gelu")),
        weno_eps=float(model_cfg.get("weno_eps", 1e-6)),
        weno_p=float(model_cfg.get("weno_p", 2.0)),
        causal_temporal=bool(model_cfg.get("causal_temporal", True)),
        radius_x=float(model_cfg["radius_x"]) if model_cfg.get("radius_x") is not None else None,
        radius_t=float(model_cfg["radius_t"]) if model_cfg.get("radius_t") is not None else None,
        readout=str(model_cfg.get("readout", "gelu")),
    ).to(device)

    # load weights: prefer run dir's final model, fall back to save_path
    if run_dir and (run_dir / "model_final.pt").exists():
        weights_path = run_dir / "model_final.pt"
    else:
        weights_path = Path(model_cfg["save_path"])
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[HypFluxNO] Loaded weights from {weights_path}")

    x_grid = torch.tensor(dataset.x, dtype=torch.float32, device=device)
    t_grid = torch.tensor(dataset.t, dtype=torch.float32, device=device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[HypFluxNO] {n_params:,} trainable parameters")

    # plots go into the run directory if available, otherwise plot_dir from config
    if run_dir:
        plot_dir = run_dir / "plots"
    else:
        plot_dir = Path(model_cfg.get("plot_dir", "hyperbolic_pde/runs/plots/hypfluxno"))
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"[HypFluxNO] Saving plots to {plot_dir}")

    # evaluate: predict full trajectory in one shot
    u0_test = torch.tensor(dataset.u0[test_idx], dtype=torch.float32, device=device)
    u_test = torch.tensor(dataset.u[test_idx], dtype=torch.float32, device=device)
    n_test = u0_test.shape[0]

    batch_size = int(model_cfg.get("batch_size", 8))
    pred_chunks = []
    beta_chunks = []
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            u0_batch = u0_test[i : i + batch_size]
            u_pred, beta_map = model(u0_batch, x_grid, t_grid)
            pred_chunks.append(u_pred)
            beta_chunks.append(beta_map)
    pred_full = torch.cat(pred_chunks, dim=0)
    beta_full = torch.cat(beta_chunks, dim=0)

    # metrics
    mse = (pred_full - u_test).pow(2).mean().item()
    mae = (pred_full - u_test).abs().mean().item()
    rel_l2 = (
        (pred_full - u_test).pow(2).sum() / u_test.pow(2).sum().clamp(min=1e-12)
    ).sqrt().item()

    print(f"[HypFluxNO] Test MSE:  {mse:.6e}")
    print(f"[HypFluxNO] Test MAE:  {mae:.6e}")
    print(f"[HypFluxNO] Test rL2:  {rel_l2:.6e}")

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

    fig_err.savefig(plot_dir / "hypfluxno_error_vs_time.png", dpi=150)
    plt.close(fig_err)

    # sample plots: 4 panels (pred, truth, error, WENO flux beta indicator)
    max_plots = int(model_cfg.get("eval_plots", 3))
    plots_made = 0
    for b in range(n_test):
        if plots_made >= max_plots:
            break
        pred_np = pred_full[b].cpu().numpy()
        truth_np = u_test[b].cpu().numpy()
        err_np = np.abs(pred_np - truth_np)
        beta_np = beta_full[b].cpu().numpy()
        vmin = float(np.min(truth_np))
        vmax = float(np.max(truth_np))

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        annotate_cfl(fig, cfl_metrics)

        im0 = axes[0].pcolormesh(
            dataset.x, dataset.t, pred_np, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
        )
        axes[0].set_title("HypFluxNO prediction")
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

        im3 = axes[3].pcolormesh(dataset.x, dataset.t, beta_np, shading="auto", cmap="hot")
        axes[3].set_title("WENO flux indicator")
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("t")
        fig.colorbar(im3, ax=axes[3])

        out_path = plot_dir / f"hypfluxno_sample_{plots_made}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        plots_made += 1

    # save metrics summary
    metrics_path = plot_dir.parent / "metrics.txt" if run_dir else plot_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"Test MSE:  {mse:.6e}\n")
        f.write(f"Test MAE:  {mae:.6e}\n")
        f.write(f"Test rL2:  {rel_l2:.6e}\n")
        f.write(f"Weights:   {weights_path}\n")
        f.write(f"N test:    {n_test}\n")
        f.write(f"N params:  {n_params:,}\n")

    print(f"[HypFluxNO] Saved {plots_made} plots + error curve to {plot_dir}")
    print(f"[HypFluxNO] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
