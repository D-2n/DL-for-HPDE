from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset
from hyperbolic_pde.cfl import annotate_cfl, print_cfl_report
from hyperbolic_pde.diagnostics import compare_one_step_update_sizes
from hyperbolic_pde.models.fluxgnn import FluxGNN1D


class FluxGNNDataset(Dataset):
    def __init__(self, u0: np.ndarray, u: np.ndarray) -> None:
        self.u0 = torch.tensor(u0, dtype=torch.float32)
        self.u = torch.tensor(u, dtype=torch.float32)

    def __len__(self) -> int:
        return self.u.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.u0[idx], self.u[idx]


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


def split_train_val(train_idx: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if val_fraction <= 0.0 or train_idx.size == 0:
        return train_idx, np.array([], dtype=train_idx.dtype)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(len(train_idx) * val_fraction))
    return perm[n_val:], perm[:n_val]


def total_variation_map(u: np.ndarray) -> np.ndarray:
    tv = np.zeros_like(u)
    tv[:-1, :] += np.abs(u[1:, :] - u[:-1, :])
    tv[:, :-1] += np.abs(u[:, 1:] - u[:, :-1])
    return tv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FluxGNN on test set.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--diag_splits",
        type=str,
        default="test",
        help="Comma-separated splits for one-step diagnostic: train,val,test.",
    )
    parser.add_argument(
        "--diag_batch_size",
        type=int,
        default=32,
        help="Batch size for one-step diagnostic.",
    )
    parser.add_argument(
        "--diag_time_idx",
        type=int,
        default=0,
        help="Time index to use for one-step diagnostic.",
    )
    parser.add_argument(
        "--diag_riemann",
        action="store_true",
        help="Also run one-step diagnostic on a Riemann initial condition.",
    )
    parser.add_argument("--u_left", type=float, default=0.2)
    parser.add_argument("--u_right", type=float, default=0.8)
    parser.add_argument("--x0", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    flux_cfg = cfg["fluxgnn"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    dataset = load_dataset(Path(data_cfg["path"]))
    cfl_metrics = print_cfl_report(data_cfg, dataset.x, dataset.t)
    train_idx, test_idx = split_indices(
        dataset.u.shape[0],
        float(data_cfg["train_fraction"]),
        int(cfg.get("seed", 42)),
    )
    val_fraction = float(data_cfg.get("val_fraction", 0.0))
    train_idx, val_idx = split_train_val(train_idx, val_fraction, int(cfg.get("seed", 42)))
    test_data = FluxGNNDataset(dataset.u0[test_idx], dataset.u[test_idx])
    loader = DataLoader(test_data, batch_size=int(flux_cfg["batch_size"]), shuffle=False)

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

    dx = float(dataset.x[1] - dataset.x[0])
    dt = float(dataset.t[1] - dataset.t[0])
    n_steps = int(dataset.u.shape[1])
    boundary = str(data_cfg.get("boundary", "ghost"))

    plot_dir = Path(flux_cfg.get("plot_dir", "hyperbolic_pde/runs/plots/fluxgnn"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    total_loss = 0.0
    total_count = 0
    plots_made = 0
    max_plots = int(flux_cfg.get("eval_plots", 3))

    with torch.no_grad():
        for u0, u in loader:
            u0 = u0.to(device)
            u = u.to(device)
            pred = model(u0, dt, dx, n_steps, boundary)
            loss = (pred - u).pow(2).mean().item()
            total_loss += loss * u0.size(0)
            total_count += u0.size(0)

            if plots_made < max_plots:
                for b in range(pred.size(0)):
                    if plots_made >= max_plots:
                        break
                    pred_np = pred[b].detach().cpu().numpy()
                    truth_np = u[b].detach().cpu().numpy()
                    err_np = np.abs(pred_np - truth_np)
                    tv_np = total_variation_map(pred_np)
                    vmin = float(np.min(truth_np))
                    vmax = float(np.max(truth_np))

                    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
                    annotate_cfl(fig, cfl_metrics)
                    im0 = axes[0].pcolormesh(
                        dataset.x, dataset.t, pred_np, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
                    )
                    axes[0].set_title("FluxGNN prediction")
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

                    out_path = plot_dir / f"fluxgnn_sample_{plots_made}.png"
                    fig.savefig(out_path, dpi=150)
                    plt.close(fig)
                    plots_made += 1

    mse = total_loss / max(1, total_count)
    print(f"[FluxGNN] Test MSE: {mse:.6e}")
    print(f"[FluxGNN] Saved {plots_made} plots to {plot_dir}")

    # One-step update diagnostics
    diag_splits = [s.strip().lower() for s in args.diag_splits.split(",") if s.strip()]
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    time_idx = int(np.clip(args.diag_time_idx, 0, dataset.u.shape[1] - 1))

    def _run_diag(split_name: str, idx: np.ndarray) -> None:
        if idx.size == 0:
            print(f"[OneStep] split={split_name} skipped (empty).")
            return
        batch_size = int(min(args.diag_batch_size, idx.size))
        chosen = rng.choice(idx, size=batch_size, replace=False)
        u_batch = torch.tensor(dataset.u[chosen, time_idx], dtype=torch.float32, device=device)
        compare_one_step_update_sizes(
            model,
            u_batch,
            dt=dt,
            dx=dx,
            boundary=boundary,
            label=f"{split_name}, t_idx={time_idx}",
        )

    for split in diag_splits:
        if split == "train":
            _run_diag("train", train_idx)
        elif split == "val":
            _run_diag("val", val_idx)
        elif split == "test":
            _run_diag("test", test_idx)
        else:
            print(f"[OneStep] Unknown split '{split}', skipping.")

    if args.diag_riemann:
        x = dataset.x
        u0 = np.where(x <= float(args.x0), float(args.u_left), float(args.u_right)).astype(np.float32)
        u_batch = torch.tensor(u0, dtype=torch.float32, device=device).unsqueeze(0)
        compare_one_step_update_sizes(
            model,
            u_batch,
            dt=dt,
            dx=dx,
            boundary=boundary,
            label=f"riemann uL={args.u_left}, uR={args.u_right}",
        )


if __name__ == "__main__":
    main()
