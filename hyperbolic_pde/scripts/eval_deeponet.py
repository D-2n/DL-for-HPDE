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
from hyperbolic_pde.models.deeponet import DeepONet


class DeepONetDataset(Dataset):
    def __init__(self, ic: np.ndarray, u: np.ndarray) -> None:
        self.ic = torch.tensor(ic, dtype=torch.float32)
        self.u = u

    def __len__(self) -> int:
        return self.u.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        branch = self.ic[idx]
        target = torch.from_numpy(self.u[idx].T).reshape(-1).float()
        return branch, target


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepONet on test set.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    deep_cfg = cfg["deeponet"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    dataset = load_dataset(Path(data_cfg["path"]))
    _, test_idx = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))

    test_data = DeepONetDataset(dataset.ic[test_idx], dataset.u[test_idx])
    loader = DataLoader(test_data, batch_size=int(deep_cfg["batch_size"]), shuffle=False)

    x = torch.tensor(dataset.x, dtype=torch.float32, device=device)
    t = torch.tensor(dataset.t, dtype=torch.float32, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    trunk_in = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

    model = DeepONet(
        branch_in=int(data_cfg["ic_points"]),
        trunk_in=2,
        hidden_width=int(deep_cfg["hidden_width"]),
        branch_layers=int(deep_cfg["branch_layers"]),
        trunk_layers=int(deep_cfg["trunk_layers"]),
        latent_dim=int(deep_cfg["latent_dim"]),
        activation=str(deep_cfg.get("activation", "tanh")),
        use_bias=bool(deep_cfg.get("use_bias", True)),
    ).to(device)
    model.load_state_dict(torch.load(Path(deep_cfg["save_path"]), map_location=device))
    model.eval()

    plot_dir = Path(deep_cfg.get("plot_dir", "hyperbolic_pde/runs/plots/deeponet"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    total_loss = 0.0
    total_count = 0
    plots_made = 0
    with torch.no_grad():
        for branch, target in loader:
            branch = branch.to(device)
            target = target.to(device)
            pred = model(branch, trunk_in)
            loss = (pred - target).pow(2).mean().item()
            total_loss += loss * branch.size(0)
            total_count += branch.size(0)

            if plots_made < int(deep_cfg.get("eval_plots", 3)):
                for b in range(pred.size(0)):
                    if plots_made >= int(deep_cfg.get("eval_plots", 3)):
                        break
                    pred_np = pred[b].reshape(x.numel(), t.numel()).detach().cpu().numpy()
                    truth_np = target[b].reshape(x.numel(), t.numel()).detach().cpu().numpy()
                    err_np = np.abs(pred_np - truth_np)
                    vmin = float(np.min(truth_np))
                    vmax = float(np.max(truth_np))

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
                    im0 = axes[0].pcolormesh(
                        dataset.t, dataset.x, pred_np, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
                    )
                    axes[0].set_title("DeepONet prediction")
                    axes[0].set_xlabel("t")
                    axes[0].set_ylabel("x")
                    fig.colorbar(im0, ax=axes[0])

                    im1 = axes[1].pcolormesh(
                        dataset.t, dataset.x, truth_np, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
                    )
                    axes[1].set_title("Godunov FVM truth")
                    axes[1].set_xlabel("t")
                    axes[1].set_ylabel("x")
                    fig.colorbar(im1, ax=axes[1])

                    im2 = axes[2].pcolormesh(dataset.t, dataset.x, err_np, shading="auto", cmap="magma")
                    axes[2].set_title("Absolute error")
                    axes[2].set_xlabel("t")
                    axes[2].set_ylabel("x")
                    fig.colorbar(im2, ax=axes[2])

                    out_path = plot_dir / f"deeponet_sample_{plots_made}.png"
                    fig.savefig(out_path, dpi=150)
                    plt.close(fig)
                    plots_made += 1

    mse = total_loss / max(1, total_count)
    print(f"[DeepONet] Test MSE: {mse:.6e}")
    print(f"[DeepONet] Saved {plots_made} plots to {plot_dir}")


if __name__ == "__main__":
    main()
