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
from hyperbolic_pde.models.fno import FNO2d


class FNODataset(Dataset):
    def __init__(self, x: np.ndarray, t: np.ndarray, u0: np.ndarray, u: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.u0 = torch.tensor(u0, dtype=torch.float32)
        self.u = torch.tensor(u, dtype=torch.float32)
        X, T = torch.meshgrid(self.x, self.t, indexing="ij")
        self.X = X
        self.T = T

    def __len__(self) -> int:
        return self.u.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        u0 = self.u0[idx]
        u0_grid = u0.unsqueeze(1).repeat(1, self.t.numel())
        inp = torch.stack([self.X, self.T, u0_grid], dim=0)
        out = self.u[idx].T.unsqueeze(0)
        return inp, out


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
    parser = argparse.ArgumentParser(description="Evaluate FNO on test set.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    fno_cfg = cfg["fno"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    dataset = load_dataset(Path(data_cfg["path"]))
    _, test_idx = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))

    test_data = FNODataset(dataset.x, dataset.t, dataset.u0[test_idx], dataset.u[test_idx])
    loader = DataLoader(test_data, batch_size=int(fno_cfg["batch_size"]), shuffle=False)

    model = FNO2d(
        in_channels=3,
        out_channels=1,
        width=int(fno_cfg["width"]),
        modes_x=int(fno_cfg["modes_x"]),
        modes_t=int(fno_cfg["modes_t"]),
        layers=int(fno_cfg["layers"]),
    ).to(device)
    model.load_state_dict(torch.load(Path(fno_cfg["save_path"]), map_location=device))
    model.eval()

    plot_dir = Path(fno_cfg.get("plot_dir", "hyperbolic_pde/runs/plots/fno"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    total_loss = 0.0
    total_count = 0
    plots_made = 0
    with torch.no_grad():
        for inp, out in loader:
            inp = inp.to(device)
            out = out.to(device)
            pred = model(inp)
            loss = (pred - out).pow(2).mean().item()
            total_loss += loss * inp.size(0)
            total_count += inp.size(0)

            if plots_made < int(fno_cfg.get("eval_plots", 3)):
                for b in range(pred.size(0)):
                    if plots_made >= int(fno_cfg.get("eval_plots", 3)):
                        break
                    pred_np = pred[b, 0].detach().cpu().numpy()
                    truth_np = out[b, 0].detach().cpu().numpy()
                    err_np = np.abs(pred_np - truth_np)

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
                    im0 = axes[0].pcolormesh(dataset.t, dataset.x, pred_np, shading="auto", cmap="jet")
                    axes[0].set_title("FNO prediction")
                    axes[0].set_xlabel("t")
                    axes[0].set_ylabel("x")
                    fig.colorbar(im0, ax=axes[0])

                    im1 = axes[1].pcolormesh(dataset.t, dataset.x, truth_np, shading="auto", cmap="jet")
                    axes[1].set_title("Godunov FVM truth")
                    axes[1].set_xlabel("t")
                    axes[1].set_ylabel("x")
                    fig.colorbar(im1, ax=axes[1])

                    im2 = axes[2].pcolormesh(dataset.t, dataset.x, err_np, shading="auto", cmap="magma")
                    axes[2].set_title("Absolute error")
                    axes[2].set_xlabel("t")
                    axes[2].set_ylabel("x")
                    fig.colorbar(im2, ax=axes[2])

                    out_path = plot_dir / f"fno_sample_{plots_made}.png"
                    fig.savefig(out_path, dpi=150)
                    plt.close(fig)
                    plots_made += 1

    mse = total_loss / max(1, total_count)
    print(f"[FNO] Test MSE: {mse:.6e}")
    print(f"[FNO] Saved {plots_made} plots to {plot_dir}")


if __name__ == "__main__":
    main()
