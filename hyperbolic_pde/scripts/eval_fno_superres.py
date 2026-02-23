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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FNO on 1x and 2x test data.")
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

    test_data_1x = FNODataset(x1, t1, u0_1, u1)
    test_data_2x = FNODataset(x2, t2, u0_2, u2)
    loader_1x = DataLoader(test_data_1x, batch_size=int(fno_cfg["batch_size"]), shuffle=False)
    loader_2x = DataLoader(test_data_2x, batch_size=int(fno_cfg["batch_size"]), shuffle=False)

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

    plot_dir = Path("hyperbolic_pde/runs/plots/fno_superres")
    plot_dir.mkdir(parents=True, exist_ok=True)

    def total_variation_map(u: np.ndarray) -> np.ndarray:
        tv = np.zeros_like(u)
        tv[:-1, :] += np.abs(u[1:, :] - u[:-1, :])
        tv[:, :-1] += np.abs(u[:, 1:] - u[:, :-1])
        return tv

    def eval_loader(loader: DataLoader) -> float:
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for inp, out in loader:
                inp = inp.to(device)
                out = out.to(device)
                pred = model(inp)
                loss = (pred - out).pow(2).mean().item()
                total_loss += loss * inp.size(0)
                total_count += inp.size(0)
        return total_loss / max(1, total_count)

    mse_1x = eval_loader(loader_1x)
    mse_2x = eval_loader(loader_2x)

    x1_t = torch.tensor(x1, dtype=torch.float32, device=device)
    t1_t = torch.tensor(t1, dtype=torch.float32, device=device)
    X1, T1 = torch.meshgrid(x1_t, t1_t, indexing="ij")
    x2_t = torch.tensor(x2, dtype=torch.float32, device=device)
    t2_t = torch.tensor(t2, dtype=torch.float32, device=device)
    X2, T2 = torch.meshgrid(x2_t, t2_t, indexing="ij")

    plots_made = 0
    max_plots = int(fno_cfg.get("eval_plots", 3))
    with torch.no_grad():
        for idx in range(min(max_plots, u1.shape[0], u2.shape[0])):
            u0_1_t = torch.tensor(u0_1[idx], dtype=torch.float32, device=device)
            u0_2_t = torch.tensor(u0_2[idx], dtype=torch.float32, device=device)

            u0_grid_1 = u0_1_t.unsqueeze(1).repeat(1, t1_t.numel())
            u0_grid_2 = u0_2_t.unsqueeze(1).repeat(1, t2_t.numel())

            inp_1 = torch.stack([X1, T1, u0_grid_1], dim=0).unsqueeze(0)
            inp_2 = torch.stack([X2, T2, u0_grid_2], dim=0).unsqueeze(0)

            pred_1 = model(inp_1)[0, 0]
            pred_2 = model(inp_2)[0, 0]

            truth_1 = torch.tensor(u1[idx].T, dtype=torch.float32, device=device)
            truth_2 = torch.tensor(u2[idx].T, dtype=torch.float32, device=device)

            pred_1_np = pred_1.detach().cpu().numpy()
            pred_2_np = pred_2.detach().cpu().numpy()
            truth_1_np = truth_1.detach().cpu().numpy()
            truth_2_np = truth_2.detach().cpu().numpy()
            err_1 = np.abs(pred_1_np - truth_1_np)
            err_2 = np.abs(pred_2_np - truth_2_np)
            tv_1 = total_variation_map(pred_1_np)
            tv_2 = total_variation_map(pred_2_np)
           # err_1 = 
           # err_2 = 
            vmin_1 = float(np.min(truth_1_np))
            vmax_1 = float(np.max(truth_1_np))
            vmin_2 = float(np.min(truth_2_np))
            vmax_2 = float(np.max(truth_2_np))
            #vmin_1 = 0
            #vmax_1 = 0.5
            #vmin_2 = 0
            #vmax_2 = 0.5

            fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
            im00 = axes[0, 0].pcolormesh(t1, x1, pred_1_np, shading="auto", cmap="jet", vmin=vmin_1, vmax=vmax_1)
            axes[0, 0].set_title("FNO prediction (1x)")
            axes[0, 0].set_xlabel("t")
            axes[0, 0].set_ylabel("x")
            fig.colorbar(im00, ax=axes[0, 0])

            im01 = axes[0, 1].pcolormesh(t1, x1, truth_1_np, shading="auto", cmap="jet", vmin=vmin_1, vmax=vmax_1)
            axes[0, 1].set_title("Godunov FVM truth (1x)")
            axes[0, 1].set_xlabel("t")
            axes[0, 1].set_ylabel("x")
            fig.colorbar(im01, ax=axes[0, 1])

            im02 = axes[0, 2].pcolormesh(t1, x1, err_1, shading="auto", cmap="magma")
            axes[0, 2].set_title("Abs error (1x)")
            axes[0, 2].set_xlabel("t")
            axes[0, 2].set_ylabel("x")
            fig.colorbar(im02, ax=axes[0, 2])

            im03 = axes[0, 3].pcolormesh(t1, x1, tv_1, shading="auto", cmap="viridis")
            axes[0, 3].set_title("Total variation (1x)")
            axes[0, 3].set_xlabel("t")
            axes[0, 3].set_ylabel("x")
            fig.colorbar(im03, ax=axes[0, 3])

            im10 = axes[1, 0].pcolormesh(t2, x2, pred_2_np, shading="auto", cmap="jet", vmin=vmin_2, vmax=vmax_2)
            axes[1, 0].set_title("FNO prediction (2x)")
            axes[1, 0].set_xlabel("t")
            axes[1, 0].set_ylabel("x")
            fig.colorbar(im10, ax=axes[1, 0])

            im11 = axes[1, 1].pcolormesh(t2, x2, truth_2_np, shading="auto", cmap="jet", vmin=vmin_2, vmax=vmax_2)
            axes[1, 1].set_title("Godunov FVM truth (2x)")
            axes[1, 1].set_xlabel("t")
            axes[1, 1].set_ylabel("x")
            fig.colorbar(im11, ax=axes[1, 1])

            im12 = axes[1, 2].pcolormesh(t2, x2, err_2, shading="auto", cmap="magma")
            axes[1, 2].set_title("Abs error (2x)")
            axes[1, 2].set_xlabel("t")
            axes[1, 2].set_ylabel("x")
            fig.colorbar(im12, ax=axes[1, 2])

            im13 = axes[1, 3].pcolormesh(t2, x2, tv_2, shading="auto", cmap="viridis")
            axes[1, 3].set_title("Total variation (2x)")
            axes[1, 3].set_xlabel("t")
            axes[1, 3].set_ylabel("x")
            fig.colorbar(im13, ax=axes[1, 3])

            out_path = plot_dir / f"fno_superres_{plots_made}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            plots_made += 1

    print(f"[FNO SuperRes] Test MSE 1x: {mse_1x:.6e}")
    print(f"[FNO SuperRes] Test MSE 2x: {mse_2x:.6e}")
    print(f"[FNO SuperRes] Saved {plots_made} plots to {plot_dir}")


if __name__ == "__main__":
    main()
