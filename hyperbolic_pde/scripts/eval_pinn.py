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
from hyperbolic_pde.models.pinn import UniversalPINN


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
    parser = argparse.ArgumentParser(description="Evaluate universal PINN on test set.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    pinn_cfg = cfg["pinn"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    dataset = load_dataset(Path(data_cfg["path"]))
    _, test_idx = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))

    x = torch.tensor(dataset.x, dtype=torch.float32, device=device)
    t = torch.tensor(dataset.t, dtype=torch.float32, device=device)
    ic_all = torch.tensor(dataset.ic, dtype=torch.float32, device=device)

    model = UniversalPINN(
        hidden_layers=int(pinn_cfg["hidden_layers"]),
        hidden_width=int(pinn_cfg["hidden_width"]),
        cond_dim=int(data_cfg["ic_points"]),
        activation=str(pinn_cfg.get("activation", "tanh")),
        hard_boundary=bool(pinn_cfg.get("hard_boundary", False)),
        x_min=float(data_cfg["x_min"]),
        x_max=float(data_cfg["x_max"]),
    ).to(device)
    model.load_state_dict(torch.load(Path(pinn_cfg["save_path"]), map_location=device))
    model.eval()

    plot_dir = Path(pinn_cfg.get("plot_dir", "hyperbolic_pde/runs/plots/pinn"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    eval_samples = int(pinn_cfg.get("eval_samples", 5))
    eval_idx = test_idx[:eval_samples]

    X, T = torch.meshgrid(x, t, indexing="ij")
    Xf = X.reshape(-1, 1)
    Tf = T.reshape(-1, 1)

    mse_list = []
    plots_made = 0
    with torch.no_grad():
        for idx in eval_idx:
            cond = ic_all[idx]
            cond_rep = cond.unsqueeze(0).repeat(Xf.size(0), 1)
            pred = model(Xf, Tf, cond_rep).reshape(x.numel(), t.numel())
            truth = torch.tensor(dataset.u[idx].T, dtype=torch.float32, device=device)
            mse = torch.mean((pred - truth) ** 2).item()
            mse_list.append(mse)
            print(f"[PINN] sample {int(idx)} MSE: {mse:.6e}")

            if plots_made < int(pinn_cfg.get("eval_plots", 3)):
                pred_np = pred.detach().cpu().numpy()
                truth_np = truth.detach().cpu().numpy()
                err_np = np.abs(pred_np - truth_np)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
                im0 = axes[0].pcolormesh(dataset.t, dataset.x, pred_np.T, shading="auto", cmap="jet")
                axes[0].set_title("PINN prediction")
                axes[0].set_xlabel("t")
                axes[0].set_ylabel("x")
                fig.colorbar(im0, ax=axes[0])

                im1 = axes[1].pcolormesh(dataset.t, dataset.x, truth_np.T, shading="auto", cmap="jet")
                axes[1].set_title("Godunov FVM truth")
                axes[1].set_xlabel("t")
                axes[1].set_ylabel("x")
                fig.colorbar(im1, ax=axes[1])

                im2 = axes[2].pcolormesh(dataset.t, dataset.x, err_np.T, shading="auto", cmap="magma")
                axes[2].set_title("Absolute error")
                axes[2].set_xlabel("t")
                axes[2].set_ylabel("x")
                fig.colorbar(im2, ax=axes[2])

                out_path = plot_dir / f"pinn_sample_{plots_made}.png"
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                plots_made += 1

    if mse_list:
        print(f"[PINN] mean MSE: {float(np.mean(mse_list)):.6e}")
    print(f"[PINN] Saved {plots_made} plots to {plot_dir}")


if __name__ == "__main__":
    main()
