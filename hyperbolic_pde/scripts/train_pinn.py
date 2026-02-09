from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset
from hyperbolic_pde.models.pinn import (
    UniversalPINN,
    hyperbolic_residual,
    repeat_cond,
    sample_boundary,
    sample_initial,
    sample_uniform,
)


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


def sample_u0_at(
    x: torch.Tensor,
    u0_grid: torch.Tensor,
    x_grid: torch.Tensor,
) -> torch.Tensor:
    idx = torch.searchsorted(x_grid, x.squeeze(1))
    idx = torch.clamp(idx, 0, x_grid.numel() - 1)
    return u0_grid[idx].unsqueeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train universal PINN on hyperbolic PDE dataset.")
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
    train_idx, _ = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))

    x_grid = torch.tensor(dataset.x, dtype=torch.float32, device=device)
    u0_all = torch.tensor(dataset.u0, dtype=torch.float32, device=device)
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

    opt = torch.optim.Adam(model.parameters(), lr=float(pinn_cfg["lr"]))

    steps = int(pinn_cfg["steps"])
    batch_pdes = int(pinn_cfg["batch_pdes"])
    n_int = int(pinn_cfg["interior_samples"])
    n_init = int(pinn_cfg["initial_samples"])
    n_bc = int(pinn_cfg["boundary_samples"])

    lam_res = float(pinn_cfg.get("lambda_res", 1.0))
    lam_init = float(pinn_cfg.get("lambda_init", 1.0))
    lam_bc = float(pinn_cfg.get("lambda_bc", 0.1))

    x_min = float(data_cfg["x_min"])
    x_max = float(data_cfg["x_max"])
    t_min = 0.0
    t_max = float(data_cfg["t_max"])

    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        batch_indices = rng.choice(train_idx, size=batch_pdes, replace=True)

        loss_res = torch.tensor(0.0, device=device)
        loss_init = torch.tensor(0.0, device=device)
        loss_bc = torch.tensor(0.0, device=device)

        for idx in batch_indices:
            cond = ic_all[idx]
            cond_int = repeat_cond(cond, n_int)
            cond_init = repeat_cond(cond, n_init)
            cond_bc = repeat_cond(cond, n_bc)

            x_i, t_i = sample_uniform(n_int, x_min, x_max, t_min, t_max, device)
            res = hyperbolic_residual(model, x_i, t_i, cond_int)
            loss_res = loss_res + res.pow(2).mean()

            x0, t0 = sample_initial(n_init, x_min, x_max, t_min, device)
            u0_target = sample_u0_at(x0, u0_all[idx], x_grid)
            u0_pred = model(x0, t0, cond_init)
            loss_init = loss_init + (u0_pred - u0_target).pow(2).mean()

            x_l, t_l, x_r, t_r = sample_boundary(n_bc, x_min, x_max, t_min, t_max, device)
            u_l = model(x_l, t_l, cond_bc)
            u_r = model(x_r, t_r, cond_bc)
            loss_bc = loss_bc + (u_l - u_r).pow(2).mean()

        loss_res = loss_res / batch_pdes
        loss_init = loss_init / batch_pdes
        loss_bc = loss_bc / batch_pdes

        loss = lam_res * loss_res + lam_init * loss_init + lam_bc * loss_bc
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == 1:
            print(
                f"[PINN] step {step:5d} | total={loss.item():.3e} | "
                f"res={loss_res.item():.3e} | init={loss_init.item():.3e} | bc={loss_bc.item():.3e}"
            )

    save_path = Path(pinn_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved PINN checkpoint to {save_path}")


if __name__ == "__main__":
    main()
