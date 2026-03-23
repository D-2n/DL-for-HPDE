from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset
from hyperbolic_pde.models.pinn import (
    repeat_cond,
    sample_boundary,
    sample_initial,
    sample_uniform,
)
from hyperbolic_pde.models.vpinn import VPINN


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
    if val_fraction <= 0.0:
        return train_idx, np.array([], dtype=train_idx.dtype)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(len(train_idx) * val_fraction))
    return perm[n_val:], perm[:n_val]


def make_optimizer(params, cfg: dict) -> tuple[torch.optim.Optimizer, bool]:
    name = str(cfg.get("optimizer", "lbfgs")).lower()
    lr = float(cfg.get("lr", 1.0e-3))
    if name == "lbfgs":
        opt = torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=int(cfg.get("lbfgs_max_iter", 1)),
            history_size=int(cfg.get("lbfgs_history_size", 100)),
            line_search_fn=cfg.get("lbfgs_line_search"),
        )
        return opt, True
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0))), False
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.0))), False
    return torch.optim.Adam(params, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0))), False


def sample_u0_at(
    x: torch.Tensor,
    u0_grid: torch.Tensor,
    x_grid: torch.Tensor,
) -> torch.Tensor:
    idx = torch.searchsorted(x_grid, x.squeeze(1))
    idx = torch.clamp(idx, 0, x_grid.numel() - 1)
    return u0_grid[idx].unsqueeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VPINN on hyperbolic PDE dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    vpinn_cfg = cfg["vpinn"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    dataset = load_dataset(Path(data_cfg["path"]))
    train_idx, _ = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    train_idx, val_idx = split_train_val(train_idx, val_fraction, int(cfg.get("seed", 42)))

    x_grid = torch.tensor(dataset.x, dtype=torch.float32, device=device)
    u0_all = torch.tensor(dataset.u0, dtype=torch.float32, device=device)
    ic_all = torch.tensor(dataset.ic, dtype=torch.float32, device=device)

    model = VPINN(
        hidden_layers=int(vpinn_cfg["hidden_layers"]),
        hidden_width=int(vpinn_cfg["hidden_width"]),
        cond_dim=int(data_cfg["ic_points"]),
        activation=str(vpinn_cfg.get("activation", "tanh")),
        hard_boundary=bool(vpinn_cfg.get("hard_boundary", False)),
        x_min=float(data_cfg["x_min"]),
        x_max=float(data_cfg["x_max"]),
        t_min=0.0,
        t_max=float(data_cfg["t_max"]),
        n_test=int(vpinn_cfg.get("n_test", 2)),
    ).to(device)

    opt, use_lbfgs = make_optimizer(model.parameters(), vpinn_cfg)

    epochs = int(vpinn_cfg["epochs"])
    batch_pdes = int(vpinn_cfg["batch_pdes"])
    val_batch_pdes = int(vpinn_cfg.get("val_batch_pdes", batch_pdes))
    n_int = int(vpinn_cfg["interior_samples"])
    n_init = int(vpinn_cfg["initial_samples"])
    n_bc = int(vpinn_cfg["boundary_samples"])

    lam_weak = float(vpinn_cfg.get("lambda_weak", 1.0))
    lam_init = float(vpinn_cfg.get("lambda_init", 1.0))
    lam_bc = float(vpinn_cfg.get("lambda_bc", 0.1))

    x_min = float(data_cfg["x_min"])
    x_max = float(data_cfg["x_max"])
    t_min = 0.0
    t_max = float(data_cfg["t_max"])

    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        batch_indices = rng.choice(train_idx, size=batch_pdes, replace=True)

        samples = []
        for idx in batch_indices:
            x_i, t_i = sample_uniform(n_int, x_min, x_max, t_min, t_max, device)
            x0, t0 = sample_initial(n_init, x_min, x_max, t_min, device)
            x_l, t_l, x_r, t_r = sample_boundary(n_bc, x_min, x_max, t_min, t_max, device)
            samples.append((idx, x_i, t_i, x0, t0, x_l, t_l, x_r, t_r))

        def compute_losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            loss_weak = torch.tensor(0.0, device=device)
            loss_init = torch.tensor(0.0, device=device)
            loss_bc = torch.tensor(0.0, device=device)

            for idx, x_i, t_i, x0, t0, x_l, t_l, x_r, t_r in samples:
                cond = ic_all[idx]
                cond_int = repeat_cond(cond, n_int)
                cond_init = repeat_cond(cond, n_init)
                cond_bc = repeat_cond(cond, n_bc)

                loss_weak = loss_weak + model.weak_residual_loss(x_i, t_i, cond_int)

                u0_target = sample_u0_at(x0, u0_all[idx], x_grid)
                u0_pred = model(x0, t0, cond_init)
                loss_init = loss_init + (u0_pred - u0_target).pow(2).mean()

                u_l = model(x_l, t_l, cond_bc)
                u_r = model(x_r, t_r, cond_bc)
                loss_bc = loss_bc + (u_l - u_r).pow(2).mean()

            loss_weak = loss_weak / batch_pdes
            loss_init = loss_init / batch_pdes
            loss_bc = loss_bc / batch_pdes

            loss = lam_weak * loss_weak + lam_init * loss_init + lam_bc * loss_bc
            return loss, loss_weak, loss_init, loss_bc

        loss_parts: dict[str, torch.Tensor] = {}

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            loss, loss_weak, loss_init, loss_bc = compute_losses()
            loss.backward()
            loss_parts["loss"] = loss
            loss_parts["loss_weak"] = loss_weak
            loss_parts["loss_init"] = loss_init
            loss_parts["loss_bc"] = loss_bc
            return loss

        if use_lbfgs:
            opt.step(closure)
            loss = loss_parts["loss"]
            loss_weak = loss_parts["loss_weak"]
            loss_init = loss_parts["loss_init"]
            loss_bc = loss_parts["loss_bc"]
        else:
            opt.zero_grad(set_to_none=True)
            loss, loss_weak, loss_init, loss_bc = compute_losses()
            loss.backward()
            opt.step()

        print(
            f"[VPINN] epoch {epoch:5d}/{epochs} | total={loss.item():.3e} | "
            f"weak={loss_weak.item():.3e} | init={loss_init.item():.3e} | bc={loss_bc.item():.3e}"
        )

        if val_idx.size > 0:
            model.eval()
            val_indices = rng.choice(val_idx, size=val_batch_pdes, replace=True)
            val_loss_weak = torch.tensor(0.0, device=device)
            val_loss_init = torch.tensor(0.0, device=device)
            val_loss_bc = torch.tensor(0.0, device=device)

            for idx in val_indices:
                cond = ic_all[idx]
                cond_int = repeat_cond(cond, n_int)
                cond_init = repeat_cond(cond, n_init)
                cond_bc = repeat_cond(cond, n_bc)

                x_i, t_i = sample_uniform(n_int, x_min, x_max, t_min, t_max, device)
                val_loss_weak = val_loss_weak + model.weak_residual_loss(x_i, t_i, cond_int)

                x0, t0 = sample_initial(n_init, x_min, x_max, t_min, device)
                u0_target = sample_u0_at(x0, u0_all[idx], x_grid)
                u0_pred = model(x0, t0, cond_init)
                val_loss_init = val_loss_init + (u0_pred - u0_target).pow(2).mean()

                x_l, t_l, x_r, t_r = sample_boundary(n_bc, x_min, x_max, t_min, t_max, device)
                u_l = model(x_l, t_l, cond_bc)
                u_r = model(x_r, t_r, cond_bc)
                val_loss_bc = val_loss_bc + (u_l - u_r).pow(2).mean()

            val_loss_weak = val_loss_weak / val_batch_pdes
            val_loss_init = val_loss_init / val_batch_pdes
            val_loss_bc = val_loss_bc / val_batch_pdes
            val_loss = lam_weak * val_loss_weak + lam_init * val_loss_init + lam_bc * val_loss_bc
            print(
                f"[VPINN] epoch {epoch:5d}/{epochs} | val_total={val_loss.item():.3e} | "
                f"val_weak={val_loss_weak.item():.3e} | val_init={val_loss_init.item():.3e} | "
                f"val_bc={val_loss_bc.item():.3e}"
            )
            model.train()

    save_path = Path(vpinn_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    elapsed = time.perf_counter() - start_time
    print(f"Saved VPINN checkpoint to {save_path}")
    print(f"[VPINN] Training time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
