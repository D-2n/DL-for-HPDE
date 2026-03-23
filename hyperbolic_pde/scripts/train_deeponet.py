from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepONet on hyperbolic PDE dataset.")
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
    train_idx, _ = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    train_idx, val_idx = split_train_val(train_idx, val_fraction, int(cfg.get("seed", 42)))

    train_data = DeepONetDataset(dataset.ic[train_idx], dataset.u[train_idx])
    loader = DataLoader(train_data, batch_size=int(deep_cfg["batch_size"]), shuffle=True)
    val_loader = None
    if val_idx.size > 0:
        val_data = DeepONetDataset(dataset.ic[val_idx], dataset.u[val_idx])
        val_loader = DataLoader(val_data, batch_size=int(deep_cfg["batch_size"]), shuffle=False)

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

    opt, use_lbfgs = make_optimizer(model.parameters(), deep_cfg)

    epochs = int(deep_cfg["epochs"])
    scheduler = None
    schedule = deep_cfg.get("lr_schedule")
    total_steps = epochs * max(1, len(loader))
    if not use_lbfgs:
        if schedule == "cosine":
            lr_min = float(deep_cfg.get("lr_min", 1.0e-5))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
        elif schedule == "step":
            lr_step = int(deep_cfg.get("lr_step", 1000))
            lr_gamma = float(deep_cfg.get("lr_gamma", 0.5))
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)

    step = 0
    model.train()
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        for branch, target in loader:
            step += 1
            branch = branch.to(device)
            target = target.to(device)

            def closure() -> torch.Tensor:
                opt.zero_grad(set_to_none=True)
                pred = model(branch, trunk_in)
                loss = (pred - target).pow(2).mean()
                loss.backward()
                grad_clip = deep_cfg.get("grad_clip")
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                return loss

            if use_lbfgs:
                loss = opt.step(closure)
            else:
                loss = closure()
                opt.step()
            if scheduler is not None:
                scheduler.step()

            if step % 100 == 0 or step == 1:
                lr_now = opt.param_groups[0]["lr"]
                print(
                    f"[DeepONet] epoch {epoch:3d}/{epochs} | step {step:5d}/{total_steps} | "
                    f"mse={loss.item():.3e} | lr={lr_now:.2e}"
                )

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for v_branch, v_target in val_loader:
                    v_branch = v_branch.to(device)
                    v_target = v_target.to(device)
                    v_pred = model(v_branch, trunk_in)
                    v_loss = (v_pred - v_target).pow(2).mean().item()
                    val_loss += v_loss * v_branch.size(0)
                    val_count += v_branch.size(0)
            val_mse = val_loss / max(1, val_count)
            print(f"[DeepONet] epoch {epoch:3d}/{epochs} | val_mse={val_mse:.3e}")
            model.train()

    save_path = Path(deep_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    elapsed = time.perf_counter() - start_time
    print(f"Saved DeepONet checkpoint to {save_path}")
    print(f"[DeepONet] Training time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
