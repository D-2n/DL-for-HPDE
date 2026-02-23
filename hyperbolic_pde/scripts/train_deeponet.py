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

    train_data = DeepONetDataset(dataset.ic[train_idx], dataset.u[train_idx])
    loader = DataLoader(train_data, batch_size=int(deep_cfg["batch_size"]), shuffle=True)

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

    weight_decay = float(deep_cfg.get("weight_decay", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=float(deep_cfg["lr"]), weight_decay=weight_decay)

    epochs = int(deep_cfg["epochs"])
    scheduler = None
    schedule = deep_cfg.get("lr_schedule")
    total_steps = epochs * max(1, len(loader))
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

            opt.zero_grad(set_to_none=True)
            pred = model(branch, trunk_in)
            loss = (pred - target).pow(2).mean()
            loss.backward()

            grad_clip = deep_cfg.get("grad_clip")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            if scheduler is not None:
                scheduler.step()

            if step % 100 == 0 or step == 1:
                lr_now = opt.param_groups[0]["lr"]
                print(
                    f"[DeepONet] epoch {epoch:3d}/{epochs} | step {step:5d}/{total_steps} | "
                    f"mse={loss.item():.3e} | lr={lr_now:.2e}"
                )

    save_path = Path(deep_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    elapsed = time.perf_counter() - start_time
    print(f"Saved DeepONet checkpoint to {save_path}")
    print(f"[DeepONet] Training time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
