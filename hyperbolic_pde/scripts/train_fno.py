from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
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
    parser = argparse.ArgumentParser(description="Train FNO on hyperbolic PDE dataset.")
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
    train_idx, _ = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))

    train_data = FNODataset(dataset.x, dataset.t, dataset.u0[train_idx], dataset.u[train_idx])
    loader = DataLoader(train_data, batch_size=int(fno_cfg["batch_size"]), shuffle=True)

    model = FNO2d(
        in_channels=3,
        out_channels=1,
        width=int(fno_cfg["width"]),
        modes_x=int(fno_cfg["modes_x"]),
        modes_t=int(fno_cfg["modes_t"]),
        layers=int(fno_cfg["layers"]),
    ).to(device)

    weight_decay = float(fno_cfg.get("weight_decay", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=float(fno_cfg["lr"]), weight_decay=weight_decay)
    scheduler = None
    schedule = fno_cfg.get("lr_schedule")
    if schedule == "cosine":
        lr_min = float(fno_cfg.get("lr_min", 1.0e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr_min)
    elif schedule == "step":
        lr_step = int(fno_cfg.get("lr_step", 1000))
        lr_gamma = float(fno_cfg.get("lr_gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)
    steps = int(fno_cfg["steps"])
    step = 0
    model.train()
    while step < steps:
        for inp, out in loader:
            step += 1
            inp = inp.to(device)
            out = out.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = (pred - out).pow(2).mean()
            loss.backward()
            grad_clip = fno_cfg.get("grad_clip")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            if scheduler is not None:
                scheduler.step()

            if step % 100 == 0 or step == 1:
                lr_now = opt.param_groups[0]["lr"]
                print(f"[FNO] step {step:5d} | mse={loss.item():.3e} | lr={lr_now:.2e}")

            if step >= steps:
                break

    save_path = Path(fno_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved FNO checkpoint to {save_path}")


if __name__ == "__main__":
    main()
