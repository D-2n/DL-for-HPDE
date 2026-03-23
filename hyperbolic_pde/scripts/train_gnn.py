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
from hyperbolic_pde.models.gnn import GridGNN


class GNNDataset(Dataset):
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
    parser = argparse.ArgumentParser(description="Train GridGNN on hyperbolic PDE dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    gnn_cfg = cfg["gnn"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    dataset = load_dataset(Path(data_cfg["path"]))
    train_idx, _ = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    train_idx, val_idx = split_train_val(train_idx, val_fraction, int(cfg.get("seed", 42)))

    train_data = GNNDataset(dataset.x, dataset.t, dataset.u0[train_idx], dataset.u[train_idx])
    loader = DataLoader(train_data, batch_size=int(gnn_cfg["batch_size"]), shuffle=True)
    val_loader = None
    if val_idx.size > 0:
        val_data = GNNDataset(dataset.x, dataset.t, dataset.u0[val_idx], dataset.u[val_idx])
        val_loader = DataLoader(val_data, batch_size=int(gnn_cfg["batch_size"]), shuffle=False)

    model = GridGNN(
        in_channels=3,
        out_channels=1,
        hidden=int(gnn_cfg["hidden"]),
        layers=int(gnn_cfg["layers"]),
    ).to(device)

    opt, use_lbfgs = make_optimizer(model.parameters(), gnn_cfg)

    epochs = int(gnn_cfg["epochs"])
    scheduler = None
    schedule = gnn_cfg.get("lr_schedule")
    total_steps = epochs * max(1, len(loader))
    if not use_lbfgs:
        if schedule == "cosine":
            lr_min = float(gnn_cfg.get("lr_min", 1.0e-5))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
        elif schedule == "step":
            lr_step = int(gnn_cfg.get("lr_step", 1000))
            lr_gamma = float(gnn_cfg.get("lr_gamma", 0.5))
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)

    step = 0
    model.train()
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        for inp, out in loader:
            step += 1
            inp = inp.to(device)
            out = out.to(device)
            def closure() -> torch.Tensor:
                opt.zero_grad(set_to_none=True)
                pred = model(inp)
                loss = (pred - out).pow(2).mean()
                loss.backward()
                grad_clip = gnn_cfg.get("grad_clip")
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
                    f"[GNN] epoch {epoch:3d}/{epochs} | step {step:5d}/{total_steps} | "
                    f"mse={loss.item():.3e} | lr={lr_now:.2e}"
                )

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for v_inp, v_out in val_loader:
                    v_inp = v_inp.to(device)
                    v_out = v_out.to(device)
                    v_pred = model(v_inp)
                    v_loss = (v_pred - v_out).pow(2).mean().item()
                    val_loss += v_loss * v_inp.size(0)
                    val_count += v_inp.size(0)
            val_mse = val_loss / max(1, val_count)
            print(f"[GNN] epoch {epoch:3d}/{epochs} | val_mse={val_mse:.3e}")
            model.train()

    save_path = Path(gnn_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    elapsed = time.perf_counter() - start_time
    print(f"Saved GNN checkpoint to {save_path}")
    print(f"[GNN] Training time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
