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
from hyperbolic_pde.models.fluxgnn import FluxGNN1D


class FluxGNNDataset(Dataset):
    def __init__(self, u0: np.ndarray, u: np.ndarray) -> None:
        self.u0 = torch.tensor(u0, dtype=torch.float32)
        self.u = torch.tensor(u, dtype=torch.float32)

    def __len__(self) -> int:
        return self.u.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.u0[idx], self.u[idx]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FluxGNN on hyperbolic PDE dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    flux_cfg = cfg["fluxgnn"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    dataset = load_dataset(Path(data_cfg["path"]))
    train_idx, _ = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    train_idx, val_idx = split_train_val(train_idx, val_fraction, int(cfg.get("seed", 42)))

    train_data = FluxGNNDataset(dataset.u0[train_idx], dataset.u[train_idx])
    loader = DataLoader(train_data, batch_size=int(flux_cfg["batch_size"]), shuffle=True)
    val_loader = None
    if val_idx.size > 0:
        val_data = FluxGNNDataset(dataset.u0[val_idx], dataset.u[val_idx])
        val_loader = DataLoader(val_data, batch_size=int(flux_cfg["batch_size"]), shuffle=False)

    model = FluxGNN1D(
        hidden=int(flux_cfg["hidden"]),
        layers=int(flux_cfg["layers"]),
        activation=str(flux_cfg.get("activation", "gelu")),
        latent_dim=flux_cfg.get("latent_dim"),
        flux_hidden=flux_cfg.get("flux_hidden"),
        use_base_flux=bool(flux_cfg.get("use_base_flux", True)),
        base_flux_weight=float(flux_cfg.get("base_flux_weight", 0.5)),
        flux_scale=float(flux_cfg.get("flux_scale", 0.25)),
    ).to(device)

    epochs = flux_cfg.get("epochs")
    if epochs is None:
        if "steps" in flux_cfg:
            steps = int(flux_cfg["steps"])
            steps_per_epoch = max(1, len(loader))
            epochs = max(1, math.ceil(steps / steps_per_epoch))
            print(
                f"[FluxGNN] config uses steps={steps}; converting to epochs={epochs} "
                f"(steps/epoch={steps_per_epoch})."
            )
        else:
            raise KeyError("fluxgnn config must define 'epochs' (or legacy 'steps')")
    epochs = int(epochs)

    weight_decay = float(flux_cfg.get("weight_decay", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=float(flux_cfg["lr"]), weight_decay=weight_decay)
    scheduler = None
    schedule = flux_cfg.get("lr_schedule")
    total_steps = epochs * max(1, len(loader))
    if schedule == "cosine":
        lr_min = float(flux_cfg.get("lr_min", 1.0e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
    elif schedule == "step":
        lr_step = int(flux_cfg.get("lr_step", 1000))
        lr_gamma = float(flux_cfg.get("lr_gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)

    dx = float(dataset.x[1] - dataset.x[0])
    dt = float(dataset.t[1] - dataset.t[0])
    n_steps = int(dataset.u.shape[1])
    boundary = str(data_cfg.get("boundary", "ghost"))

    step = 0
    model.train()
    start_time = time.perf_counter()
    checkpoint_every = int(flux_cfg.get("checkpoint_every", 20))
    for epoch in range(1, epochs + 1):
        for u0, u in loader:
            step += 1
            u0 = u0.to(device)
            u = u.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(u0, dt, dx, n_steps, boundary)
            loss = (pred - u).pow(2).mean()
            loss.backward()
            grad_clip = flux_cfg.get("grad_clip")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            if scheduler is not None:
                scheduler.step()

            if step % 50 == 0 or step == 1:
                lr_now = opt.param_groups[0]["lr"]
                print(
                    f"[FluxGNN] epoch {epoch:3d}/{epochs} | step {step:5d}/{total_steps} | "
                    f"mse={loss.item():.3e} | lr={lr_now:.2e}"
                )

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for v_u0, v_u in val_loader:
                    v_u0 = v_u0.to(device)
                    v_u = v_u.to(device)
                    v_pred = model(v_u0, dt, dx, n_steps, boundary)
                    v_loss = (v_pred - v_u).pow(2).mean().item()
                    val_loss += v_loss * v_u0.size(0)
                    val_count += v_u0.size(0)
            val_mse = val_loss / max(1, val_count)
            print(f"[FluxGNN] epoch {epoch:3d}/{epochs} | val_mse={val_mse:.3e}")
            model.train()

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_path = Path(flux_cfg["save_path"])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_path.with_name(f"{save_path.stem}_epoch{epoch}{save_path.suffix}")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[FluxGNN] Saved checkpoint to {ckpt_path}")

    save_path = Path(flux_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    elapsed = time.perf_counter() - start_time
    print(f"Saved FluxGNN checkpoint to {save_path}")
    print(f"[FluxGNN] Training time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
