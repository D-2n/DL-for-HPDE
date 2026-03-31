"""Pre-train the standalone PINN shock detector.

Proper PINN training: PDE residual loss (autograd) + IC loss + optional data loss.
No ground truth needed at interior points — the PDE itself is the supervision.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import yaml
from hyperbolic_pde.utils.runtime import apply_runtime_overrides

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset
from hyperbolic_pde.models.shock_detector import ShockDetector


# --------------------------------------------------------------------------- #
# dataset — only need u0 for pure PINN, but keep u for optional data loss + val
# --------------------------------------------------------------------------- #
class DetectorDataset(Dataset):
    def __init__(self, u0: np.ndarray, u: np.ndarray) -> None:
        self.u0 = torch.tensor(u0, dtype=torch.float32)
        self.u = torch.tensor(u, dtype=torch.float32)

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.u0[idx], self.u[idx]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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


def create_run_dir(base: str = "hyperbolic_pde/runs/shock_detector") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> logging.Logger:
    log = logging.getLogger("shock_detector")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(run_dir / "train.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    return log


def gpu_status() -> str:
    if not torch.cuda.is_available():
        return "no-gpu"
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        temp, mem_used, mem_total, util = out.split(", ")
        return f"temp={temp}C  mem={mem_used}/{mem_total}MB  util={util}%"
    except Exception:
        return "nvidia-smi-failed"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train PINN shock detector.")
    parser.add_argument("--config", type=str, default="hyperbolic_pde/configs/hyperbolic_pde.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    cfg = apply_runtime_overrides(cfg)
    data_cfg = cfg["data"]
    det_cfg = cfg.get("shock_detector", {})
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    run_dir = create_run_dir()
    log = setup_logging(run_dir)
    log.info(f"Run directory: {run_dir}")
    log.info(f"Device: {device}")
    log.info(f"GPU: {gpu_status()}")

    # data
    dataset = load_dataset(Path(data_cfg["path"]))
    train_idx, _ = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    train_idx, val_idx = split_train_val(train_idx, val_fraction, int(cfg.get("seed", 42)))

    train_data = DetectorDataset(dataset.u0[train_idx], dataset.u[train_idx])
    val_data = DetectorDataset(dataset.u0[val_idx], dataset.u[val_idx]) if val_idx.size > 0 else None

    batch_size = int(det_cfg.get("batch_size", 16))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data else None

    x_grid = torch.tensor(dataset.x, dtype=torch.float32, device=device)  # [nx]
    t_grid = torch.tensor(dataset.t, dtype=torch.float32, device=device)  # [nt]
    nx = x_grid.shape[0]
    nt = t_grid.shape[0]
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    t_max = float(t_grid[-1])

    # model
    model = ShockDetector(
        d_latent=int(det_cfg.get("d_latent", 64)),
        d_hidden=int(det_cfg.get("d_hidden", 128)),
        n_layers=int(det_cfg.get("n_layers", 6)),
        activation=str(det_cfg.get("activation", "tanh")),
        ic_points=int(det_cfg.get("ic_points", nx)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {n_params:,}")

    # save config
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    shutil.copy2(args.config, run_dir / "config_source.yaml")

    # training config
    epochs = int(det_cfg.get("epochs", 300))
    lr = float(det_cfg.get("lr", 1e-3))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=float(det_cfg.get("weight_decay", 1e-4)),
    )
    grad_clip = det_cfg.get("grad_clip", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    lr_min = float(det_cfg.get("lr_min", 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)

    checkpoint_every = int(det_cfg.get("checkpoint_every", 20))
    lambda_pde = float(det_cfg.get("lambda_pde", 1.0))
    lambda_ic = float(det_cfg.get("lambda_ic", 10.0))
    lambda_data = float(det_cfg.get("lambda_data", 0.0))  # optional supervised loss
    n_colloc = int(det_cfg.get("n_collocation", 2048))

    # training loop
    train_losses, val_losses = [], []
    total_steps = epochs * len(loader)

    log.info(f"Training for {epochs} epochs, {len(loader)} batches/epoch, {total_steps} total steps")
    log.info(f"Loss weights: lambda_pde={lambda_pde}, lambda_ic={lambda_ic}, lambda_data={lambda_data}")
    log.info(f"Collocation points per batch: {n_colloc}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0

        for step_in_epoch, (u0_b, u_b) in enumerate(loader, 1):
            u0_b = u0_b.to(device)  # [B, nx]
            u_b = u_b.to(device)    # [B, nt, nx]
            B = u0_b.shape[0]

            z = model.encode_ic(u0_b)  # [B, d_latent]

            # --- IC loss: u(x, 0) = u0(x) ---
            x_ic = x_grid.unsqueeze(0).expand(B, -1)          # [B, nx]
            t_ic = torch.zeros(B, nx, device=device)           # [B, nx] all zeros
            u_at_0 = model.predict_pointwise(x_ic, t_ic, z)   # [B, nx]
            loss_ic = F.mse_loss(u_at_0, u0_b)

            # --- PDE loss at random collocation points ---
            x_col = (x_min + (x_max - x_min) * torch.rand(B, n_colloc, device=device)).requires_grad_(True)
            t_col = (t_max * torch.rand(B, n_colloc, device=device)).clamp(min=1e-4).requires_grad_(True)
            residual = model.pde_residual(x_col, t_col, z)    # [B, n_colloc]
            loss_pde = (residual ** 2).mean()

            # --- optional data loss ---
            loss_data = torch.tensor(0.0, device=device)
            if lambda_data > 0:
                u_pred = model.predict_grid(u0_b, x_grid, t_grid)
                loss_data = F.l1_loss(u_pred, u_b)

            loss = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_data * loss_data

            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * B
            epoch_count += B

            step = (epoch - 1) * len(loader) + step_in_epoch
            if step % 50 == 0 or step == 1:
                gpu = gpu_status() if step % 200 == 0 or step == 1 else ""
                msg = (f"epoch {epoch:3d}/{epochs} | step {step:5d}/{total_steps} | "
                       f"L={loss.item():.4e}  pde={loss_pde.item():.4e}  ic={loss_ic.item():.4e}")
                if lambda_data > 0:
                    msg += f"  data={loss_data.item():.4e}"
                msg += f" | lr={optimizer.param_groups[0]['lr']:.2e}"
                if gpu:
                    msg += f" | {gpu}"
                log.info(msg)
                for h in log.handlers:
                    h.flush()

        train_losses.append(epoch_loss / max(1, epoch_count))
        scheduler.step()

        # validation: check prediction quality against ground truth
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for u0_b, u_b in val_loader:
                    u0_b, u_b = u0_b.to(device), u_b.to(device)
                    u_pred = model.predict_grid(u0_b, x_grid, t_grid)
                    val_loss += F.l1_loss(u_pred, u_b, reduction="sum").item()
                    val_count += u_b.numel()
            val_avg = val_loss / max(1, val_count)
            val_losses.append(val_avg)
            log.info(f"epoch {epoch:3d}/{epochs} | val_L1={val_avg:.4e} | {gpu_status()}")
            for h in log.handlers:
                h.flush()
            model.train()

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt = run_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt)
            log.info(f"Saved checkpoint to {ckpt}")

    # save final
    final_path = run_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    save_path = Path(det_cfg.get("save_path", "hyperbolic_pde/runs/shock_detector.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_path, save_path)
    log.info(f"Saved final model to {final_path} and {save_path}")

    # plot training curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(1, epochs + 1), train_losses, label="train loss")
    if val_losses:
        ax.plot(range(1, epochs + 1), val_losses, label="val L1 (vs FVM)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("PINN Shock Detector Training")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    log.info("Done.")


if __name__ == "__main__":
    main()
