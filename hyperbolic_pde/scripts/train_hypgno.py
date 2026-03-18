from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset
from hyperbolic_pde.models.hypgno import HypGNO


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class HypGNODataset(Dataset):
    """Each __getitem__ returns (u0, t, u_target) with a *random* query time."""

    def __init__(self, u0: np.ndarray, u: np.ndarray, t: np.ndarray) -> None:
        self.u0 = torch.tensor(u0, dtype=torch.float32)
        self.u = torch.tensor(u, dtype=torch.float32)        # [N, nt, nx]
        self.t = torch.tensor(t, dtype=torch.float32)         # [nt]

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_idx = torch.randint(0, len(self.t), (1,)).item()
        return self.u0[idx], self.t[t_idx], self.u[idx, t_idx]


# --------------------------------------------------------------------------- #
# helpers (same style as existing scripts)
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


def make_optimizer(params, cfg: dict) -> tuple[torch.optim.Optimizer, bool]:
    name = str(cfg.get("optimizer", "adamw")).lower()
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
    return torch.optim.Adam(params, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0))), False


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def total_variation(u: torch.Tensor) -> torch.Tensor:
    """TV along spatial dim.  u: [B, nx] -> [B]."""
    return torch.abs(u[:, 1:] - u[:, :-1]).sum(dim=1)


def hypgno_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_state: float = 1.0,
    lambda_conservation: float = 1.0,
    lambda_tv: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss: L1 state + mass conservation + TV bound."""
    # L1 state loss (robust to discontinuities)
    loss_state = (pred - target).abs().mean()

    # mass conservation: penalise deviation of total mass (normalised by nx)
    nx = pred.shape[1]
    mass_pred = pred.sum(dim=1)
    mass_target = target.sum(dim=1)
    loss_mass = (mass_pred - mass_target).abs().mean() / nx

    # TV bound: penalise pred TV exceeding target TV (prevents oscillations)
    tv_pred = total_variation(pred)
    tv_target = total_variation(target)
    loss_tv = torch.clamp(tv_pred - tv_target, min=0.0).mean()

    loss = lambda_state * loss_state + lambda_conservation * loss_mass + lambda_tv * loss_tv

    info = {
        "state": loss_state.item(),
        "mass": loss_mass.item(),
        "tv": loss_tv.item(),
        "total": loss.item(),
    }
    return loss, info


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train HypGNO on hyperbolic PDE dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    model_cfg = cfg["hypgno"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training HypGNO on device: {device}")
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    dataset = load_dataset(Path(data_cfg["path"]))
    train_idx, _ = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    train_idx, val_idx = split_train_val(train_idx, val_fraction, int(cfg.get("seed", 42)))

    train_data = HypGNODataset(dataset.u0[train_idx], dataset.u[train_idx], dataset.t)
    loader = DataLoader(train_data, batch_size=int(model_cfg["batch_size"]), shuffle=True)
    val_loader = None
    if val_idx.size > 0:
        val_data = HypGNODataset(dataset.u0[val_idx], dataset.u[val_idx], dataset.t)
        val_loader = DataLoader(val_data, batch_size=int(model_cfg["batch_size"]), shuffle=False)

    model = HypGNO(
        stencil_k=int(model_cfg.get("stencil_k", 3)),
        d_latent=int(model_cfg.get("d_latent", 128)),
        d_hidden=int(model_cfg.get("d_hidden", 128)),
        d_time=int(model_cfg.get("d_time", 32)),
        n_layers=int(model_cfg.get("n_layers", 8)),
        activation=str(model_cfg.get("activation", "gelu")),
    ).to(device)

    x_grid = torch.tensor(dataset.x, dtype=torch.float32, device=device)  # [nx]

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[HypGNO] {n_params:,} trainable parameters")

    epochs = int(model_cfg["epochs"])
    opt, use_lbfgs = make_optimizer(model.parameters(), model_cfg)
    scheduler = None
    total_steps = epochs * max(1, len(loader))
    schedule = model_cfg.get("lr_schedule")
    if not use_lbfgs:
        if schedule == "cosine":
            lr_min = float(model_cfg.get("lr_min", 1.0e-5))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
        elif schedule == "step":
            lr_step = int(model_cfg.get("lr_step", 1000))
            lr_gamma = float(model_cfg.get("lr_gamma", 0.5))
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)

    lambda_state = float(model_cfg.get("lambda_state", 1.0))
    lambda_conservation = float(model_cfg.get("lambda_conservation", 1.0))
    lambda_tv = float(model_cfg.get("lambda_tv", 0.1))

    step = 0
    model.train()
    start_time = time.perf_counter()
    checkpoint_every = int(model_cfg.get("checkpoint_every", 20))

    # tracking for loss curves
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        epoch_loss_sum = 0.0
        epoch_count = 0
        for u0, t_query, u_target in loader:
            step += 1
            u0 = u0.to(device)
            t_query = t_query.to(device)
            u_target = u_target.to(device)

            def closure() -> torch.Tensor:
                opt.zero_grad(set_to_none=True)
                pred = model(u0, t_query, x_grid)
                loss, _ = hypgno_loss(pred, u_target, lambda_state, lambda_conservation, lambda_tv)
                loss.backward()
                grad_clip = model_cfg.get("grad_clip")
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

            epoch_loss_sum += loss.item() * u0.size(0)
            epoch_count += u0.size(0)

            if step % 50 == 0 or step == 1:
                lr_now = opt.param_groups[0]["lr"]
                # recompute with info dict for logging
                with torch.no_grad():
                    pred = model(u0, t_query, x_grid)
                    _, info = hypgno_loss(pred, u_target, lambda_state, lambda_conservation, lambda_tv)
                print(
                    f"[HypGNO] epoch {epoch:3d}/{epochs} | step {step:5d}/{total_steps} | "
                    f"L={info['total']:.3e}  state={info['state']:.3e}  mass={info['mass']:.3e}  "
                    f"tv={info['tv']:.3e} | lr={lr_now:.2e}"
                )

        train_losses.append(epoch_loss_sum / max(1, epoch_count))

        # validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for v_u0, v_t, v_u in val_loader:
                    v_u0 = v_u0.to(device)
                    v_t = v_t.to(device)
                    v_u = v_u.to(device)
                    v_pred = model(v_u0, v_t, x_grid)
                    v_l, _ = hypgno_loss(v_pred, v_u, lambda_state, lambda_conservation, lambda_tv)
                    val_loss += v_l.item() * v_u0.size(0)
                    val_count += v_u0.size(0)
            val_avg = val_loss / max(1, val_count)
            val_losses.append(val_avg)
            print(f"[HypGNO] epoch {epoch:3d}/{epochs} | val_loss={val_avg:.3e}")
            model.train()
        else:
            val_losses.append(float("nan"))

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_path = Path(model_cfg["save_path"])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_path.with_name(f"{save_path.stem}_epoch{epoch}{save_path.suffix}")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[HypGNO] Saved checkpoint to {ckpt_path}")

    save_path = Path(model_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    elapsed = time.perf_counter() - start_time
    print(f"Saved HypGNO checkpoint to {save_path}")
    print(f"[HypGNO] Training time: {elapsed:.2f}s")

    # --- plot train/val loss curves ---
    plot_dir = Path(model_cfg.get("plot_dir", "hyperbolic_pde/runs/plots/hypgno"))
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ep_range = list(range(1, len(train_losses) + 1))
    ax.plot(ep_range, train_losses, label="Train loss")
    if val_loader is not None and val_losses:
        ax.plot(ep_range, val_losses, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("HypGNO training curves")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    curve_path = plot_dir / "hypgno_loss_curves.png"
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[HypGNO] Saved loss curves to {curve_path}")


if __name__ == "__main__":
    main()
