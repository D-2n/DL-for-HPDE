"""Train HypNO-ST v2 — revised edge features and physics gating."""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.utils.runtime import apply_runtime_overrides, resolve_config_path
from hyperbolic_pde.data.fvm import load_dataset
from hyperbolic_pde.models.hypno_st2 import HypNO_ST2, precompute_lwr_edge_features_v2


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class HypNODataset(Dataset):
    """Each __getitem__ returns (u0, u_full, edge_feats).

    ``edge_feats`` contains the 14 static LWR edge features (v2) precomputed
    once at construction time.  Shape per sample: ``[nx, 2k+1, 14]``.
    """

    def __init__(
        self,
        u0: np.ndarray,
        u: np.ndarray,
        x: np.ndarray,
        stencil_k: int,
        radius_x: float | None = None,
        skip_edge_feats: bool = False,
    ) -> None:
        self.u0 = torch.tensor(u0, dtype=torch.float32)
        self.u = torch.tensor(u, dtype=torch.float32)
        if skip_edge_feats:
            self.edge_feats = None
        else:
            self.edge_feats = precompute_lwr_edge_features_v2(
                self.u0,
                torch.tensor(x, dtype=torch.float32),
                stencil_k,
                radius_x=radius_x,
            )

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ef = self.edge_feats[idx] if self.edge_feats is not None else torch.empty(0)
        return self.u0[idx], self.u[idx], ef


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


def create_run_dir(base: str = "hyperbolic_pde/runs/hypno_st2") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(
    run_dir: Path,
    cfg: dict,
    model: torch.nn.Module,
    config_path: str,
) -> None:
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with (run_dir / "architecture.txt").open("w", encoding="utf-8") as f:
        f.write(f"Model: HypNO-ST2\n")
        f.write(f"Trainable parameters: {n_params:,}\n")
        f.write(f"Config source: {config_path}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"\n{'='*60}\n\n")
        f.write(str(model))
        f.write(f"\n\n{'='*60}\n")
        f.write(f"Parameter breakdown:\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"  {name}: {list(param.shape)} = {param.numel():,}\n")


def make_optimizer(params, cfg: dict) -> tuple[torch.optim.Optimizer, bool]:
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("lr", 1.0e-3))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0))), False
    return torch.optim.Adam(params, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0))), False


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def total_variation(u: torch.Tensor) -> torch.Tensor:
    return torch.abs(u[:, :, 1:] - u[:, :, :-1]).sum(dim=2)


def hypno_pinn_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    u_coarse: torch.Tensor,
    u_hats: list[torch.Tensor] | None = None,
    lambda_state: float = 1.0,
    lambda_conservation: float = 1.0,
    lambda_tv: float = 0.0,
    lambda_pinn: float = 0.1,
    lambda_probe: float = 0.0,
    shock_weighted: bool = False,
    shock_alpha: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if shock_weighted:
        du_dx = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        du_dx = torch.nn.functional.pad(du_dx, (0, 1), mode='replicate')
        w = 1.0 + (shock_alpha - 1.0) * du_dx / (du_dx.amax(dim=(1, 2), keepdim=True) + 1e-8)
        loss_state = (w * (pred - target).abs()).mean()
    else:
        loss_state = (pred - target).abs().mean()

    nx = pred.shape[2]
    mass_pred = pred.sum(dim=2)
    mass_target = target.sum(dim=2)
    loss_mass = (mass_pred - mass_target).abs().mean() / nx

    tv_pred = total_variation(pred)
    tv_target = total_variation(target)
    loss_tv = torch.clamp(tv_pred - tv_target, min=0.0).mean()

    loss_pinn = (u_coarse - target).abs().mean()

    # deep supervision on state_probe u_hats (one per MP layer)
    if u_hats is not None and len(u_hats) > 0 and lambda_probe > 0.0:
        loss_probe = sum((uh - target).abs().mean() for uh in u_hats) / len(u_hats)
    else:
        loss_probe = torch.zeros((), device=pred.device)

    loss = (
        lambda_state * loss_state
        + lambda_conservation * loss_mass
        + lambda_tv * loss_tv
        + lambda_pinn * loss_pinn
        + lambda_probe * loss_probe
    )

    info = {
        "state": loss_state.item(),
        "mass": loss_mass.item(),
        "tv": loss_tv.item(),
        "pinn": loss_pinn.item(),
        "probe": float(loss_probe.item()) if torch.is_tensor(loss_probe) else float(loss_probe),
        "total": loss.item(),
    }
    return loss, info


# --------------------------------------------------------------------------- #
# logging + GPU monitoring
# --------------------------------------------------------------------------- #
def setup_logging(run_dir: Path) -> logging.Logger:
    log = logging.getLogger("hypno_st2")
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
    parser = argparse.ArgumentParser(description="Train HypNO-ST2 on hyperbolic PDE dataset.")
    parser.add_argument(
        "--config", type=str,
        default=str(resolve_config_path(ROOT / "configs")),
    )
    args = parser.parse_args()
    print('--- STARTING HYPNO_ST2 TRAINING ---')
    cfg = load_config(Path(args.config))
    cfg = apply_runtime_overrides(cfg)
    data_cfg = cfg["data"]
    # prefer hypno_st2 config section, fall back to hypno_st
    model_cfg = cfg.get("hypno_st2", cfg.get("hypno_st"))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    run_dir = create_run_dir()
    log = setup_logging(run_dir)
    log.info(f"Run directory: {run_dir}")
    log.info(f"Device: {device}")
    log.info(f"GPU: {gpu_status()}")

    dataset = load_dataset(Path(data_cfg["path"]))
    train_idx, _ = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    train_idx, val_idx = split_train_val(train_idx, val_fraction, int(cfg.get("seed", 42)))

    _rx = model_cfg.get("radius_x", None)
    _rt = model_cfg.get("radius_t", None)
    radius_x = float(_rx) if _rx is not None else None
    radius_t = float(_rt) if _rt is not None else None
    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"

    train_data = HypNODataset(
        dataset.u0[train_idx], dataset.u[train_idx],
        dataset.x, stencil_k_x, radius_x=radius_x, skip_edge_feats=skip_ef,
    )
    loader = DataLoader(train_data, batch_size=int(model_cfg["batch_size"]), shuffle=True)
    val_loader = None
    if val_idx.size > 0:
        val_data = HypNODataset(
            dataset.u0[val_idx], dataset.u[val_idx],
            dataset.x, stencil_k_x, radius_x=radius_x, skip_edge_feats=skip_ef,
        )
        val_loader = DataLoader(val_data, batch_size=int(model_cfg["batch_size"]), shuffle=False)

    model = HypNO_ST2(
        stencil_k_x=stencil_k_x,
        stencil_k_t=int(model_cfg.get("stencil_k_t", 2)),
        d_latent=int(model_cfg.get("d_latent", 128)),
        d_hidden=int(model_cfg.get("d_hidden", 128)),
        n_layers=int(model_cfg.get("n_layers", 6)),
        activation=str(model_cfg.get("activation", "gelu")),
        shock_delta=float(model_cfg.get("shock_delta", 0.01)),
        shock_threshold=float(model_cfg.get("shock_threshold", 0.1)),
        causal_temporal=bool(model_cfg.get("causal_temporal", True)),
        radius_x=radius_x,
        radius_t=radius_t,
        shock_mode=str(model_cfg.get("shock_mode", "pinn")),
        weno_eps=float(model_cfg.get("weno_eps", 1e-6)),
        weno_p=float(model_cfg.get("weno_p", 2.0)),
        unified_mp=bool(model_cfg.get("unified_mp", False)),
        readout=str(model_cfg.get("readout", "gelu")),
        encoder_scaling=str(model_cfg.get("encoder_scaling", "gate_net")),
        encoder_type=encoder_type,
        skip=bool(model_cfg.get("skip", True)),
        detector_path=model_cfg.get("detector_path", None),
        detector_cfg=cfg.get("shock_detector", {}),
    ).to(device)

    x_grid = torch.tensor(dataset.x, dtype=torch.float32, device=device)
    t_grid = torch.tensor(dataset.t, dtype=torch.float32, device=device)

    resume_path = model_cfg.get("resume_from", None)
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        log.info(f"Resumed weights from {resume_path}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"{n_params:,} trainable parameters")

    save_run_metadata(run_dir, cfg, model, args.config)

    epochs = int(model_cfg["epochs"])
    opt, _ = make_optimizer(model.parameters(), model_cfg)
    scheduler = None
    total_steps = epochs * max(1, len(loader))
    schedule = model_cfg.get("lr_schedule")
    if schedule == "cosine":
        lr_min = float(model_cfg.get("lr_min", 1.0e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
    elif schedule == "step":
        lr_step = int(model_cfg.get("lr_step", 1000))
        lr_gamma = float(model_cfg.get("lr_gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)

    lambda_state = float(model_cfg.get("lambda_state", 1.0))
    lambda_conservation = float(model_cfg.get("lambda_conservation", 1.0))
    lambda_tv = float(model_cfg.get("lambda_tv", 0.0))
    lambda_pinn = float(model_cfg.get("lambda_pinn", 0.1))
    lambda_probe = float(model_cfg.get("lambda_probe", 0.0))
    shock_weighted = bool(model_cfg.get("shock_weighted", False))
    shock_alpha = float(model_cfg.get("shock_alpha", 5.0))

    step = 0
    model.train()
    start_time = time.perf_counter()
    checkpoint_every = int(model_cfg.get("checkpoint_every", 50))
    val_every = int(model_cfg.get("val_every", 5))

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        epoch_loss_sum = 0.0
        epoch_count = 0
        for u0, u_full, edge_feats in loader:
            step += 1
            u0 = u0.to(device)
            u_full = u_full.to(device)
            ef = edge_feats.to(device) if not skip_ef else None

            opt.zero_grad(set_to_none=True)
            pred, u_coarse, _, u_hats = model(u0, x_grid, t_grid, edge_feats_pre=ef)
            loss, _ = hypno_pinn_loss(
                pred, u_full, u_coarse,
                u_hats=u_hats,
                lambda_state=lambda_state,
                lambda_conservation=lambda_conservation,
                lambda_tv=lambda_tv,
                lambda_pinn=lambda_pinn,
                lambda_probe=lambda_probe,
                shock_weighted=shock_weighted,
                shock_alpha=shock_alpha,
            )
            loss.backward()
            grad_clip = model_cfg.get("grad_clip")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss_sum += loss.item() * u0.size(0)
            epoch_count += u0.size(0)

        train_losses.append(epoch_loss_sum / max(1, epoch_count))
        lr_now = opt.param_groups[0]["lr"]
        log.info(
            f"epoch {epoch:3d}/{epochs} | train_loss={train_losses[-1]:.3e} | lr={lr_now:.2e} | {gpu_status()}"
        )
        for h in log.handlers:
            h.flush()

        # validation (every val_every epochs)
        if val_loader is not None and epoch % val_every == 0:
            model.eval()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_loss = 0.0
            val_count = 0
            with torch.inference_mode():
                for v_u0, v_u, v_ef in val_loader:
                    v_u0 = v_u0.to(device)
                    v_u = v_u.to(device)
                    v_ef = v_ef.to(device) if not skip_ef else None
                    v_pred, v_coarse, _, v_u_hats = model(v_u0, x_grid, t_grid, edge_feats_pre=v_ef)
                    v_l, _ = hypno_pinn_loss(
                        v_pred, v_u, v_coarse,
                        u_hats=v_u_hats,
                        lambda_state=lambda_state,
                        lambda_conservation=lambda_conservation,
                        lambda_tv=lambda_tv,
                        lambda_pinn=lambda_pinn,
                        lambda_probe=lambda_probe,
                        shock_weighted=shock_weighted,
                        shock_alpha=shock_alpha,
                    )
                    val_loss += v_l.item() * v_u0.size(0)
                    val_count += v_u0.size(0)
            val_avg = val_loss / max(1, val_count)
            val_losses.append(val_avg)
            log.info(f"epoch {epoch:3d}/{epochs} | val_loss={val_avg:.3e} | {gpu_status()}")
            for h in log.handlers:
                h.flush()
            model.train()

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt_path = run_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            log.info(f"Saved checkpoint to {ckpt_path}")

    final_path = run_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    save_path = Path(model_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    elapsed = time.perf_counter() - start_time
    log.info(f"Saved final model to {final_path}")
    log.info(f"Training time: {elapsed:.2f}s")

    fig, ax = plt.subplots(figsize=(8, 5))
    ep_range = list(range(1, len(train_losses) + 1))
    ax.plot(ep_range, train_losses, label="Train loss")
    if val_loader is not None and val_losses:
        val_epochs = list(range(val_every, epochs + 1, val_every))[:len(val_losses)]
        ax.plot(val_epochs, val_losses, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("HypNO-ST2 training curves")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    curve_path = run_dir / "loss_curves.png"
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved loss curves to {curve_path}")

    latest_path = Path("hyperbolic_pde/runs/hypno_st2/latest_run.txt")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir), encoding="utf-8")
    log.info(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
