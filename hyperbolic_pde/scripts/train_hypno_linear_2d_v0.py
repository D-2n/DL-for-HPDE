"""Train HypNO_Linear_2D_v0 on the 2D linear-advection v0 dataset.

Loss = lambda_state * MAE(u_pred, truth)
     + lambda_probe * mean_k MAE(u_hats[k], truth)        (deep supervision)
     + lambda_conservation * mean | M_pred(t) - M_ic |    (Option A: temporal
                                                           mass conservation)

For pure linear advection with replicate (ghost) BC the analytical solution
exactly conserves mass *up to outflow*, so lambda_conservation defaults to 0
in the v0 config — the term is kept for parity with the LWR pipeline.

Usage from repo root:
    python hyperbolic_pde/scripts/train_hypno_linear_2d_v0.py
    python hyperbolic_pde/scripts/train_hypno_linear_2d_v0.py --config <yaml>
    python hyperbolic_pde/scripts/train_hypno_linear_2d_v0.py --resume_run <run_dir>
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from hyperbolic_pde.data.fvm_2d import load_dataset_2d
from hyperbolic_pde.models.hypno_linear_2d_v0 import HypNO_Linear_2D_v0


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class Linear2DDataset(Dataset):
    """Returns (u0, u_full): u0 [ny, nx], u_full [nt, ny, nx]."""

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
def split_indices(n: int, train_fraction: float, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_fraction * n)
    n_val = int(val_fraction * n)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def create_run_dir(base: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> logging.Logger:
    log = logging.getLogger("hypno_linear_2d_v0")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(run_dir / "train.log", mode="a", encoding="utf-8")
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
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        mem_used, mem_total, util = out.split(", ")
        return f"mem={mem_used}/{mem_total}MB util={util}%"
    except Exception:
        return "nvidia-smi-failed"


def _find_latest_checkpoint(run_dir: Path) -> tuple[Path | None, int]:
    candidates: list[tuple[int, Path]] = []
    for p in run_dir.glob("checkpoint_epoch*.pt"):
        try:
            ep = int(p.stem.replace("checkpoint_epoch", ""))
        except ValueError:
            continue
        candidates.append((ep, p))
    if not candidates:
        return None, 0
    epoch, latest = max(candidates, key=lambda kv: kv[0])
    return latest, epoch


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def linear_2d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    u0: torch.Tensor,
    u_hats: list[torch.Tensor],
    lambda_state: float,
    lambda_probe: float,
    lambda_conservation: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_state = (pred - target).abs().mean()

    if u_hats and lambda_probe > 0.0:
        loss_probe = sum((uh - target).abs().mean() for uh in u_hats) / len(u_hats)
    else:
        loss_probe = torch.zeros((), device=pred.device)

    m_pred = pred.sum(dim=(2, 3))
    m_ic = u0.sum(dim=(1, 2)).unsqueeze(1)
    loss_cons = (m_pred - m_ic).abs().mean()

    loss = (
        lambda_state * loss_state
        + lambda_probe * loss_probe
        + lambda_conservation * loss_cons
    )
    info = {
        "state": loss_state.item(),
        "probe": float(loss_probe.item()) if torch.is_tensor(loss_probe) else 0.0,
        "cons": loss_cons.item(),
        "total": loss.item(),
    }
    return loss, info


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train HypNO_Linear_2D_v0 on the 2D linear-advection dataset.")
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "hyperbolic_pde_linear_2d_v0.yaml"),
    )
    parser.add_argument(
        "--resume_run", type=str, default=None,
        help="Path to a previous run directory. Resumes from its latest checkpoint.",
    )
    args = parser.parse_args()
    print("--- STARTING HYPNO_LINEAR_2D_v0 TRAINING ---")

    resume_run_dir = Path(args.resume_run) if args.resume_run else None
    if resume_run_dir and (resume_run_dir / "config.yaml").exists():
        with (resume_run_dir / "config.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"Resuming from run: {resume_run_dir}")
    else:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["hypno_linear_2d_v0"]
    print(f"[train] config={args.config}")

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    if resume_run_dir:
        run_dir = resume_run_dir
    else:
        run_dir = create_run_dir(model_cfg.get("run_base", "hyperbolic_pde/runs/hypno_linear_2d_v0"))
    log = setup_logging(run_dir)
    log.info(f"Run directory: {run_dir}")
    log.info(f"Device: {device} | {gpu_status()}")

    # ---- data ----
    data_path = Path(data_cfg["path"])
    if not data_path.is_absolute():
        data_path = ROOT.parent / data_path
    bundle = load_dataset_2d(data_path)
    n = bundle.u.shape[0]
    train_idx, val_idx, _ = split_indices(
        n, float(data_cfg["train_fraction"]), float(data_cfg["val_fraction"]), seed,
    )
    log.info(f"Dataset {bundle.u.shape}: {len(train_idx)} train / {len(val_idx)} val")

    train_data = Linear2DDataset(bundle.u0[train_idx], bundle.u[train_idx])
    loader = DataLoader(train_data, batch_size=int(model_cfg["batch_size"]), shuffle=True)
    val_loader = None
    if val_idx.size > 0:
        val_data = Linear2DDataset(bundle.u0[val_idx], bundle.u[val_idx])
        val_loader = DataLoader(val_data, batch_size=int(model_cfg["batch_size"]), shuffle=False)

    x_grid = torch.tensor(bundle.x, dtype=torch.float32, device=device)
    y_grid = torch.tensor(bundle.y, dtype=torch.float32, device=device)
    t_grid = torch.tensor(bundle.t, dtype=torch.float32, device=device)

    # ---- model ----
    # `h_s` (the spatial-scale denominator in the char-alignment gate) is the
    # *training-grid* cell scale, not the fine-grid scale: the model only ever
    # sees the coarse arrays, so its gates must be calibrated against coarse dx.
    dx = float(bundle.x[1] - bundle.x[0])
    dy = float(bundle.y[1] - bundle.y[0])
    h_s = math.sqrt(dx * dy)
    a = float(data_cfg.get("a", 1.0))
    b = float(data_cfg.get("b", 0.7))
    log.info(f"grid dx={dx:.4e} dy={dy:.4e} h_s={h_s:.4e} | (a, b)=({a}, {b})")

    _dhn = model_cfg.get("d_hidden_nonadj", None)
    model = HypNO_Linear_2D_v0(
        stencil_k_x=int(model_cfg.get("stencil_k_x", 3)),
        stencil_k_y=int(model_cfg.get("stencil_k_y", 3)),
        stencil_k_t=int(model_cfg.get("stencil_k_t", 3)),
        d_latent=int(model_cfg.get("d_latent", 64)),
        d_hidden=int(model_cfg.get("d_hidden", 64)),
        d_hidden_nonadj=int(_dhn) if _dhn is not None else None,
        n_layers=int(model_cfg.get("n_layers", 6)),
        activation=str(model_cfg.get("activation", "gelu")),
        causal_temporal=bool(model_cfg.get("causal_temporal", True)),
        readout=str(model_cfg.get("readout", "gelu")),
        encoder_type=str(model_cfg.get("encoder_type", "gnn")),
        skip=bool(model_cfg.get("skip", True)),
        use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
        a=a, b=b, h_s=h_s,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"{n_params:,} trainable parameters")
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # ---- resume ----
    start_epoch = 1
    if resume_run_dir:
        ckpt_path, resumed_epoch = _find_latest_checkpoint(resume_run_dir)
        if ckpt_path:
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            start_epoch = resumed_epoch + 1
            log.info(f"Resumed from {ckpt_path} (epoch {resumed_epoch}) -> epoch {start_epoch}")
        else:
            log.warning(f"No checkpoints in {resume_run_dir}; starting from scratch")

    # ---- optimizer / schedule ----
    epochs = int(model_cfg["epochs"])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = None
    total_steps = epochs * max(1, len(loader))
    if model_cfg.get("lr_schedule") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=float(model_cfg.get("lr_min", 1e-5)),
        )
        for _ in range((start_epoch - 1) * len(loader)):
            scheduler.step()

    lambda_state = float(model_cfg.get("lambda_state", 1.0))
    lambda_probe = float(model_cfg.get("lambda_probe", 0.05))
    lambda_conservation = float(model_cfg.get("lambda_conservation", 0.0))

    amp_dtype_str = str(model_cfg.get("amp_dtype", "none")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str in ("bf16", "bfloat16") else None
    amp_enabled = amp_dtype is not None and device.type == "cuda"
    log.info(
        f"loss weights: state={lambda_state} probe={lambda_probe} cons={lambda_conservation} | "
        f"amp={'bf16' if amp_enabled else 'off'}"
    )

    checkpoint_every = int(model_cfg.get("checkpoint_every", 25))
    val_every = int(model_cfg.get("val_every", 5))
    grad_clip = model_cfg.get("grad_clip")

    train_losses: list[float] = []
    train_maes: list[float] = []
    val_losses: list[float] = []
    val_maes: list[float] = []

    model.train()
    start_time = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        loss_sum = mae_sum = count = 0.0
        for u0, u_full in loader:
            u0 = u0.to(device)
            u_full = u_full.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                pred, u_hats = model(u0, x_grid, y_grid, t_grid)
                loss, info = linear_2d_loss(
                    pred, u_full, u0, u_hats,
                    lambda_state, lambda_probe, lambda_conservation,
                )
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            if scheduler is not None:
                scheduler.step()

            bs = u0.size(0)
            loss_sum += loss.item() * bs
            mae_sum += info["state"] * bs
            count += bs

        train_losses.append(loss_sum / max(1, count))
        train_maes.append(mae_sum / max(1, count))
        log.info(
            f"epoch {epoch:3d}/{epochs} | loss={train_losses[-1]:.3e} | "
            f"mae={train_maes[-1]:.3e} | lr={opt.param_groups[0]['lr']:.2e} | {gpu_status()}"
        )
        for h in log.handlers:
            h.flush()

        if val_loader is not None and epoch % val_every == 0:
            model.eval()
            v_loss = v_mae = v_count = 0.0
            with torch.inference_mode():
                for v_u0, v_u in val_loader:
                    v_u0 = v_u0.to(device)
                    v_u = v_u.to(device)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        v_pred, v_hats = model(v_u0, x_grid, y_grid, t_grid)
                        v_l, v_info = linear_2d_loss(
                            v_pred, v_u, v_u0, v_hats,
                            lambda_state, lambda_probe, lambda_conservation,
                        )
                    bs = v_u0.size(0)
                    v_loss += v_l.item() * bs
                    v_mae += v_info["state"] * bs
                    v_count += bs
            val_losses.append(v_loss / max(1, v_count))
            val_maes.append(v_mae / max(1, v_count))
            log.info(
                f"epoch {epoch:3d}/{epochs} | val_loss={val_losses[-1]:.3e} | "
                f"val_mae={val_maes[-1]:.3e}"
            )
            model.train()

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt_path = run_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            log.info(f"Saved checkpoint {ckpt_path}")

    final_path = run_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    log.info(f"Saved final model to {final_path}")
    log.info(f"Training time: {time.perf_counter() - start_time:.1f}s")

    # ---- loss curves ----
    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(14, 5))
    ep = list(range(start_epoch, start_epoch + len(train_losses)))
    val_ep = list(range(
        ((start_epoch - 1) // val_every + 1) * val_every,
        epochs + 1, val_every,
    ))[:len(val_losses)]
    ax_loss.plot(ep, train_losses, label="train loss")
    if val_losses:
        ax_loss.plot(val_ep, val_losses, label="val loss")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("total loss")
    ax_loss.set_yscale("log"); ax_loss.legend(); ax_loss.grid(True, alpha=0.3)
    ax_mae.plot(ep, train_maes, label="train MAE")
    if val_maes:
        ax_mae.plot(val_ep, val_maes, label="val MAE")
    ax_mae.set_xlabel("epoch"); ax_mae.set_ylabel("state MAE")
    ax_mae.set_yscale("log"); ax_mae.legend(); ax_mae.grid(True, alpha=0.3)
    fig.suptitle("HypNO_Linear_2D_v0 training curves")
    fig.savefig(run_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
