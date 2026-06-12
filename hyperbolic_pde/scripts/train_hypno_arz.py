"""Train HypNO-ARZ — structural mirror of train_hypno_st3.py.

Config-driven. Run-dir convention, file logging via train.log, per-batch and
per-epoch lines in the same format as hypno_st3, val cadence, checkpoint
cadence, training-curve plot. ARZ-specific bits live in the loss section and
the (rho, w) per-channel logging.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.utils.runtime import apply_runtime_overrides
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.losses_arz import (
    state_loss, rho_mass_loss, balance_residual_loss, probe_loss,
    mark2_total_loss,
)
from hyperbolic_pde.arz.model_arz import HypNO_ARZ
from hyperbolic_pde.arz.model_arz_mark2 import HypNO_ARZ_Mark2, cfg_to_kwargs_m2


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class ArzTorchDataset(Dataset):
    def __init__(self, rho0, w0, rho, w):
        self.rho0 = torch.tensor(rho0, dtype=torch.float32)
        self.w0   = torch.tensor(w0,   dtype=torch.float32)
        self.rho  = torch.tensor(rho,  dtype=torch.float32)
        self.w    = torch.tensor(w,    dtype=torch.float32)

    def __len__(self): return self.rho0.shape[0]

    def __getitem__(self, i):
        return self.rho0[i], self.w0[i], self.rho[i], self.w[i]


# --------------------------------------------------------------------------- #
# helpers (mirror train_hypno_st3.py)
# --------------------------------------------------------------------------- #
def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> dict:
    base_path = ROOT / "configs" / "hyperbolic_pde_arz.yaml"
    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path.resolve() == base_path.resolve():
        return cfg
    with path.open("r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    return _deep_update(cfg, override or {})


def split_indices(n: int, train_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_fraction * n)
    return idx[:n_train], idx[n_train:]


def split_train_val(train_idx: np.ndarray, val_fraction: float, seed: int):
    if val_fraction <= 0.0:
        return train_idx, np.array([], dtype=train_idx.dtype)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(len(train_idx) * val_fraction))
    return perm[n_val:], perm[:n_val]


def create_run_dir(base: str = "hyperbolic_pde/runs/hypno_arz") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir: Path, cfg: dict, model: torch.nn.Module, config_path: str) -> None:
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with (run_dir / "architecture.txt").open("w", encoding="utf-8") as f:
        f.write("Model: HypNO-ARZ\n")
        f.write(f"Trainable parameters: {n_params:,}\n")
        f.write(f"Config source: {config_path}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"\n{'='*60}\n\n")
        f.write(str(model))
        f.write(f"\n\n{'='*60}\n")
        f.write("Parameter breakdown:\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"  {name}: {list(param.shape)} = {param.numel():,}\n")


def make_optimizer(params, cfg: dict):
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("lr", 1.0e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def setup_logging(run_dir: Path) -> logging.Logger:
    log = logging.getLogger("hypno_arz")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(run_dir / "train.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); log.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt); log.addHandler(ch)
    return log


def gpu_status() -> str:
    if not torch.cuda.is_available():
        return "no-gpu"
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        temp, mem_used, mem_total, util = out.split(", ")
        return f"temp={temp}C  mem={mem_used}/{mem_total}MB  util={util}%"
    except Exception:
        return "nvidia-smi-failed"


def _find_latest_checkpoint(run_dir: Path):
    candidates = []
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


def _p_gt(rho: torch.Tensor) -> torch.Tensor:
    """Pressure p(rho) for ground-truth tensors, honoring the active closure."""
    from hyperbolic_pde.arz import physics_arz as _Pm
    if _Pm._P_FORM == "rho":
        return rho
    return rho + rho * rho


def _diseq_score(rho0: np.ndarray, w0: np.ndarray) -> np.ndarray:
    """Average |v0 - V(rho0)| per sample; > thresh -> off-equilibrium."""
    v0 = w0 - (rho0 + rho0 * rho0)
    return np.abs(v0 - (1.0 - rho0)).mean(axis=1)


# --------------------------------------------------------------------------- #
# composed loss (ARZ-specific, returns scalar + per-term dict in the same
# shape as train_hypno_st3.hypno_pinn_loss)
# --------------------------------------------------------------------------- #
def arz_total_loss(
    rho_pred, w_pred, u_hats,
    rho_gt, w_gt, t, tau,
    *, lambda_state, lambda_conservation, lambda_balance, lambda_probe,
    w_weight, loss_type,
):
    L_state = state_loss(rho_pred, w_pred, rho_gt, w_gt, w_weight=w_weight)
    L_cons  = rho_mass_loss(rho_pred, rho_gt)
    # Skip balance term when disabled or when tau is non-finite (homogeneous
    # / exact-Riemann setting): tau*dIdt would otherwise blow up to inf.
    bal_active = (lambda_balance > 0.0) and np.isfinite(tau)
    if bal_active:
        L_bal = balance_residual_loss(rho_pred, w_pred, t, tau)
    else:
        L_bal = torch.zeros((), device=rho_pred.device, dtype=rho_pred.dtype)
    L_probe = probe_loss(u_hats, rho_gt, w_gt, w_weight=w_weight)
    L = (lambda_state * L_state + lambda_conservation * L_cons
         + lambda_balance * L_bal + lambda_probe * L_probe)
    info = {
        "state": float(L_state.detach().item()),
        "cons":  float(L_cons.detach().item()),
        "bal":   float(L_bal.detach().item()),
        "probe": float(L_probe.detach().item()),
        "total": float(L.detach().item()),
    }
    return L, info


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train HypNO-ARZ on ARZ dataset.")
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "hyperbolic_pde_arz.yaml"),
    )
    parser.add_argument(
        "--data-section", type=str, default="arz_data",
        help="Which dataset section to use (e.g. arz_data, arz_trial, arz_riemann_trial).",
    )
    parser.add_argument(
        "--model-section", type=str, default="hypno_arz",
        help="Which model/training section to use (e.g. hypno_arz, hypno_arz_riemann).",
    )
    parser.add_argument(
        "--model-variant", type=str, default="auto",
        choices=["auto", "mark1", "mark2"],
        help="Which model class. 'auto' picks mark2 when the model-section name "
             "contains 'mark2', else mark1 (HypNO_ARZ).",
    )
    parser.add_argument(
        "--resume_run", type=str, default=None,
        help="Path to a previous run directory. Resumes from its latest checkpoint.",
    )
    parser.add_argument(
        "--auto_resume", action="store_true",
        help="If set and --resume_run is not provided, look at "
             "hyperbolic_pde/runs/hypno_arz/latest_run.txt and resume from "
             "the run it points to (if it has at least one checkpoint). "
             "Otherwise start a fresh run. Safe to use under SLURM --requeue.",
    )
    args = parser.parse_args()
    print('--- STARTING HYPNO_ARZ TRAINING ---')

    resume_run_dir = Path(args.resume_run) if args.resume_run else None

    # Auto-resume: if no explicit --resume_run, consult the latest-run pointer
    # and use that dir IF it contains at least one checkpoint_epoch*.pt. This
    # makes SLURM preemption transparent -- on requeue the rerun picks up the
    # previous run dir instead of starting from scratch.
    #
    # Under SLURM the pointer is keyed by job ID (latest_run_job<ID>.txt):
    # requeue preserves the job ID, so each job resumes exactly its OWN run.
    # The global latest_run.txt is NOT used as a fallback there -- with
    # concurrent jobs it could point at another job's run dir, and two jobs
    # writing one run dir would clobber each other's checkpoints.
    if resume_run_dir is None and args.auto_resume:
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            latest_pointer = Path(
                f"hyperbolic_pde/runs/hypno_arz/latest_run_job{slurm_job_id}.txt")
        else:
            latest_pointer = Path("hyperbolic_pde/runs/hypno_arz/latest_run.txt")
        if latest_pointer.exists():
            candidate = Path(latest_pointer.read_text(encoding="utf-8").strip())
            ckpts = list(candidate.glob("checkpoint_epoch*.pt")) if candidate.exists() else []
            if ckpts:
                resume_run_dir = candidate
                print(f"[auto_resume] picked up run from latest_run.txt: {resume_run_dir}")
            else:
                print(f"[auto_resume] {candidate} has no checkpoints; starting fresh")
        else:
            print(f"[auto_resume] {latest_pointer} missing; starting fresh")

    if resume_run_dir and (resume_run_dir / "config.yaml").exists():
        with (resume_run_dir / "config.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"Resuming from run: {resume_run_dir}")
    else:
        cfg = load_config(Path(args.config))
    cfg = apply_runtime_overrides(cfg)

    data_cfg = cfg[args.data_section]
    model_cfg = cfg[args.model_section]
    print(
        f"[train_arz] config={args.config}  data_section={args.data_section}  "
        f"normalize_edge_offsets={model_cfg.get('normalize_edge_offsets', True)!r}  "
        f"amp_dtype={model_cfg.get('amp_dtype', 'none')!r}"
    )

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    if resume_run_dir:
        run_dir = resume_run_dir
    else:
        run_dir = create_run_dir()
        latest_path = Path("hyperbolic_pde/runs/hypno_arz/latest_run.txt")
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(str(run_dir), encoding="utf-8")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            # Per-job pointer for --auto_resume under SLURM requeue (the job
            # keeps its ID across preemption, so the rerun finds this run).
            (latest_path.parent / f"latest_run_job{slurm_job_id}.txt").write_text(
                str(run_dir), encoding="utf-8")
    log = setup_logging(run_dir)
    log.info(f"Run directory: {run_dir}")
    log.info(f"Device: {device}")
    log.info(f"GPU: {gpu_status()}")

    bundle = load_arz_dataset(Path(data_cfg["path"]))
    tau = float(data_cfg.get("tau", bundle.tau))

    # Pressure-closure switch: model + losses must agree with the dataset.
    from hyperbolic_pde.arz.physics_arz import set_pressure_form, get_pressure_form
    cfg_p_form = str(data_cfg.get("pressure_form", bundle.p_form))
    if cfg_p_form != bundle.p_form:
        log.warning(
            f"pressure_form mismatch: config says {cfg_p_form!r}, dataset says "
            f"{bundle.p_form!r}. Using dataset value to keep training consistent "
            f"with the ground truth."
        )
        cfg_p_form = bundle.p_form
    set_pressure_form(cfg_p_form)
    log.info(f"pressure_form={get_pressure_form()}")
    log.info(
        f"Dataset: {data_cfg['path']}  N={bundle.rho.shape[0]}  "
        f"nt={bundle.t.shape[0]}  nx={bundle.x.shape[0]}  tau={tau}"
    )

    N = bundle.rho.shape[0]
    if "num_samples" in data_cfg:
        N = min(N, int(data_cfg["num_samples"]))
    train_idx, _ = split_indices(N, float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    train_idx, val_idx = split_train_val(train_idx, val_fraction, int(cfg.get("seed", 42)))

    train_ds = ArzTorchDataset(
        bundle.rho0[train_idx], bundle.w0[train_idx],
        bundle.rho[train_idx],  bundle.w[train_idx],
    )
    val_loader = None
    val_diseq = None
    if val_idx.size > 0:
        val_ds = ArzTorchDataset(
            bundle.rho0[val_idx], bundle.w0[val_idx],
            bundle.rho[val_idx],  bundle.w[val_idx],
        )
        val_loader = DataLoader(val_ds, batch_size=int(model_cfg["batch_size"]),
                                shuffle=False)
        val_diseq = _diseq_score(bundle.rho0[val_idx], bundle.w0[val_idx])
        diseq_thr = float(model_cfg.get("diseq_threshold", 0.1))
        n_eq = int((val_diseq < diseq_thr).sum())
        n_off = int((val_diseq >= diseq_thr).sum())
        log.info(f"Val split: near-eq={n_eq}, off-eq={n_off}  (thresh={diseq_thr})")
    loader = DataLoader(train_ds, batch_size=int(model_cfg["batch_size"]), shuffle=True)
    log.info(f"train/val batches: {len(loader)}/{len(val_loader) if val_loader else 0}")

    _dhn = model_cfg.get("d_hidden_nonadj", None)
    d_hidden_nonadj = int(_dhn) if _dhn is not None else None

    # Resolve model variant (auto -> mark2 iff section name says so).
    variant = args.model_variant
    if variant == "auto":
        variant = "mark2" if "mark2" in args.model_section else "mark1"
    log.info(f"Model variant: {variant}")

    if variant == "mark2":
        m2_kwargs = cfg_to_kwargs_m2(model_cfg)
        m2_kwargs["use_checkpoint"] = bool(model_cfg.get("use_checkpoint", True))
        model = HypNO_ARZ_Mark2(**m2_kwargs).to(device)
    else:
        model = HypNO_ARZ(
            stencil_k_x=int(model_cfg.get("stencil_k_x", 2)),
            stencil_k_t=int(model_cfg.get("stencil_k_t", 2)),
            d_latent=int(model_cfg.get("d_latent", 96)),
            d_hidden=int(model_cfg.get("d_hidden", 96)),
            n_layers=int(model_cfg.get("n_layers", 7)),
            activation=str(model_cfg.get("activation", "gelu")),
            causal_temporal=bool(model_cfg.get("causal_temporal", True)),
            d_hidden_nonadj=d_hidden_nonadj,
            decoder_depth=int(model_cfg.get("decoder_depth", 3)),
            skip=bool(model_cfg.get("skip", True)),
            use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
            normalize_edge_offsets=bool(model_cfg.get("normalize_edge_offsets", True)),
            use_relaxation_features=bool(model_cfg.get("use_relaxation_features", True)),
        ).to(device)

    x_grid = torch.tensor(bundle.x, dtype=torch.float32, device=device)
    t_grid = torch.tensor(bundle.t, dtype=torch.float32, device=device)

    start_epoch = 1
    if resume_run_dir:
        ckpt_path, resumed_epoch = _find_latest_checkpoint(resume_run_dir)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            if any(k.startswith("_orig_mod.") for k in ckpt):
                ckpt = {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
            start_epoch = resumed_epoch + 1
            log.info(f"Resumed from {ckpt_path} (epoch {resumed_epoch}), continuing from epoch {start_epoch}")
        else:
            log.warning(f"No checkpoints found in {resume_run_dir}, starting from scratch")

    torch.set_float32_matmul_precision("high")
    use_compile = bool(model_cfg.get("compile", False)) and hasattr(torch, "compile")
    if use_compile:
        log.info("Compiling model with torch.compile ...")
        model = torch.compile(model)
        log.info("torch.compile done")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"{n_params:,} trainable parameters")

    save_run_metadata(run_dir, cfg, model, args.config)

    epochs = int(model_cfg["epochs"])
    opt = make_optimizer(model.parameters(), model_cfg)
    scheduler = None
    total_steps = epochs * max(1, len(loader))
    schedule = model_cfg.get("lr_schedule")
    warmup_steps = int(model_cfg.get("lr_warmup_steps", 0))
    if schedule == "cosine":
        lr_min = float(model_cfg.get("lr_min", 1.0e-5))
        if warmup_steps > 0:
            # Linear warmup 0->lr over warmup_steps, then cosine over the rest.
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, total_steps - warmup_steps), eta_min=lr_min)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
            log.info(f"LR schedule: linear warmup {warmup_steps} steps -> cosine to {lr_min}")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
        if start_epoch > 1:
            for _ in range((start_epoch - 1) * len(loader)):
                scheduler.step()
    elif schedule == "step":
        lr_step = int(model_cfg.get("lr_step", 1000))
        lr_gamma = float(model_cfg.get("lr_gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)
        if start_epoch > 1:
            for _ in range((start_epoch - 1) * len(loader)):
                scheduler.step()

    lambda_state        = float(model_cfg.get("lambda_state", 1.0))
    lambda_conservation = float(model_cfg.get("lambda_conservation", 1.0))
    lambda_balance      = float(model_cfg.get("lambda_balance", 0.1))
    lambda_probe        = float(model_cfg.get("lambda_probe", 0.01))
    w_weight            = float(model_cfg.get("w_weight", 1.0))
    loss_type           = str(model_cfg.get("loss_type", "mae")).lower()
    bal_disabled = (lambda_balance <= 0.0) or (not np.isfinite(tau))
    bal_str = "OFF" if bal_disabled else f"{lambda_balance}"
    log.info(f"Training objective: {loss_type.upper()}  tau={tau}  "
             f"weights state={lambda_state} cons={lambda_conservation} "
             f"bal={bal_str} probe={lambda_probe} w={w_weight}")

    amp_dtype_str = str(model_cfg.get("amp_dtype", "none")).lower()
    if amp_dtype_str in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    elif amp_dtype_str in ("none", "fp32", "off", ""):
        amp_dtype = None
    else:
        raise ValueError(f"hypno_arz.amp_dtype must be 'bf16' or 'none', got {amp_dtype_str!r}")
    amp_enabled = amp_dtype is not None and device.type == "cuda"
    if amp_enabled:
        log.info(f"Mixed precision: torch.autocast(cuda, dtype={amp_dtype})")
    else:
        log.info("Mixed precision: off (fp32)")

    grad_clip = model_cfg.get("grad_clip")
    checkpoint_every = int(model_cfg.get("checkpoint_every", 10))
    val_every = int(model_cfg.get("val_every", 10))
    log_batch_every = int(model_cfg.get("log_batch_every", 50))

    step = (start_epoch - 1) * len(loader)
    model.train()
    start_time = time.perf_counter()

    train_losses: list[float] = []
    train_mses:   list[float] = []
    val_losses:   list[float] = []
    val_mses:     list[float] = []

    for epoch in range(start_epoch, epochs + 1):
        epoch_loss_sum = 0.0
        epoch_mse_sum = 0.0
        epoch_count = 0
        for rho0, w0, rho_gt, w_gt in loader:
            step += 1
            rho0 = rho0.to(device); w0 = w0.to(device)
            rho_gt = rho_gt.to(device); w_gt = w_gt.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                if variant == "mark2":
                    rho_p, v_p, w_p, u_hats_rv = model.forward_primitive(rho0, w0, x_grid, t_grid)
                    v_gt = w_gt - _p_gt(rho_gt)
                    loss, info = mark2_total_loss(
                        rho_p, v_p, u_hats_rv, rho_gt, v_gt,
                        lambda_state=lambda_state,
                        lambda_conservation=lambda_conservation,
                        lambda_probe=lambda_probe,
                        v_weight=w_weight,
                    )
                else:
                    rho_p, w_p, u_hats = model(rho0, w0, x_grid, t_grid)
                    loss, info = arz_total_loss(
                        rho_p, w_p, u_hats, rho_gt, w_gt, t_grid, tau,
                        lambda_state=lambda_state,
                        lambda_conservation=lambda_conservation,
                        lambda_balance=lambda_balance,
                        lambda_probe=lambda_probe,
                        w_weight=w_weight, loss_type=loss_type,
                    )
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                batch_mse = float(((rho_p - rho_gt).pow(2) + (w_p - w_gt).pow(2)).mean().item())
            epoch_loss_sum += loss.item() * rho0.size(0)
            epoch_mse_sum  += batch_mse   * rho0.size(0)
            epoch_count += rho0.size(0)

            if log_batch_every > 0 and step % log_batch_every == 0:
                n_batches = len(loader)
                batch_idx = (step - 1) % n_batches + 1
                log.info(
                    f"  epoch {epoch:3d} batch {batch_idx}/{n_batches} | "
                    f"loss={loss.item():.3e} | state={info['state']:.3e} "
                    f"cons={info['cons']:.3e} bal={info['bal']:.3e} "
                    f"probe={info['probe']:.3e} | mse={batch_mse:.3e}"
                )
                for h in log.handlers: h.flush()

        train_losses.append(epoch_loss_sum / max(1, epoch_count))
        train_mses.append(epoch_mse_sum / max(1, epoch_count))
        lr_now = opt.param_groups[0]["lr"]
        log.info(
            f"epoch {epoch:3d}/{epochs} | optim_loss={train_losses[-1]:.3e} | "
            f"mse={train_mses[-1]:.3e} | lr={lr_now:.2e} | {gpu_status()}"
        )
        for h in log.handlers: h.flush()

        if val_loader is not None and epoch % val_every == 0:
            model.eval()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_loss = 0.0; val_mse = 0.0; val_count = 0
            mae_rho = 0.0; mae_w = 0.0; mae_v = 0.0
            mae_eq_sum = 0.0; mae_off_sum = 0.0
            n_eq = 0; n_off = 0
            diseq_thr = float(model_cfg.get("diseq_threshold", 0.1))
            with torch.inference_mode():
                idx_start = 0
                for v_rho0, v_w0, v_rho, v_w in val_loader:
                    v_rho0 = v_rho0.to(device); v_w0 = v_w0.to(device)
                    v_rho = v_rho.to(device);   v_w  = v_w.to(device)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        if variant == "mark2":
                            vrp, vvp_, vwp, vuh_rv = model.forward_primitive(v_rho0, v_w0, x_grid, t_grid)
                            v_v_gt = v_w - _p_gt(v_rho)
                            vL, _ = mark2_total_loss(
                                vrp, vvp_, vuh_rv, v_rho, v_v_gt,
                                lambda_state=lambda_state,
                                lambda_conservation=lambda_conservation,
                                lambda_probe=lambda_probe,
                                v_weight=w_weight,
                            )
                        else:
                            vrp, vwp, vuh = model(v_rho0, v_w0, x_grid, t_grid)
                            vL, _ = arz_total_loss(
                                vrp, vwp, vuh, v_rho, v_w, t_grid, tau,
                                lambda_state=lambda_state,
                                lambda_conservation=lambda_conservation,
                                lambda_balance=lambda_balance,
                                lambda_probe=lambda_probe,
                                w_weight=w_weight, loss_type=loss_type,
                            )
                    B = v_rho0.size(0)
                    v_mse = float(((vrp - v_rho).pow(2) + (vwp - v_w).pow(2)).mean().item())
                    val_loss += vL.item() * B
                    val_mse  += v_mse * B
                    val_count += B
                    # Per-channel and eq/off MAE.
                    vvp = vwp - (vrp + vrp * vrp)
                    vvgt = v_w - (v_rho + v_rho * v_rho)
                    mae_rho_b = (vrp - v_rho).abs().mean(dim=(1, 2))
                    mae_w_b   = (vwp - v_w).abs().mean(dim=(1, 2))
                    mae_v_b   = (vvp - vvgt).abs().mean(dim=(1, 2))
                    mae_rho += float(mae_rho_b.sum().item())
                    mae_w   += float(mae_w_b.sum().item())
                    mae_v   += float(mae_v_b.sum().item())
                    for b in range(B):
                        is_eq = val_diseq[idx_start + b] < diseq_thr
                        cmae = float(mae_rho_b[b] + mae_w_b[b])
                        if is_eq:
                            mae_eq_sum += cmae; n_eq += 1
                        else:
                            mae_off_sum += cmae; n_off += 1
                    idx_start += B
            n = max(1, val_count)
            val_avg = val_loss / n
            val_mse_avg = val_mse / n
            val_losses.append(val_avg); val_mses.append(val_mse_avg)
            log.info(
                f"epoch {epoch:3d}/{epochs} | val_optim_loss={val_avg:.3e} | "
                f"val_mse={val_mse_avg:.3e} | "
                f"mae rho/w/v={mae_rho/n:.3e}/{mae_w/n:.3e}/{mae_v/n:.3e} | "
                f"mae eq/off={(mae_eq_sum/max(1,n_eq)):.3e}/{(mae_off_sum/max(1,n_off)):.3e} | "
                f"{gpu_status()}"
            )
            for h in log.handlers: h.flush()
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss, ax_mse = axes
    ep_range = list(range(1, len(train_losses) + 1))
    val_epochs = list(range(val_every, epochs + 1, val_every))[:len(val_losses)]
    ax_loss.plot(ep_range, train_losses, label="Train optim loss")
    if val_loader is not None and val_losses:
        ax_loss.plot(val_epochs, val_losses, label="Val optim loss")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel(f"Optim loss ({loss_type.upper()}-based)")
    ax_loss.set_title("Optimization loss"); ax_loss.set_yscale("log")
    ax_loss.legend(); ax_loss.grid(True, alpha=0.3)
    ax_mse.plot(ep_range, train_mses, label="Train MSE")
    if val_loader is not None and val_mses:
        ax_mse.plot(val_epochs, val_mses, label="Val MSE")
    ax_mse.set_xlabel("Epoch"); ax_mse.set_ylabel("MSE")
    ax_mse.set_title("State MSE (monitoring)"); ax_mse.set_yscale("log")
    ax_mse.legend(); ax_mse.grid(True, alpha=0.3)
    fig.suptitle("HypNO-ARZ training curves")
    curve_path = run_dir / "loss_curves.png"
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved loss curves to {curve_path}")

    latest_path = Path("hyperbolic_pde/runs/hypno_arz/latest_run.txt")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir), encoding="utf-8")
    log.info(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
