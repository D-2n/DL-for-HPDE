"""Train a 2-channel FNO baseline on the ARZ dataset.

Structural mirror of hyperbolic_pde/scripts/train_fno.py (the LWR FNO), adapted
to ARZ:

  * Loads the ARZ dataset via load_arz_dataset and sets the pressure closure from
    the .npz, exactly like train_hypno_arz.py.
  * The input is FOUR channels (X, T, rho0_grid, w0_grid) and the output is TWO
    channels (rho, w) -- the SAME (rho, w) frame that model_arz_orig /
    arz_total_loss optimize, so this is an apples-to-apples baseline against the
    original HypNO-ARZ.
  * Uses the SAME split_indices / split_train_val logic and seed as
    train_hypno_arz.py, so the FNO sees the identical train/val partition of the
    identical dataset section that "arz hypno orig" trained on.

Reuses the generic FNO2d from hyperbolic_pde.models.competitive_architectures.fno (it is already
parameterised by in_channels / out_channels).
"""
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

from hyperbolic_pde.utils.runtime import apply_runtime_overrides
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.models.competitive_architectures.fno import FNO2d


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class ArzFNODataset(Dataset):
    """Build the (4, nx, nt) input / (2, nx, nt) target tensors per sample.

    Mirrors FNODataset from the LWR train_fno.py: the spatial / temporal grids
    are broadcast as coordinate channels, the initial condition (here BOTH rho0
    and w0) is broadcast along time, and the target is the (rho, w) history laid
    out as (channel, nx, nt) -- i.e. each field transposed from (nt, nx).
    """

    def __init__(self, x, t, rho0, w0, rho, w):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.rho0 = torch.tensor(rho0, dtype=torch.float32)
        self.w0 = torch.tensor(w0, dtype=torch.float32)
        self.rho = torch.tensor(rho, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)
        X, T = torch.meshgrid(self.x, self.t, indexing="ij")
        self.X = X
        self.T = T

    def __len__(self) -> int:
        return self.rho.shape[0]

    def __getitem__(self, idx: int):
        nt = self.t.numel()
        rho0_grid = self.rho0[idx].unsqueeze(1).repeat(1, nt)
        w0_grid = self.w0[idx].unsqueeze(1).repeat(1, nt)
        inp = torch.stack([self.X, self.T, rho0_grid, w0_grid], dim=0)  # (4, nx, nt)
        out = torch.stack([self.rho[idx].T, self.w[idx].T], dim=0)      # (2, nx, nt)
        return inp, out


# --------------------------------------------------------------------------- #
# helpers (mirror train_hypno_arz.py so the split matches exactly)
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


def make_optimizer(params, cfg: dict):
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("lr", 1.0e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.0)))
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 2-channel FNO baseline on the ARZ dataset.")
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "hyperbolic_pde_arz.yaml"),
    )
    parser.add_argument(
        "--data-section", type=str, default="arz_stratified_homog_prho",
        help="Which dataset section to use (must match the one 'arz hypno orig' trained on).",
    )
    parser.add_argument(
        "--fno-section", type=str, default="fno_arz",
        help="Which FNO model/training section to use.",
    )
    args = parser.parse_args()
    print("--- STARTING ARZ FNO TRAINING ---")

    cfg = load_config(Path(args.config))
    cfg = apply_runtime_overrides(cfg)
    data_cfg = cfg[args.data_section]
    fno_cfg = cfg[args.fno_section]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    bundle = load_arz_dataset(Path(data_cfg["path"]))

    # Pressure-closure switch: keep consistent with the dataset (only relevant if
    # the eval/recovery of v = w - p(rho) is done later; the FNO itself trains on
    # (rho, w) directly, but we set it so the whole run is closure-consistent).
    from hyperbolic_pde.arz.physics_arz import set_pressure_form, get_pressure_form
    cfg_p_form = str(data_cfg.get("pressure_form", bundle.p_form))
    if cfg_p_form != bundle.p_form:
        print(
            f"[FNO-ARZ] pressure_form mismatch: config says {cfg_p_form!r}, dataset "
            f"says {bundle.p_form!r}. Using dataset value."
        )
        cfg_p_form = bundle.p_form
    set_pressure_form(cfg_p_form)
    print(f"[FNO-ARZ] pressure_form={get_pressure_form()}")
    print(
        f"[FNO-ARZ] dataset={data_cfg['path']}  N={bundle.rho.shape[0]}  "
        f"nt={bundle.t.shape[0]}  nx={bundle.x.shape[0]}  tau={bundle.tau}"
    )

    # SAME split as train_hypno_arz.py: same num_samples cap, train_fraction,
    # val_fraction and seed -> the FNO sees the identical train/val partition.
    N = bundle.rho.shape[0]
    if "num_samples" in data_cfg:
        N = min(N, int(data_cfg["num_samples"]))
    train_idx, _ = split_indices(N, float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    train_idx, val_idx = split_train_val(train_idx, val_fraction, int(cfg.get("seed", 42)))

    train_data = ArzFNODataset(
        bundle.x, bundle.t,
        bundle.rho0[train_idx], bundle.w0[train_idx],
        bundle.rho[train_idx], bundle.w[train_idx],
    )
    loader = DataLoader(train_data, batch_size=int(fno_cfg["batch_size"]), shuffle=True)
    val_loader = None
    if val_idx.size > 0:
        val_data = ArzFNODataset(
            bundle.x, bundle.t,
            bundle.rho0[val_idx], bundle.w0[val_idx],
            bundle.rho[val_idx], bundle.w[val_idx],
        )
        val_loader = DataLoader(val_data, batch_size=int(fno_cfg["batch_size"]), shuffle=False)
    print(f"[FNO-ARZ] train/val batches: {len(loader)}/{len(val_loader) if val_loader else 0}")

    model = FNO2d(
        in_channels=4,   # X, T, rho0, w0
        out_channels=2,  # rho, w
        width=int(fno_cfg["width"]),
        modes_x=int(fno_cfg["modes_x"]),
        modes_t=int(fno_cfg["modes_t"]),
        layers=int(fno_cfg["layers"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FNO-ARZ] {n_params:,} trainable parameters")

    epochs = fno_cfg.get("epochs")
    if epochs is None:
        if "steps" in fno_cfg:
            steps = int(fno_cfg["steps"])
            steps_per_epoch = max(1, len(loader))
            epochs = max(1, math.ceil(steps / steps_per_epoch))
            print(f"[FNO-ARZ] config uses steps={steps}; converting to epochs={epochs}.")
        else:
            raise KeyError("fno_arz config must define 'epochs' (or legacy 'steps')")
    epochs = int(epochs)

    opt = make_optimizer(model.parameters(), fno_cfg)
    scheduler = None
    schedule = fno_cfg.get("lr_schedule")
    total_steps = epochs * max(1, len(loader))
    if schedule == "cosine":
        lr_min = float(fno_cfg.get("lr_min", 1.0e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
    elif schedule == "step":
        lr_step = int(fno_cfg.get("lr_step", 1000))
        lr_gamma = float(fno_cfg.get("lr_gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)

    loss_type = str(fno_cfg.get("loss_type", "mae")).lower()
    if loss_type not in ("mae", "mse"):
        raise ValueError(f"loss_type must be 'mae' or 'mse', got {loss_type!r}")

    def _loss(pred, out):
        if loss_type == "mae":
            return (pred - out).abs().mean()
        return (pred - out).pow(2).mean()

    print(f"[FNO-ARZ] objective={loss_type.upper()}  epochs={epochs}  total_steps={total_steps}")

    step = 0
    model.train()
    start_time = time.perf_counter()
    checkpoint_every = int(fno_cfg.get("checkpoint_every", 20))
    grad_clip = fno_cfg.get("grad_clip")
    save_path = Path(fno_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        for inp, out in loader:
            step += 1
            inp = inp.to(device)
            out = out.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = _loss(pred, out)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            if scheduler is not None:
                scheduler.step()

            if step % 100 == 0 or step == 1:
                lr_now = opt.param_groups[0]["lr"]
                print(
                    f"[FNO-ARZ] epoch {epoch:3d}/{epochs} | step {step:5d}/{total_steps} | "
                    f"{loss_type}={loss.item():.3e} | lr={lr_now:.2e}",
                    flush=True,
                )

        if val_loader is not None:
            model.eval()
            v_sum = 0.0
            v_rho = 0.0
            v_w = 0.0
            v_count = 0
            with torch.no_grad():
                for v_inp, v_out in val_loader:
                    v_inp = v_inp.to(device)
                    v_out = v_out.to(device)
                    v_pred = model(v_inp)
                    B = v_inp.size(0)
                    v_sum += _loss(v_pred, v_out).item() * B
                    v_rho += (v_pred[:, 0] - v_out[:, 0]).abs().mean().item() * B
                    v_w += (v_pred[:, 1] - v_out[:, 1]).abs().mean().item() * B
                    v_count += B
            n = max(1, v_count)
            print(
                f"[FNO-ARZ] epoch {epoch:3d}/{epochs} | val_{loss_type}={v_sum / n:.3e} | "
                f"mae rho/w={v_rho / n:.3e}/{v_w / n:.3e}",
                flush=True,
            )
            model.train()

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt_path = save_path.with_name(f"{save_path.stem}_epoch{epoch}{save_path.suffix}")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[FNO-ARZ] Saved checkpoint to {ckpt_path}")

    torch.save(model.state_dict(), save_path)
    elapsed = time.perf_counter() - start_time
    print(f"[FNO-ARZ] Saved FNO checkpoint to {save_path}")
    print(f"[FNO-ARZ] Training time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
