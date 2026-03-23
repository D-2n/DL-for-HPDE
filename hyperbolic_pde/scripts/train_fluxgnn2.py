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
from hyperbolic_pde.models.fluxgnn2 import ConservativeFluxOperator1D


class ConservativeOperatorDataset(Dataset):
    def __init__(
        self,
        u: np.ndarray,
        t: np.ndarray,
        sample_pairs: str = "all",
        max_horizon_steps: int | None = None,
    ) -> None:
        """
        u: [num_samples, num_times, num_cells]
        t: [num_times]
        """
        self.u = torch.tensor(u, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)

        n_samples, n_times, _ = self.u.shape
        max_h = n_times - 1 if max_horizon_steps is None else min(max_horizon_steps, n_times - 1)

        pairs: list[tuple[int, int, int]] = []
        for s in range(n_samples):
            for i in range(n_times - 1):
                if sample_pairs == "final_only":
                    j = n_times - 1
                    if j > i:
                        pairs.append((s, i, j))
                elif sample_pairs == "random_one":
                    j = min(i + np.random.randint(1, max_h + 1), n_times - 1)
                    if j > i:
                        pairs.append((s, i, j))
                else:  # "all"
                    for h in range(1, max_h + 1):
                        j = i + h
                        if j < n_times:
                            pairs.append((s, i, j))

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s, i, j = self.pairs[idx]
        u_t = self.u[s, i]         # [N]
        u_target = self.u[s, j]    # [N]
        tau = self.t[j] - self.t[i]  # scalar
        return u_t, tau.unsqueeze(0), u_target


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
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.0))), False
    return torch.optim.Adam(params, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0))), False


def semigroup_loss(
    model: ConservativeFluxOperator1D,
    u_t: torch.Tensor,
    dx: float,
    tau: torch.Tensor,
) -> torch.Tensor:
    """
    Enforce O_{tau}(u) ≈ O_{tau/2}( O_{tau/2}(u) )
    """
    tau_half = 0.5 * tau
    with torch.no_grad():
        direct = model(u_t, tau, dx, boundary="ghost")
    half_1 = model(u_t, tau_half, dx, boundary="ghost")
    half_2 = model(half_1, tau_half, dx, boundary="ghost")
    return torch.mean(torch.abs(half_2 - direct))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conservative one-shot operator on hyperbolic PDE dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    op_cfg = cfg["conservative_operator"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(cfg.get("seed", 42))
    print(f"Training ConservativeFluxOperator1D on device: {device}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = load_dataset(Path(data_cfg["path"]))
    train_idx, _ = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), seed)
    train_idx, val_idx = split_train_val(train_idx, float(data_cfg.get("val_fraction", 0.1)), seed)

    pair_mode = str(op_cfg.get("pair_mode", "all"))
    max_horizon_steps = op_cfg.get("max_horizon_steps")
    max_horizon_steps = None if max_horizon_steps is None else int(max_horizon_steps)

    train_data = ConservativeOperatorDataset(
        dataset.u[train_idx],
        dataset.t,
        sample_pairs=pair_mode,
        max_horizon_steps=max_horizon_steps,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=int(op_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(op_cfg.get("num_workers", 0)),
        pin_memory=bool(op_cfg.get("pin_memory", False)),
    )

    val_loader = None
    if val_idx.size > 0:
        val_data = ConservativeOperatorDataset(
            dataset.u[val_idx],
            dataset.t,
            sample_pairs="all",
            max_horizon_steps=max_horizon_steps,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=int(op_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(op_cfg.get("num_workers", 0)),
            pin_memory=bool(op_cfg.get("pin_memory", False)),
        )

    model = ConservativeFluxOperator1D(
        latent_dim=int(op_cfg.get("latent_dim", 32)),
        hidden=int(op_cfg.get("hidden", 64)),
        layers=int(op_cfg.get("layers", 3)),
        activation=str(op_cfg.get("activation", "gelu")),
        tau_embed_dim=int(op_cfg.get("tau_embed_dim", 16)),
        flux_scale=float(op_cfg.get("flux_scale", 1.0)),
    ).to(device)

    epochs = op_cfg.get("epochs")
    if epochs is None:
        if "steps" in op_cfg:
            steps = int(op_cfg["steps"])
            steps_per_epoch = max(1, len(train_loader))
            epochs = max(1, math.ceil(steps / steps_per_epoch))
            print(f"[ConservativeOp] config uses steps={steps}; converting to epochs={epochs}.")
        else:
            raise KeyError("conservative_operator config must define 'epochs' (or legacy 'steps')")
    epochs = int(epochs)

    opt, use_lbfgs = make_optimizer(model.parameters(), op_cfg)
    scheduler = None
    schedule = op_cfg.get("lr_schedule")
    total_steps = epochs * max(1, len(train_loader))
    if not use_lbfgs:
        if schedule == "cosine":
            lr_min = float(op_cfg.get("lr_min", 1.0e-5))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
        elif schedule == "step":
            lr_step = int(op_cfg.get("lr_step", 1000))
            lr_gamma = float(op_cfg.get("lr_gamma", 0.5))
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)

    dx = float(dataset.x[1] - dataset.x[0])

    lambda_mass = float(op_cfg.get("lambda_mass", 0.0))
    lambda_range = float(op_cfg.get("lambda_range", 0.0))
    lambda_semigroup = float(op_cfg.get("lambda_semigroup", 0.0))

    step = 0
    start_time = time.perf_counter()
    checkpoint_every = int(op_cfg.get("checkpoint_every", 20))

    for epoch in range(1, epochs + 1):
        model.train()
        for u_t, tau, u_target in train_loader:
            step += 1
            u_t = u_t.to(device)
            tau = tau.to(device)
            u_target = u_target.to(device)

            def closure() -> torch.Tensor:
                opt.zero_grad(set_to_none=True)

                pred = model(u_t, tau, dx, boundary="ghost")

                loss_state = torch.mean(torch.abs(pred - u_target))

                mass_pred = pred.sum(dim=1)
                mass_true = u_target.sum(dim=1)
                loss_mass = torch.mean(torch.abs(mass_pred - mass_true))

                below = torch.relu(-pred)
                above = torch.relu(pred - 1.0)
                loss_range = torch.mean(below + above)

                loss_sg = pred.new_tensor(0.0)
                if lambda_semigroup > 0.0:
                    loss_sg = semigroup_loss(model, u_t, dx, tau)

                loss = (
                    loss_state
                    + lambda_mass * loss_mass
                    + lambda_range * loss_range
                    + lambda_semigroup * loss_sg
                )
                loss.backward()

                grad_clip = op_cfg.get("grad_clip")
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

            if step % 50 == 0 or step == 1:
                lr_now = opt.param_groups[0]["lr"]
                print(
                    f"[ConservativeOp] epoch {epoch:3d}/{epochs} | "
                    f"step {step:5d}/{total_steps} | "
                    f"loss={loss.item():.3e} | lr={lr_now:.2e}"
                )

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for v_u_t, v_tau, v_u_target in val_loader:
                    v_u_t = v_u_t.to(device)
                    v_tau = v_tau.to(device)
                    v_u_target = v_u_target.to(device)

                    v_pred = model(v_u_t, v_tau, dx, boundary="ghost")
                    v_loss = torch.mean(torch.abs(v_pred - v_u_target)).item()
                    val_loss += v_loss * v_u_t.size(0)
                    val_count += v_u_t.size(0)

            val_mae = val_loss / max(1, val_count)
            print(f"[ConservativeOp] epoch {epoch:3d}/{epochs} | val_mae={val_mae:.3e}")

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_path = Path(op_cfg["save_path"])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_path.with_name(f"{save_path.stem}_epoch{epoch}{save_path.suffix}")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[ConservativeOp] Saved checkpoint to {ckpt_path}")

    save_path = Path(op_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    elapsed = time.perf_counter() - start_time
    print(f"Saved ConservativeFluxOperator1D checkpoint to {save_path}")
    print(f"[ConservativeOp] Training time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
