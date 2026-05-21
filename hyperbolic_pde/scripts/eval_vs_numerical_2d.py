"""Evaluate HypNO-ST3-2D against the WENO5 numerical reference.

Loads a trained checkpoint, runs it on the test split (the tail of the
dataset, matching the split in train_hypno_st3_2d.py), and reports
MSE / MAE / relative-L2 against the WENO5 ground truth, plus per-timestep
error curves and per-sample pred / truth / |error| heatmap rows.

Usage from repo root:
    python hyperbolic_pde/scripts/eval_vs_numerical_2d.py --run-dir <dir>
    python hyperbolic_pde/scripts/eval_vs_numerical_2d.py --ckpt <path> --config <yaml>

If --run-dir is given, the config and weights are read from it
(config.yaml + model_final.pt, falling back to the latest checkpoint).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from hyperbolic_pde.data.fvm_2d import load_dataset_2d
from hyperbolic_pde.models.hypno_st3_2d import HypNO_ST3_2D


def split_indices(n: int, train_fraction: float, val_fraction: float, seed: int):
    """Return (train_idx, val_idx, test_idx) -- same split as training."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_fraction * n)
    n_val = int(val_fraction * n)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def _resolve_weights(run_dir: Path) -> Path:
    final = run_dir / "model_final.pt"
    if final.exists():
        return final
    # Sort by numeric epoch (filenames are unpadded, so lex order would pick
    # epoch 90 over epoch 150).
    candidates: list[tuple[int, Path]] = []
    for p in run_dir.glob("checkpoint_epoch*.pt"):
        try:
            ep = int(p.stem.replace("checkpoint_epoch", ""))
        except ValueError:
            continue
        candidates.append((ep, p))
    if candidates:
        return max(candidates, key=lambda kv: kv[0])[1]
    raise FileNotFoundError(f"No model_final.pt or checkpoint in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HypNO-ST3-2D vs WENO5.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "hyperbolic_pde_2d.yaml"))
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run directory: reads config.yaml + weights from it.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Explicit checkpoint path (overrides run-dir weights).")
    parser.add_argument("--eval-plots", type=int, default=4)
    args = parser.parse_args()

    # ---- config + weights ----
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir and (run_dir / "config.yaml").exists():
        with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"config from {run_dir / 'config.yaml'}")
    else:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"config from {args.config}")

    if args.ckpt:
        weights_path = Path(args.ckpt)
    elif run_dir:
        weights_path = _resolve_weights(run_dir)
    else:
        raise SystemExit("provide --run-dir or --ckpt")

    data_cfg = cfg["data"]
    model_cfg = cfg["hypno_st3_2d"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg.get("seed", 42))

    # ---- data + test split ----
    data_path = Path(data_cfg["path"])
    if not data_path.is_absolute():
        data_path = ROOT.parent / data_path
    bundle = load_dataset_2d(data_path)
    n = bundle.u.shape[0]
    _, _, test_idx = split_indices(
        n, float(data_cfg["train_fraction"]), float(data_cfg["val_fraction"]), seed,
    )
    print(f"dataset {bundle.u.shape}  |  {len(test_idx)} test samples")

    x = torch.tensor(bundle.x, dtype=torch.float32, device=device)
    y = torch.tensor(bundle.y, dtype=torch.float32, device=device)
    t = torch.tensor(bundle.t, dtype=torch.float32, device=device)

    # ---- model ----
    _dhn = model_cfg.get("d_hidden_nonadj", None)
    model = HypNO_ST3_2D(
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
    ).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"weights: {weights_path}  |  {n_params:,} params")

    u0_test = torch.tensor(bundle.u0[test_idx], dtype=torch.float32, device=device)
    u_test = torch.tensor(bundle.u[test_idx], dtype=torch.float32, device=device)
    n_test = u0_test.shape[0]

    batch_size = int(model_cfg.get("batch_size", 8))
    preds = []
    with torch.inference_mode():
        for i in range(0, n_test, batch_size):
            u_pred, _ = model(u0_test[i : i + batch_size], x, y, t)
            preds.append(u_pred)
    pred = torch.cat(preds, dim=0)                              # [n_test, nt, ny, nx]

    # ---- aggregate metrics ----
    mse = (pred - u_test).pow(2).mean().item()
    mae = (pred - u_test).abs().mean().item()
    rel_l2 = (
        (pred - u_test).pow(2).sum() / u_test.pow(2).sum().clamp(min=1e-12)
    ).sqrt().item()
    print(f"Test MSE: {mse:.6e}")
    print(f"Test MAE: {mae:.6e}")
    print(f"Test rL2: {rel_l2:.6e}")

    # ---- per-timestep error ----
    per_t_mae = (pred - u_test).abs().mean(dim=(0, 2, 3)).cpu().numpy()
    per_t_mse = (pred - u_test).pow(2).mean(dim=(0, 2, 3)).cpu().numpy()

    # ---- output dir ----
    out_dir = (run_dir / "plots") if run_dir else (ROOT / "scripts" / "_sanity_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].plot(bundle.t, per_t_mse, "o-")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("MSE"); ax[0].set_yscale("log")
    ax[0].set_title("MSE vs time"); ax[0].grid(True, alpha=0.3)
    ax[1].plot(bundle.t, per_t_mae, "o-")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("MAE"); ax[1].set_yscale("log")
    ax[1].set_title("MAE vs time"); ax[1].grid(True, alpha=0.3)
    fig.savefig(out_dir / "error_vs_time.png", dpi=150)
    plt.close(fig)

    # ---- per-sample heatmap rows ----
    nt = bundle.t.shape[0]
    frames = np.linspace(0, nt - 1, 5).round().astype(int)
    for b in range(min(args.eval_plots, n_test)):
        p = pred[b].cpu().numpy()
        tr = u_test[b].cpu().numpy()
        fig, axes = plt.subplots(3, len(frames), figsize=(3 * len(frames), 8))
        for col, k in enumerate(frames):
            for row, (field, name, vmax) in enumerate([
                (p[k], "pred", 1.0),
                (tr[k], "WENO5", 1.0),
                (np.abs(p[k] - tr[k]), "|err|", 0.3),
            ]):
                ax = axes[row, col]
                ax.imshow(field, origin="lower", vmin=0.0, vmax=vmax, cmap="viridis")
                if col == 0:
                    ax.set_ylabel(name)
                if row == 0:
                    ax.set_title(f"t={bundle.t[k]:.2f}")
                ax.set_xticks([]); ax.set_yticks([])
        s_mae = np.abs(p - tr).mean()
        fig.suptitle(f"test sample {b} | MAE={s_mae:.3e}")
        fig.savefig(out_dir / f"eval_sample_{b}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    metrics_path = out_dir / "metrics_2d.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"Test MSE: {mse:.6e}\n")
        f.write(f"Test MAE: {mae:.6e}\n")
        f.write(f"Test rL2: {rel_l2:.6e}\n")
        f.write(f"Weights:  {weights_path}\n")
        f.write(f"N test:   {n_test}\n")
        f.write(f"N params: {n_params:,}\n")
    print(f"saved plots + metrics to {out_dir}")


if __name__ == "__main__":
    main()
