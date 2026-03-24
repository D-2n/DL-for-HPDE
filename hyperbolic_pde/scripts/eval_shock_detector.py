"""Evaluate the pre-trained PINN shock detector.

Produces:
  1. Per-sample plots: u_true vs u_pred, PDE residual shock indicator
  2. Threshold sweep: precision/recall/F1 vs threshold
  3. Recommended threshold for HypNO_PINN's shock capping
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset
from hyperbolic_pde.models.shock_detector import ShockDetector


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


def compute_true_shock_indicator(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Ground truth shock indicator from |du/dx|, normalized per sample."""
    du_dx = torch.zeros_like(u)
    du_dx[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dx)
    du_dx[:, :, 0] = (u[:, :, 1] - u[:, :, 0]) / dx
    du_dx[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) / dx
    abs_grad = du_dx.abs()
    g_max = abs_grad.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    return abs_grad / g_max


def threshold_metrics(pred_indicator, true_indicator, pred_threshold, true_threshold=0.3):
    pred_shock = (pred_indicator > pred_threshold).float()
    true_shock = (true_indicator > true_threshold).float()
    tp = (pred_shock * true_shock).sum().item()
    fp = (pred_shock * (1 - true_shock)).sum().item()
    fn = ((1 - pred_shock) * true_shock).sum().item()
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PINN shock detector.")
    parser.add_argument("--config", type=str, default="hyperbolic_pde/configs/hyperbolic_pde.yaml")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    det_cfg = cfg.get("shock_detector", {})
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(Path(data_cfg["path"]))
    _, test_idx = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))

    x_grid = torch.tensor(dataset.x, dtype=torch.float32, device=device)
    t_grid = torch.tensor(dataset.t, dtype=torch.float32, device=device)
    nx = x_grid.shape[0]
    dx = (x_grid[1] - x_grid[0]).abs().item()

    # model
    model = ShockDetector(
        d_latent=int(det_cfg.get("d_latent", 64)),
        d_hidden=int(det_cfg.get("d_hidden", 128)),
        n_layers=int(det_cfg.get("n_layers", 6)),
        activation=str(det_cfg.get("activation", "tanh")),
        ic_points=nx,
    ).to(device)

    # load weights
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir and (run_dir / "model_final.pt").exists():
        weights_path = run_dir / "model_final.pt"
    else:
        weights_path = Path(det_cfg.get("save_path", "hyperbolic_pde/runs/shock_detector.pt"))
    print(f"Loading weights from {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # output dir
    plot_dir = Path(det_cfg.get("plot_dir", "hyperbolic_pde/runs/plots/shock_detector"))
    if run_dir:
        plot_dir = run_dir / "eval_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_plots = int(det_cfg.get("eval_plots", 8))
    eval_idx = test_idx[:n_plots]

    # ------------------------------------------------------------------ #
    # 1. Per-sample plots
    # ------------------------------------------------------------------ #
    print(f"Generating {len(eval_idx)} sample plots...")
    all_pred_indicators = []
    all_true_indicators = []

    for i, si in enumerate(eval_idx):
        u0 = torch.tensor(dataset.u0[si], dtype=torch.float32, device=device).unsqueeze(0)
        u_true = torch.tensor(dataset.u[si], dtype=torch.float32, device=device).unsqueeze(0)

        # compute_shock_indicator_grid uses autograd internally, no torch.no_grad
        shock_ind, u_pred = model(u0, x_grid, t_grid)
        true_shock = compute_true_shock_indicator(u_true, dx)

        all_pred_indicators.append(shock_ind.cpu())
        all_true_indicators.append(true_shock.cpu())

        x_np = dataset.x
        t_np = dataset.t
        u_true_np = u_true[0].cpu().numpy()
        u_pred_np = u_pred[0].cpu().numpy()
        shock_np = shock_ind[0].cpu().numpy()
        true_shock_np = true_shock[0].cpu().numpy()
        error_np = np.abs(u_true_np - u_pred_np)

        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        c1 = ax1.pcolormesh(x_np, t_np, u_true_np, shading="auto", cmap="viridis")
        fig.colorbar(c1, ax=ax1)
        ax1.set_title("u true"); ax1.set_xlabel("x"); ax1.set_ylabel("t")

        ax2 = fig.add_subplot(gs[0, 1])
        c2 = ax2.pcolormesh(x_np, t_np, u_pred_np, shading="auto", cmap="viridis")
        fig.colorbar(c2, ax=ax2)
        ax2.set_title("u predicted (PINN)"); ax2.set_xlabel("x"); ax2.set_ylabel("t")

        ax3 = fig.add_subplot(gs[0, 2])
        c3 = ax3.pcolormesh(x_np, t_np, error_np, shading="auto", cmap="hot")
        fig.colorbar(c3, ax=ax3)
        ax3.set_title(f"|error| (L1={error_np.mean():.4e})"); ax3.set_xlabel("x"); ax3.set_ylabel("t")

        ax4 = fig.add_subplot(gs[1, 0])
        c4 = ax4.pcolormesh(x_np, t_np, shock_np, shading="auto", cmap="Reds", vmin=0, vmax=1)
        fig.colorbar(c4, ax=ax4)
        ax4.set_title("PDE residual shock indicator"); ax4.set_xlabel("x"); ax4.set_ylabel("t")

        ax5 = fig.add_subplot(gs[1, 1])
        c5 = ax5.pcolormesh(x_np, t_np, true_shock_np, shading="auto", cmap="Reds", vmin=0, vmax=1)
        fig.colorbar(c5, ax=ax5)
        ax5.set_title("True shock (|du/dx| norm.)"); ax5.set_xlabel("x"); ax5.set_ylabel("t")

        ax6 = fig.add_subplot(gs[1, 2])
        c6 = ax6.pcolormesh(x_np, t_np, u_true_np, shading="auto", cmap="viridis")
        fig.colorbar(c6, ax=ax6)
        contour_thresholds = [0.6, 0.8]
        contour_colors = ["orange", "red"]
        for thresh, col in zip(contour_thresholds, contour_colors):
            ax6.contour(x_np, t_np, shock_np, levels=[thresh],
                        colors=[col], linewidths=1.5)
        ax6.set_title("u_true + shock contours (0.6/0.8)"); ax6.set_xlabel("x"); ax6.set_ylabel("t")

        fig.suptitle(f"Sample {si}", fontsize=14)
        fig.tight_layout()
        fig.savefig(plot_dir / f"sample_{i:03d}.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2. Threshold sweep
    # ------------------------------------------------------------------ #
    print("Computing threshold sweep...")
    pred_all = torch.cat(all_pred_indicators, dim=0)
    true_all = torch.cat(all_true_indicators, dim=0)

    pred_thresholds = np.linspace(0.01, 0.99, 50)
    true_thresholds = [0.2, 0.3, 0.5]

    fig, axes = plt.subplots(1, len(true_thresholds), figsize=(6 * len(true_thresholds), 5))
    if len(true_thresholds) == 1:
        axes = [axes]

    for ax, tt in zip(axes, true_thresholds):
        precisions, recalls, f1s = [], [], []
        for pt in pred_thresholds:
            m = threshold_metrics(pred_all, true_all, pt, tt)
            precisions.append(m["precision"])
            recalls.append(m["recall"])
            f1s.append(m["f1"])

        ax.plot(pred_thresholds, precisions, label="Precision", linewidth=2)
        ax.plot(pred_thresholds, recalls, label="Recall", linewidth=2)
        ax.plot(pred_thresholds, f1s, label="F1", linewidth=2, linestyle="--")

        best_idx = np.argmax(f1s)
        best_pt = pred_thresholds[best_idx]
        ax.axvline(best_pt, color="gray", linestyle=":", alpha=0.7)
        ax.annotate(f"best F1={f1s[best_idx]:.2f}\nthresh={best_pt:.2f}",
                     xy=(best_pt, f1s[best_idx]), fontsize=9,
                     xytext=(best_pt + 0.1, f1s[best_idx] - 0.15),
                     arrowprops=dict(arrowstyle="->", color="gray"))

        ax.set_xlabel("Detector threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"True shock def: |du/dx|_norm > {tt}")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Shock Detection: Threshold vs Precision/Recall/F1", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_dir / "threshold_sweep.png", dpi=150)
    plt.close(fig)
    print(f"Saved threshold sweep to {plot_dir / 'threshold_sweep.png'}")

    # ------------------------------------------------------------------ #
    # 3. Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("THRESHOLD RECOMMENDATIONS")
    print("=" * 60)
    for tt in true_thresholds:
        best_f1 = 0
        best_pt = 0
        best_m = {}
        for pt in pred_thresholds:
            m = threshold_metrics(pred_all, true_all, pt, tt)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_pt = pt
                best_m = m
        print(f"  True shock def |du/dx|_norm > {tt:.1f}:")
        print(f"    Best threshold: {best_pt:.2f}  "
              f"(P={best_m['precision']:.3f}, R={best_m['recall']:.3f}, F1={best_f1:.3f})")
    print("=" * 60)

    # prediction quality
    print("\nPrediction quality on test set:")
    total_l1 = 0.0
    total_pts = 0
    with torch.no_grad():
        for si in test_idx:
            u0 = torch.tensor(dataset.u0[si], dtype=torch.float32, device=device).unsqueeze(0)
            u_true = torch.tensor(dataset.u[si], dtype=torch.float32, device=device).unsqueeze(0)
            u_pred = model.predict_grid(u0, x_grid, t_grid)
            total_l1 += F.l1_loss(u_pred, u_true, reduction="sum").item()
            total_pts += u_true.numel()
    print(f"  Mean L1 error: {total_l1 / total_pts:.6e}")
    print(f"  Test samples: {len(test_idx)}")
    print(f"\nAll plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
