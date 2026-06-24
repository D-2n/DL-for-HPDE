"""Compare FNO vs WENO5 vs Godunov against Lax-Hopf exact solution.

Mirror of `eval_vs_numerical.py` but for the baseline FNO2d model. Loads the
same dataset as the FNO was trained on (data.path), takes the held-out test
split, and for each sample evaluates FNO, WENO5, Godunov and reports MAE vs
the Lax-Hopf exact solution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from hyperbolic_pde.utils.runtime import apply_runtime_overrides, resolve_config_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset, solve_conservation_fvm
from hyperbolic_pde.data.lax_hopf import solve_lax_hopf
from hyperbolic_pde.models.competitive_architectures.fno import FNO2d


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


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred - truth).mean())


def per_t_mae(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(pred - truth).mean(axis=1)


def make_fno_input(u0_np: np.ndarray, x_np: np.ndarray, t_np: np.ndarray) -> torch.Tensor:
    """Build the (1, 3, nx, nt) input tensor the FNO expects, mirroring FNODataset.__getitem__."""
    x_t = torch.tensor(x_np, dtype=torch.float32)
    t_t = torch.tensor(t_np, dtype=torch.float32)
    u0_t = torch.tensor(u0_np, dtype=torch.float32)
    X, T = torch.meshgrid(x_t, t_t, indexing="ij")
    u0_grid = u0_t.unsqueeze(1).repeat(1, t_t.numel())
    inp = torch.stack([X, T, u0_grid], dim=0).unsqueeze(0)  # (1, 3, nx, nt)
    return inp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(resolve_config_path(ROOT / "configs")))
    parser.add_argument("--n_samples", type=int, default=70, help="Number of test samples to evaluate")
    parser.add_argument("--n_plots", type=int, default=5, help="Number of samples to plot")
    parser.add_argument(
        "--only_idx", type=str, default=None,
        help="Comma-separated dataset indices to evaluate (e.g. '1547,1753'). "
             "Bypasses --n_samples and the train/test split, runs only these.",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Override the dataset path from config.yaml.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    cfg = apply_runtime_overrides(cfg)

    data_cfg = cfg["data"]
    fno_cfg = cfg["fno"]
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    data_path = Path(args.data_path) if args.data_path else Path(data_cfg["path"])
    print(f"Loading dataset from {data_path}")
    dataset = load_dataset(data_path)

    if args.only_idx:
        test_idx = np.array([int(s) for s in args.only_idx.split(",") if s.strip()])
        print(f"Using --only_idx override: indices={test_idx.tolist()}")
        args.n_plots = max(args.n_plots, len(test_idx))
    else:
        _, test_idx = split_indices(
            dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42))
        )
        test_idx = test_idx[:args.n_samples]

    x_np = dataset.x
    t_np = dataset.t
    x_min = float(x_np[0] - 0.5 * (x_np[1] - x_np[0]))
    x_max = float(x_np[-1] + 0.5 * (x_np[1] - x_np[0]))
    t_max = float(t_np[-1])
    nt = len(t_np)
    cfl = float(data_cfg.get("cfl", 0.3))
    boundary = str(data_cfg.get("boundary", "ghost"))

    # load FNO
    model = FNO2d(
        in_channels=3,
        out_channels=1,
        width=int(fno_cfg["width"]),
        modes_x=int(fno_cfg["modes_x"]),
        modes_t=int(fno_cfg["modes_t"]),
        layers=int(fno_cfg["layers"]),
    ).to(device)
    weights_path = Path(fno_cfg["save_path"])
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded FNO from {weights_path}")

    solvers_to_run = ["FNO", "WENO5", "Godunov"]
    metrics: dict[str, list[float]] = {k: [] for k in solvers_to_run}
    per_t_errors: dict[str, list[np.ndarray]] = {k: [] for k in solvers_to_run}

    plot_dir = Path(fno_cfg.get("plot_dir", "hyperbolic_pde/runs/plots/fno")) / "vs_numerical"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for plot_i, idx in enumerate(test_idx):
        u0_np = dataset.u0[idx]
        print(f"Sample {plot_i + 1}/{len(test_idx)} (dataset idx {idx}) ...", flush=True)

        # Lax-Hopf exact (GT)
        print(f"  running Lax-Hopf...", flush=True)
        _, _, lh = solve_lax_hopf(u0_np, x_min, x_max, t_max, nt, boundary=boundary)

        # WENO5
        print(f"  running WENO5...", flush=True)
        _, _, weno = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="weno5"
        )

        # Godunov
        print(f"  running Godunov...", flush=True)
        _, _, godunov = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="godunov"
        )

        # FNO
        print(f"  running FNO...", flush=True)
        inp = make_fno_input(u0_np, x_np, t_np).to(device)
        with torch.no_grad():
            pred = model(inp)  # (1, 1, nx, nt)
        # FNODataset stores u as (nt, nx) but returns out as u.T -> (nx, nt).
        # FNO predicts in the same (nx, nt) layout, so transpose to (nt, nx).
        fno_np = pred[0, 0].cpu().numpy().T

        pairs = [("FNO", fno_np), ("WENO5", weno), ("Godunov", godunov)]
        for name, p in pairs:
            metrics[name].append(mae(p, lh))
            per_t_errors[name].append(per_t_mae(p, lh))

        if plot_i < args.n_plots:
            vmin, vmax = float(lh.min()), float(lh.max())
            solvers = [("Lax-Hopf (GT)", lh), ("FNO", fno_np), ("WENO5", weno), ("Godunov", godunov)]
            fig, axes = plt.subplots(2, len(solvers), figsize=(4 * len(solvers), 8), constrained_layout=True)

            for c, (name, sol) in enumerate(solvers):
                im_sol = axes[0, c].pcolormesh(
                    x_np, t_np, sol, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
                )
                axes[0, c].set_title(name)
                axes[0, c].set_xlabel("x")
                axes[0, c].set_ylabel("t")
                fig.colorbar(im_sol, ax=axes[0, c], label="u")

            err_vmax = None
            for c, (name, sol) in enumerate(solvers[1:], start=1):
                err = np.abs(sol - lh)
                if err_vmax is None:
                    err_vmax = err.max()
                im = axes[1, c].pcolormesh(
                    x_np, t_np, err, shading="auto", cmap="magma", vmin=0, vmax=err_vmax
                )
                axes[1, c].set_title(f"|{name} - GT|")
                axes[1, c].set_xlabel("x")
                axes[1, c].set_ylabel("t")
                fig.colorbar(im, ax=axes[1, c])
            axes[1, 0].axis("off")

            fig.suptitle(
                f"Sample {idx} — MAE: FNO={metrics['FNO'][-1]:.3e}  "
                f"WENO5={metrics['WENO5'][-1]:.3e}  Godunov={metrics['Godunov'][-1]:.3e}"
            )
            fig.savefig(plot_dir / f"compare_sample_{idx}.png", dpi=150)
            plt.close(fig)

    # summary
    print("\n=== Summary (MAE vs Lax-Hopf) ===")
    for name in solvers_to_run:
        vals = metrics[name]
        print(
            f"  {name:10s}: mean={np.mean(vals):.4e}  std={np.std(vals):.4e}  "
            f"min={np.min(vals):.4e}  max={np.max(vals):.4e}"
        )

    # error vs time
    fig_t, ax_t = plt.subplots(figsize=(8, 4), constrained_layout=True)
    colors = {"FNO": "tab:blue", "WENO5": "tab:orange", "Godunov": "tab:green"}
    for name in solvers_to_run:
        mean_curve = np.stack(per_t_errors[name]).mean(axis=0)
        ax_t.plot(t_np, mean_curve, label=name, color=colors[name])
    ax_t.set_xlabel("t")
    ax_t.set_ylabel("MAE vs Lax-Hopf")
    ax_t.set_title("Error vs time")
    ax_t.set_yscale("log")
    ax_t.legend()
    ax_t.grid(True, alpha=0.3)
    fig_t.savefig(plot_dir / "error_vs_time.png", dpi=150)
    plt.close(fig_t)

    metrics_path = plot_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("MAE vs Lax-Hopf exact solution\n")
        f.write(f"N samples: {len(test_idx)}\n\n")
        for name in solvers_to_run:
            vals = metrics[name]
            f.write(f"{name}:\n")
            f.write(f"  mean = {np.mean(vals):.6e}\n")
            f.write(f"  std  = {np.std(vals):.6e}\n")
            f.write(f"  min  = {np.min(vals):.6e}\n")
            f.write(f"  max  = {np.max(vals):.6e}\n\n")

    print(f"\nSaved plots to {plot_dir}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
