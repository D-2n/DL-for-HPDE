from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset
from hyperbolic_pde.models.deeponet import DeepONet
from hyperbolic_pde.models.fno import FNO2d
from hyperbolic_pde.models.gnn import GridGNN
from hyperbolic_pde.models.pinn import UniversalPINN
from hyperbolic_pde.models.vpinn import VPINN


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


def mse_mae(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    err = pred - truth
    mse = torch.mean(err.pow(2)).item()
    mae = torch.mean(err.abs()).item()
    return mse, mae


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluation metrics across models.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    dataset = load_dataset(Path(data_cfg["path"]))
    _, test_idx = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))

    compare_samples = int(
        cfg.get(
            "compare_samples",
            data_cfg.get("compare_samples", cfg.get("eval_compare_samples", 5)),
        )
    )
    compare_samples = max(1, min(compare_samples, test_idx.shape[0]))
    eval_idx = test_idx[:compare_samples]

    x = torch.tensor(dataset.x, dtype=torch.float32, device=device)
    t = torch.tensor(dataset.t, dtype=torch.float32, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    Xf = X.reshape(-1, 1)
    Tf = T.reshape(-1, 1)

    ic_all = torch.tensor(dataset.ic, dtype=torch.float32, device=device)

    results: dict[str, tuple[float, float]] = {}

    # FNO
    if "fno" in cfg and Path(cfg["fno"]["save_path"]).exists():
        fno_cfg = cfg["fno"]
        model = FNO2d(
            in_channels=3,
            out_channels=1,
            width=int(fno_cfg["width"]),
            modes_x=int(fno_cfg["modes_x"]),
            modes_t=int(fno_cfg["modes_t"]),
            layers=int(fno_cfg["layers"]),
        ).to(device)
        model.load_state_dict(torch.load(Path(fno_cfg["save_path"]), map_location=device))
        model.eval()

        mse_sum = 0.0
        mae_sum = 0.0
        with torch.no_grad():
            for idx in eval_idx:
                u0 = torch.tensor(dataset.u0[idx], dtype=torch.float32, device=device)
                u0_grid = u0.unsqueeze(1).repeat(1, t.numel())
                inp = torch.stack([X, T, u0_grid], dim=0).unsqueeze(0)
                pred = model(inp)[0, 0]
                truth = torch.tensor(dataset.u[idx].T, dtype=torch.float32, device=device)
                mse, mae = mse_mae(pred, truth)
                mse_sum += mse
                mae_sum += mae
        results["FNO"] = (mse_sum / compare_samples, mae_sum / compare_samples)
    else:
        print("[Compare] FNO checkpoint missing, skipping.")

    # DeepONet
    if "deeponet" in cfg and Path(cfg["deeponet"]["save_path"]).exists():
        d_cfg = cfg["deeponet"]
        model = DeepONet(
            branch_in=int(data_cfg["ic_points"]),
            trunk_in=2,
            hidden_width=int(d_cfg["hidden_width"]),
            branch_layers=int(d_cfg["branch_layers"]),
            trunk_layers=int(d_cfg["trunk_layers"]),
            latent_dim=int(d_cfg["latent_dim"]),
            activation=str(d_cfg.get("activation", "tanh")),
            use_bias=bool(d_cfg.get("use_bias", True)),
        ).to(device)
        model.load_state_dict(torch.load(Path(d_cfg["save_path"]), map_location=device))
        model.eval()

        trunk_in = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
        mse_sum = 0.0
        mae_sum = 0.0
        with torch.no_grad():
            for idx in eval_idx:
                branch = ic_all[idx]
                pred = model(branch, trunk_in).reshape(x.numel(), t.numel())
                truth = torch.tensor(dataset.u[idx].T, dtype=torch.float32, device=device)
                mse, mae = mse_mae(pred, truth)
                mse_sum += mse
                mae_sum += mae
        results["DeepONet"] = (mse_sum / compare_samples, mae_sum / compare_samples)
    else:
        print("[Compare] DeepONet checkpoint missing, skipping.")

    # PINN
    if "pinn" in cfg and Path(cfg["pinn"]["save_path"]).exists():
        p_cfg = cfg["pinn"]
        model = UniversalPINN(
            hidden_layers=int(p_cfg["hidden_layers"]),
            hidden_width=int(p_cfg["hidden_width"]),
            cond_dim=int(data_cfg["ic_points"]),
            activation=str(p_cfg.get("activation", "tanh")),
            hard_boundary=bool(p_cfg.get("hard_boundary", False)),
            x_min=float(data_cfg["x_min"]),
            x_max=float(data_cfg["x_max"]),
        ).to(device)
        model.load_state_dict(torch.load(Path(p_cfg["save_path"]), map_location=device))
        model.eval()

        mse_sum = 0.0
        mae_sum = 0.0
        with torch.no_grad():
            for idx in eval_idx:
                cond = ic_all[idx]
                cond_rep = cond.unsqueeze(0).repeat(Xf.size(0), 1)
                pred = model(Xf, Tf, cond_rep).reshape(x.numel(), t.numel())
                truth = torch.tensor(dataset.u[idx].T, dtype=torch.float32, device=device)
                mse, mae = mse_mae(pred, truth)
                mse_sum += mse
                mae_sum += mae
        results["PINN"] = (mse_sum / compare_samples, mae_sum / compare_samples)
    else:
        print("[Compare] PINN checkpoint missing, skipping.")

    # VPINN
    if "vpinn" in cfg and Path(cfg["vpinn"]["save_path"]).exists():
        v_cfg = cfg["vpinn"]
        model = VPINN(
            hidden_layers=int(v_cfg["hidden_layers"]),
            hidden_width=int(v_cfg["hidden_width"]),
            cond_dim=int(data_cfg["ic_points"]),
            activation=str(v_cfg.get("activation", "tanh")),
            hard_boundary=bool(v_cfg.get("hard_boundary", False)),
            x_min=float(data_cfg["x_min"]),
            x_max=float(data_cfg["x_max"]),
            t_min=0.0,
            t_max=float(data_cfg["t_max"]),
            n_test=int(v_cfg.get("n_test", 2)),
        ).to(device)
        model.load_state_dict(torch.load(Path(v_cfg["save_path"]), map_location=device))
        model.eval()

        mse_sum = 0.0
        mae_sum = 0.0
        with torch.no_grad():
            for idx in eval_idx:
                cond = ic_all[idx]
                cond_rep = cond.unsqueeze(0).repeat(Xf.size(0), 1)
                pred = model(Xf, Tf, cond_rep).reshape(x.numel(), t.numel())
                truth = torch.tensor(dataset.u[idx].T, dtype=torch.float32, device=device)
                mse, mae = mse_mae(pred, truth)
                mse_sum += mse
                mae_sum += mae
        results["VPINN"] = (mse_sum / compare_samples, mae_sum / compare_samples)
    else:
        print("[Compare] VPINN checkpoint missing, skipping.")

    # GNN
    if "gnn" in cfg and Path(cfg["gnn"]["save_path"]).exists():
        g_cfg = cfg["gnn"]
        model = GridGNN(
            in_channels=3,
            out_channels=1,
            hidden=int(g_cfg["hidden"]),
            layers=int(g_cfg["layers"]),
        ).to(device)
        model.load_state_dict(torch.load(Path(g_cfg["save_path"]), map_location=device))
        model.eval()

        mse_sum = 0.0
        mae_sum = 0.0
        with torch.no_grad():
            for idx in eval_idx:
                u0 = torch.tensor(dataset.u0[idx], dtype=torch.float32, device=device)
                u0_grid = u0.unsqueeze(1).repeat(1, t.numel())
                inp = torch.stack([X, T, u0_grid], dim=0).unsqueeze(0)
                pred = model(inp)[0, 0]
                truth = torch.tensor(dataset.u[idx].T, dtype=torch.float32, device=device)
                mse, mae = mse_mae(pred, truth)
                mse_sum += mse
                mae_sum += mae
        results["GNN"] = (mse_sum / compare_samples, mae_sum / compare_samples)
    else:
        print("[Compare] GNN checkpoint missing, skipping.")

    if not results:
        print("[Compare] No checkpoints found. Nothing to plot.")
        return

    labels = list(results.keys())
    mse_vals = [results[k][0] for k in labels]
    mae_vals = [results[k][1] for k in labels]

    plot_dir = Path("hyperbolic_pde/runs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "model_comparison_metrics.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(labels, mse_vals)
    axes[0].set_title("MSE (lower is better)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(labels, mae_vals)
    axes[1].set_title("MAE (lower is better)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=30)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("[Compare] Metrics:")
    for name, (mse, mae) in results.items():
        print(f"  {name:8s} | MSE={mse:.3e} | MAE={mae:.3e}")
    print(f"[Compare] Saved plot to {out_path}")


if __name__ == "__main__":
    main()
