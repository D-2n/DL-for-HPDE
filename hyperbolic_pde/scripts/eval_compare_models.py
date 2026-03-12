from __future__ import annotations

import argparse
import sys
import time
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
from hyperbolic_pde.models.fluxgnn import FluxGNN1D
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


def rel_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    err = pred - truth
    denom = torch.norm(truth)
    if denom.item() == 0.0:
        return torch.norm(err).item()
    return (torch.norm(err) / denom).item()


def total_variation_map(u: np.ndarray) -> np.ndarray:
    tv = np.zeros_like(u)
    tv[:-1, :] += np.abs(u[1:, :] - u[:-1, :])
    tv[:, :-1] += np.abs(u[:, 1:] - u[:, :-1])
    return tv


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

    use_test_data = True
    test_path = Path(data_cfg.get("test_path", "hyperbolic_pde/data/hyperbolic_test_superres.npz"))
    if use_test_data and test_path.exists():
        data = np.load(test_path)
        x_np = data["x_1x"]
        t_np = data["t_1x"]
        u_np = data["u_1x"]
        u0_np = data["u0_1x"]
        ic_np = data["ic_1x"]
        eval_idx = np.arange(min(u_np.shape[0], int(data_cfg.get("compare_samples", 5))))
    else:
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
        x_np = dataset.x
        t_np = dataset.t
        u_np = dataset.u
        u0_np = dataset.u0
        ic_np = dataset.ic

    compare_samples = int(min(len(eval_idx), int(data_cfg.get("compare_samples", len(eval_idx)))))
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    Xf = X.reshape(-1, 1)
    Tf = T.reshape(-1, 1)

    ic_all = torch.tensor(ic_np, dtype=torch.float32, device=device)

    metrics: dict[str, dict[str, float]] = {}
    per_sample: dict[str, dict[str, list[float]]] = {}
    sample_maps: dict[str, dict[str, np.ndarray]] = {}
    sample_idx = int(eval_idx[0])

    def _time_forward(fn) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return t1 - t0, out

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

        mse_list: list[float] = []
        mae_list: list[float] = []
        rel_list: list[float] = []
        time_list: list[float] = []
        with torch.no_grad():
            for idx in eval_idx:
                u0 = torch.tensor(u0_np[idx], dtype=torch.float32, device=device)
                u0_grid = u0.unsqueeze(1).repeat(1, t.numel())
                inp = torch.stack([X, T, u0_grid], dim=0).unsqueeze(0)
                dt, pred_out = _time_forward(lambda: model(inp))
                pred = pred_out[0, 0]
                truth = torch.tensor(u_np[idx].T, dtype=torch.float32, device=device)
                mse, mae = mse_mae(pred, truth)
                rel = rel_l2(pred, truth)
                mse_list.append(mse)
                mae_list.append(mae)
                rel_list.append(rel)
                time_list.append(dt)
                if idx == sample_idx:
                    sample_maps["FNO"] = {
                        "pred": pred.detach().cpu().numpy(),
                        "truth": truth.detach().cpu().numpy(),
                    }
        metrics["FNO"] = {
            "mse": float(np.mean(mse_list)),
            "mae": float(np.mean(mae_list)),
            "rel_l2": float(np.mean(rel_list)),
            "time_ms": float(np.mean(time_list) * 1000.0),
        }
        per_sample["FNO"] = {"mse": mse_list, "mae": mae_list, "rel_l2": rel_list}
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
        mse_list = []
        mae_list = []
        rel_list = []
        time_list = []
        with torch.no_grad():
            for idx in eval_idx:
                branch = ic_all[idx]
                dt, pred_out = _time_forward(lambda: model(branch, trunk_in))
                pred = pred_out.reshape(x.numel(), t.numel())
                truth = torch.tensor(u_np[idx].T, dtype=torch.float32, device=device)
                mse, mae = mse_mae(pred, truth)
                rel = rel_l2(pred, truth)
                mse_list.append(mse)
                mae_list.append(mae)
                rel_list.append(rel)
                time_list.append(dt)
                if idx == sample_idx:
                    sample_maps["DeepONet"] = {
                        "pred": pred.detach().cpu().numpy(),
                        "truth": truth.detach().cpu().numpy(),
                    }
        metrics["DeepONet"] = {
            "mse": float(np.mean(mse_list)),
            "mae": float(np.mean(mae_list)),
            "rel_l2": float(np.mean(rel_list)),
            "time_ms": float(np.mean(time_list) * 1000.0),
        }
        per_sample["DeepONet"] = {"mse": mse_list, "mae": mae_list, "rel_l2": rel_list}
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

        mse_list = []
        mae_list = []
        rel_list = []
        time_list = []
        with torch.no_grad():
            for idx in eval_idx:
                cond = ic_all[idx]
                cond_rep = cond.unsqueeze(0).repeat(Xf.size(0), 1)
                dt, pred_out = _time_forward(lambda: model(Xf, Tf, cond_rep))
                pred = pred_out.reshape(x.numel(), t.numel())
                truth = torch.tensor(u_np[idx].T, dtype=torch.float32, device=device)
                mse, mae = mse_mae(pred, truth)
                rel = rel_l2(pred, truth)
                mse_list.append(mse)
                mae_list.append(mae)
                rel_list.append(rel)
                time_list.append(dt)
                if idx == sample_idx:
                    sample_maps["PINN"] = {
                        "pred": pred.detach().cpu().numpy(),
                        "truth": truth.detach().cpu().numpy(),
                    }
        metrics["PINN"] = {
            "mse": float(np.mean(mse_list)),
            "mae": float(np.mean(mae_list)),
            "rel_l2": float(np.mean(rel_list)),
            "time_ms": float(np.mean(time_list) * 1000.0),
        }
        per_sample["PINN"] = {"mse": mse_list, "mae": mae_list, "rel_l2": rel_list}
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

        mse_list = []
        mae_list = []
        rel_list = []
        time_list = []
        with torch.no_grad():
            for idx in eval_idx:
                cond = ic_all[idx]
                cond_rep = cond.unsqueeze(0).repeat(Xf.size(0), 1)
                dt, pred_out = _time_forward(lambda: model(Xf, Tf, cond_rep))
                pred = pred_out.reshape(x.numel(), t.numel())
                truth = torch.tensor(u_np[idx].T, dtype=torch.float32, device=device)
                mse, mae = mse_mae(pred, truth)
                rel = rel_l2(pred, truth)
                mse_list.append(mse)
                mae_list.append(mae)
                rel_list.append(rel)
                time_list.append(dt)
                if idx == sample_idx:
                    sample_maps["VPINN"] = {
                        "pred": pred.detach().cpu().numpy(),
                        "truth": truth.detach().cpu().numpy(),
                    }
        metrics["VPINN"] = {
            "mse": float(np.mean(mse_list)),
            "mae": float(np.mean(mae_list)),
            "rel_l2": float(np.mean(rel_list)),
            "time_ms": float(np.mean(time_list) * 1000.0),
        }
        per_sample["VPINN"] = {"mse": mse_list, "mae": mae_list, "rel_l2": rel_list}
    else:
        print("[Compare] VPINN checkpoint missing, skipping.")

    # FluxGNN
    if "fluxgnn" in cfg and Path(cfg["fluxgnn"]["save_path"]).exists():
        g_cfg = cfg["fluxgnn"]
        model = FluxGNN1D(
            hidden=int(g_cfg["hidden"]),
            layers=int(g_cfg["layers"]),
            activation=str(g_cfg.get("activation", "gelu")),
            latent_dim=g_cfg.get("latent_dim"),
            flux_hidden=g_cfg.get("flux_hidden"),
            use_base_flux=bool(g_cfg.get("use_base_flux", True)),
            base_flux_weight=float(g_cfg.get("base_flux_weight", 0.5)),
            flux_scale=float(g_cfg.get("flux_scale", 0.25)),
        ).to(device)
        model.load_state_dict(torch.load(Path(g_cfg["save_path"]), map_location=device))
        model.eval()

        dx = float(x_np[1] - x_np[0])
        dt = float(t_np[1] - t_np[0])
        n_steps = int(u_np.shape[1])
        boundary = str(data_cfg.get("boundary", "ghost"))

        mse_list = []
        mae_list = []
        rel_list = []
        time_list = []
        with torch.no_grad():
            for idx in eval_idx:
                u0 = torch.tensor(u0_np[idx], dtype=torch.float32, device=device).unsqueeze(0)
                truth = torch.tensor(u_np[idx], dtype=torch.float32, device=device)
                dt_eval, pred_out = _time_forward(lambda: model(u0, dt, dx, n_steps, boundary))
                pred = pred_out[0]
                mse, mae = mse_mae(pred, truth)
                rel = rel_l2(pred, truth)
                mse_list.append(mse)
                mae_list.append(mae)
                rel_list.append(rel)
                time_list.append(dt_eval)
                if idx == sample_idx:
                    sample_maps["FluxGNN"] = {
                        "pred": pred.detach().cpu().numpy().T,
                        "truth": truth.detach().cpu().numpy().T,
                    }
        metrics["FluxGNN"] = {
            "mse": float(np.mean(mse_list)),
            "mae": float(np.mean(mae_list)),
            "rel_l2": float(np.mean(rel_list)),
            "time_ms": float(np.mean(time_list) * 1000.0),
        }
        per_sample["FluxGNN"] = {"mse": mse_list, "mae": mae_list, "rel_l2": rel_list}
    else:
        print("[Compare] FluxGNN checkpoint missing, skipping.")

    print("[Compare] GNN excluded from comparison.")

    if not metrics:
        print("[Compare] No checkpoints found. Nothing to plot.")
        return

    labels = list(metrics.keys())
    mse_vals = [metrics[k]["mse"] for k in labels]
    mae_vals = [metrics[k]["mae"] for k in labels]
    rel_vals = [metrics[k]["rel_l2"] for k in labels]
    time_vals = [metrics[k]["time_ms"] for k in labels]

    plot_dir = Path("hyperbolic_pde/runs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "model_comparison_metrics.png"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
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

    axes[2].bar(labels, rel_vals)
    axes[2].set_title("Relative L2 (lower is better)")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Rel L2")
    axes[2].tick_params(axis="x", rotation=30)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    box_path = plot_dir / "model_comparison_boxplot.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].boxplot([per_sample[k]["mse"] for k in labels], labels=labels, showfliers=False)
    axes[0].set_title("MSE distribution")
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].boxplot([per_sample[k]["mae"] for k in labels], labels=labels, showfliers=False)
    axes[1].set_title("MAE distribution")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=30)
    axes[2].boxplot([per_sample[k]["rel_l2"] for k in labels], labels=labels, showfliers=False)
    axes[2].set_title("Relative L2 distribution")
    axes[2].set_yscale("log")
    axes[2].tick_params(axis="x", rotation=30)
    fig.savefig(box_path, dpi=150)
    plt.close(fig)

    time_path = plot_dir / "model_comparison_timing.png"
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    ax.bar(labels, time_vals)
    ax.set_title("Average inference time per sample")
    ax.set_ylabel("Milliseconds")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(time_path, dpi=150)
    plt.close(fig)

    for name, maps in sample_maps.items():
        pred_np = maps["pred"]
        truth_np = maps["truth"]
        err_np = np.abs(pred_np - truth_np)
        tv_np = total_variation_map(pred_np)
        vmin = float(np.min(truth_np))
        vmax = float(np.max(truth_np))

        fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
        im0 = axes[0].pcolormesh(x_np, t_np, pred_np.T, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"{name} prediction")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("t")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].pcolormesh(x_np, t_np, truth_np.T, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        axes[1].set_title("Godunov FVM truth")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("t")
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].pcolormesh(x_np, t_np, err_np.T, shading="auto", cmap="magma")
        axes[2].set_title("Absolute error")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("t")
        fig.colorbar(im2, ax=axes[2])

        im3 = axes[3].pcolormesh(x_np, t_np, tv_np.T, shading="auto", cmap="viridis")
        axes[3].set_title("Total variation")
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("t")
        fig.colorbar(im3, ax=axes[3])

        fig.savefig(plot_dir / f"compare_{name.lower()}_sample.png", dpi=150)
        plt.close(fig)

    print("[Compare] Metrics:")
    for name, stats in metrics.items():
        print(
            f"  {name:8s} | MSE={stats['mse']:.3e} | MAE={stats['mae']:.3e} | "
            f"RelL2={stats['rel_l2']:.3e} | Time={stats['time_ms']:.2f} ms"
        )
    print(f"[Compare] Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
