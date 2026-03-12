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


def total_variation_map(u: np.ndarray) -> np.ndarray:
    tv = np.zeros_like(u)
    tv[:-1, :] += np.abs(u[1:, :] - u[:-1, :])
    tv[:, :-1] += np.abs(u[:, 1:] - u[:, :-1])
    return tv


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual comparison of models on a single sample.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Index within the eval split (or test_path) to visualize.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    test_path = Path(data_cfg.get("test_path", "hyperbolic_pde/data/hyperbolic_test_superres.npz"))
    if test_path.exists():
        data = np.load(test_path)
        x_np = data["x_1x"]
        t_np = data["t_1x"]
        u_np = data["u_1x"]
        u0_np = data["u0_1x"]
        ic_np = data["ic_1x"]
        eval_idx = np.arange(u_np.shape[0])
    else:
        dataset = load_dataset(Path(data_cfg["path"]))
        _, test_idx = split_indices(dataset.u.shape[0], float(data_cfg["train_fraction"]), int(cfg.get("seed", 42)))
        eval_idx = test_idx
        x_np = dataset.x
        t_np = dataset.t
        u_np = dataset.u
        u0_np = dataset.u0
        ic_np = dataset.ic

    if eval_idx.size == 0:
        print("[Compare] No samples available.")
        return

    sample_idx = int(args.sample_idx) if args.sample_idx is not None else 0
    sample_idx = int(eval_idx[sample_idx % len(eval_idx)])

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    Xf = X.reshape(-1, 1)
    Tf = T.reshape(-1, 1)

    u0 = torch.tensor(u0_np[sample_idx], dtype=torch.float32, device=device)
    ic = torch.tensor(ic_np[sample_idx], dtype=torch.float32, device=device)
    truth = torch.tensor(u_np[sample_idx].T, dtype=torch.float32, device=device)

    preds: dict[str, np.ndarray] = {}

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
        with torch.no_grad():
            u0_grid = u0.unsqueeze(1).repeat(1, t.numel())
            inp = torch.stack([X, T, u0_grid], dim=0).unsqueeze(0)
            pred = model(inp)[0, 0]
        preds["FNO"] = pred.detach().cpu().numpy()
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
        with torch.no_grad():
            pred = model(ic, trunk_in).reshape(x.numel(), t.numel())
        preds["DeepONet"] = pred.detach().cpu().numpy()
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
        with torch.no_grad():
            cond_rep = ic.unsqueeze(0).repeat(Xf.size(0), 1)
            pred = model(Xf, Tf, cond_rep).reshape(x.numel(), t.numel())
        preds["PINN"] = pred.detach().cpu().numpy()
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
        with torch.no_grad():
            cond_rep = ic.unsqueeze(0).repeat(Xf.size(0), 1)
            pred = model(Xf, Tf, cond_rep).reshape(x.numel(), t.numel())
        preds["VPINN"] = pred.detach().cpu().numpy()
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
        with torch.no_grad():
            pred = model(u0.unsqueeze(0), dt, dx, n_steps, boundary)[0]
        preds["FluxGNN"] = pred.detach().cpu().numpy().T
    else:
        print("[Compare] FluxGNN checkpoint missing, skipping.")

    if not preds:
        print("[Compare] No checkpoints found. Nothing to plot.")
        return

    truth_np = truth.detach().cpu().numpy()
    vmin = float(np.min(truth_np))
    vmax = float(np.max(truth_np))

    n_rows = len(preds)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (name, pred_np) in enumerate(preds.items()):
        err_np = np.abs(pred_np - truth_np)
        tv_np = total_variation_map(pred_np)

        im0 = axes[r, 0].pcolormesh(x_np, t_np, truth_np.T, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        axes[r, 0].set_title("Truth")
        axes[r, 0].set_xlabel("x")
        axes[r, 0].set_ylabel(f"{name}\nt")
        fig.colorbar(im0, ax=axes[r, 0])

        im1 = axes[r, 1].pcolormesh(x_np, t_np, pred_np.T, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        axes[r, 1].set_title("Prediction")
        axes[r, 1].set_xlabel("x")
        axes[r, 1].set_ylabel("t")
        fig.colorbar(im1, ax=axes[r, 1])

        im2 = axes[r, 2].pcolormesh(x_np, t_np, err_np.T, shading="auto", cmap="magma")
        axes[r, 2].set_title("Absolute error")
        axes[r, 2].set_xlabel("x")
        axes[r, 2].set_ylabel("t")
        fig.colorbar(im2, ax=axes[r, 2])

        im3 = axes[r, 3].pcolormesh(x_np, t_np, tv_np.T, shading="auto", cmap="viridis")
        axes[r, 3].set_title("Total variation")
        axes[r, 3].set_xlabel("x")
        axes[r, 3].set_ylabel("t")
        fig.colorbar(im3, ax=axes[r, 3])

    plot_dir = Path("hyperbolic_pde/runs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"model_compare_visual_sample_{sample_idx}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[Compare] Saved visual comparison to {out_path}")


if __name__ == "__main__":
    main()
