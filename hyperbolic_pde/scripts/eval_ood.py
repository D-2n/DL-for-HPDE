"""Out-of-distribution evaluation for HypNO-ST3.

Loads the OOD dataset specified by ``ood_data:`` in the YAML config and
evaluates each run listed in ``RUNS_TO_TEST`` below against WENO5,
Godunov, and the Lax-Hopf exact solution.  Per-run plot/metric outputs
land in ``<run_dir>/plots_ood/``.

Edit ``RUNS_TO_TEST`` (and optionally ``N_SAMPLES`` / ``N_PLOTS``) to
control which runs get evaluated.
"""
from __future__ import annotations

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
from hyperbolic_pde.models.hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3


# --------------------------------------------------------------------- #
# EDIT THIS LIST: absolute paths to run directories to evaluate.
# Each entry must contain config.yaml and model_final.pt.
# --------------------------------------------------------------------- #
RUNS_TO_TEST: list[str] = [
    #"/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_st3/run_20260505_095544",
    "/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_st3/run_20260505_181417"
    # "/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_st3/run_20260505_095544",
    # "/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_st3/run_20260506_000000",
]

N_SAMPLES: int = 100   # number of OOD samples to evaluate per run
N_PLOTS:   int = 30     # per-run side-by-side plots (rest are metrics-only)


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


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred - truth).mean())


def per_t_mae(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(pred - truth).mean(axis=1)


def _build_model(model_cfg: dict, device: torch.device) -> HypNO_ST3:
    _dhn = model_cfg.get("d_hidden_nonadj", None)
    d_hidden_nonadj = int(_dhn) if _dhn is not None else None
    return HypNO_ST3(
        stencil_k_x=int(model_cfg.get("stencil_k_x", 3)),
        stencil_k_t=int(model_cfg.get("stencil_k_t", 2)),
        d_latent=int(model_cfg.get("d_latent", 128)),
        d_hidden=int(model_cfg.get("d_hidden", 128)),
        n_layers=int(model_cfg.get("n_layers", 6)),
        activation=str(model_cfg.get("activation", "gelu")),
        shock_delta=float(model_cfg.get("shock_delta", 0.01)),
        shock_threshold=float(model_cfg.get("shock_threshold", 0.1)),
        causal_temporal=bool(model_cfg.get("causal_temporal", True)),
        radius_x=None,
        radius_t=None,
        shock_mode=str(model_cfg.get("shock_mode", "physics")),
        weno_eps=float(model_cfg.get("weno_eps", 1e-6)),
        weno_p=float(model_cfg.get("weno_p", 2.0)),
        unified_mp=bool(model_cfg.get("unified_mp", False)),
        readout=str(model_cfg.get("readout", "relu")),
        encoder_scaling=str(model_cfg.get("encoder_scaling", "classic")),
        encoder_type=str(model_cfg.get("encoder_type", "gnn")),
        skip=bool(model_cfg.get("skip", False)),
        use_char_cone=bool(model_cfg.get("use_char_cone", False)),
        detector_path=None,
        detector_cfg={},
        d_hidden_nonadj=d_hidden_nonadj,
        include_flux=bool(model_cfg.get("include_flux", True)),
        mask_same_t_nonadj=bool(model_cfg.get("mask_same_t_nonadj", True)),
        temporal_gate_type=str(model_cfg.get("temporal_gate_type", "cfl")),
        pure_pairwise_edges=bool(model_cfg.get("pure_pairwise_edges", False)),
    ).to(device)


def _evaluate_run(
    run_dir: Path,
    dataset,
    sample_indices: np.ndarray,
    boundary: str,
    cfl: float,
    device: torch.device,
) -> dict:
    """Evaluate a single run on the provided OOD samples.  Returns metrics dict."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_runtime_overrides(cfg)
    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st")))

    weights_path = run_dir / "model_final.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"No model_final.pt in {run_dir}")

    model = _build_model(model_cfg, device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  loaded weights ({n_params:,} params, include_flux="
          f"{model_cfg.get('include_flux', True)})")

    x_np = dataset.x
    t_np = dataset.t
    x_min = float(x_np[0]  - 0.5 * (x_np[1] - x_np[0]))
    x_max = float(x_np[-1] + 0.5 * (x_np[1] - x_np[0]))
    t_max = float(t_np[-1])
    nt = len(t_np)

    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)
    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    skip_ef = str(model_cfg.get("encoder_type", "gnn")) == "mlp"

    plot_dir = run_dir / "plots_ood"
    plot_dir.mkdir(parents=True, exist_ok=True)

    solvers_to_run = ["HypNO-ST3", "WENO5", "Godunov"]
    metrics: dict[str, list[float]] = {k: [] for k in solvers_to_run}
    per_t_errors: dict[str, list[np.ndarray]] = {k: [] for k in solvers_to_run}

    for plot_i, idx in enumerate(sample_indices):
        u0_np = dataset.u0[idx]
        print(f"  sample {plot_i + 1}/{len(sample_indices)} (dataset idx {idx})", flush=True)

        _, _, lh = solve_lax_hopf(u0_np, x_min, x_max, t_max, nt, boundary=boundary)
        _, _, weno = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="weno5"
        )
        _, _, godunov = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="godunov"
        )

        u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device).unsqueeze(0)
        if not skip_ef:
            ef_adj, ef_nonadj = precompute_lwr_edge_features_v3(u0_t, x_grid, stencil_k_x)
            ef_adj = ef_adj.to(device)
            ef_nonadj = ef_nonadj.to(device) if ef_nonadj.numel() > 0 else None
        else:
            ef_adj, ef_nonadj = None, None
        with torch.no_grad():
            pred_t, _, _, _ = model(
                u0_t, x_grid, t_grid,
                edge_feats_adj=ef_adj, edge_feats_nonadj=ef_nonadj,
            )
        hypno_np = pred_t[0].cpu().numpy()

        for name, pred in [("HypNO-ST3", hypno_np), ("WENO5", weno), ("Godunov", godunov)]:
            metrics[name].append(mae(pred, lh))
            per_t_errors[name].append(per_t_mae(pred, lh))

        if plot_i < N_PLOTS:
            vmin, vmax = float(lh.min()), float(lh.max())
            solvers = [
                ("Lax-Hopf (GT)", lh),
                ("HypNO-ST3", hypno_np),
                ("WENO5", weno),
                ("Godunov", godunov),
            ]
            fig, axes = plt.subplots(2, len(solvers), figsize=(4 * len(solvers), 8),
                                      constrained_layout=True)
            sol_im = None
            for c, (name, sol) in enumerate(solvers):
                sol_im = axes[0, c].pcolormesh(x_np, t_np, sol, shading="auto", cmap="jet",
                                                vmin=vmin, vmax=vmax)
                axes[0, c].set_title(name)
                axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
                fig.colorbar(sol_im, ax=axes[0, c], label="u")

            err_vmax = None
            for c, (name, sol) in enumerate(solvers[1:], start=1):
                err = np.abs(sol - lh)
                if err_vmax is None:
                    err_vmax = err.max()
                im = axes[1, c].pcolormesh(x_np, t_np, err, shading="auto",
                                            cmap="magma", vmin=0, vmax=err_vmax)
                axes[1, c].set_title(f"|{name} - GT|")
                axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
                fig.colorbar(im, ax=axes[1, c])
            axes[1, 0].axis("off")

            fig.suptitle(
                f"OOD sample {idx} — MAE: HypNO={metrics['HypNO-ST3'][-1]:.3e}  "
                f"WENO5={metrics['WENO5'][-1]:.3e}  Godunov={metrics['Godunov'][-1]:.3e}"
            )
            fig.savefig(plot_dir / f"ood_sample_{idx}.png", dpi=150)
            plt.close(fig)

    # error vs time (mean across samples)
    fig_t, ax_t = plt.subplots(figsize=(8, 4), constrained_layout=True)
    colors = {"HypNO-ST3": "tab:blue", "WENO5": "tab:orange", "Godunov": "tab:green"}
    for name in solvers_to_run:
        mean_curve = np.stack(per_t_errors[name]).mean(axis=0)
        ax_t.plot(t_np, mean_curve, label=name, color=colors[name])
    ax_t.set_xlabel("t"); ax_t.set_ylabel("MAE vs Lax-Hopf")
    ax_t.set_title(f"OOD error vs time  ({run_dir.name})")
    ax_t.set_yscale("log"); ax_t.legend(); ax_t.grid(True, alpha=0.3)
    fig_t.savefig(plot_dir / "ood_error_vs_time.png", dpi=150)
    plt.close(fig_t)

    metrics_path = plot_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"OOD evaluation — {run_dir}\n")
        f.write(f"N samples: {len(sample_indices)}\n\n")
        for name in solvers_to_run:
            vals = np.array(metrics[name])
            f.write(
                f"{name}:\n  mean={vals.mean():.6e}\n  std ={vals.std():.6e}\n"
                f"  min ={vals.min():.6e}\n  max ={vals.max():.6e}\n\n"
            )
    print(f"  metrics written to {metrics_path}")
    return {name: np.array(metrics[name]) for name in solvers_to_run}


def main() -> None:
    if not RUNS_TO_TEST:
        raise SystemExit(
            "RUNS_TO_TEST is empty.  Edit eval_ood.py and add absolute paths "
            "to run directories at the top of the file."
        )

    cfg_path = resolve_config_path(ROOT / "configs")
    cfg = load_config(Path(cfg_path))
    cfg = apply_runtime_overrides(cfg)
    if "ood_data" not in cfg or cfg["ood_data"] is None:
        raise KeyError("Config has no 'ood_data:' block. Generate the OOD "
                       "dataset first via generate_ood_data.py.")
    ood_cfg = cfg["ood_data"]
    ood_path = Path(ood_cfg["path"])
    if not ood_path.exists():
        raise FileNotFoundError(
            f"OOD dataset not found at {ood_path}. "
            "Run generate_ood_data.py first."
        )

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading OOD dataset from {ood_path}")
    dataset = load_dataset(ood_path)
    n_total = dataset.u.shape[0]
    sample_indices = np.arange(min(N_SAMPLES, n_total))
    print(f"OOD set: {n_total} samples available, evaluating {len(sample_indices)}")

    boundary = str(ood_cfg.get("boundary", "ghost"))
    cfl      = float(ood_cfg.get("cfl", 0.3))

    summary: dict[str, dict[str, np.ndarray]] = {}
    for run_str in RUNS_TO_TEST:
        run_dir = Path(run_str)
        print(f"\n=== Run: {run_dir} ===")
        if not run_dir.exists():
            print(f"  SKIP — directory does not exist")
            continue
        try:
            summary[run_dir.name] = _evaluate_run(
                run_dir, dataset, sample_indices, boundary, cfl, device,
            )
        except Exception as e:
            print(f"  ERROR — {e}")
            continue

    print("\n=== OOD summary (MAE vs Lax-Hopf) ===")
    for run_name, mset in summary.items():
        print(f"\n[{run_name}]")
        for solver, vals in mset.items():
            print(f"  {solver:12s}: mean={vals.mean():.4e}  "
                  f"std={vals.std():.4e}  min={vals.min():.4e}  max={vals.max():.4e}")


if __name__ == "__main__":
    main()
