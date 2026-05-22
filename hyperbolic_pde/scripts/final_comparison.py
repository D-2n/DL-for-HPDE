"""Final OOD comparison: HypNO-ST3 vs WENO5 vs Godunov on the grouped OOD set.

Loads a trained model from --run-dir, runs all three solvers on every sample of
the grouped OOD dataset (produced by generate_ood_data.py), and reports:

  * average MAE vs Lax-Hopf exact, per method, as a function of num_segments
  * average wall-clock time to produce one output, per method, vs num_segments
  * eval_vs_numerical-style solution + error comparison plots, one per
    num_segments group (a representative sample from that group)

Every plot title states the IC type and num_segments so the qualitative
behaviour can be read off directly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from hyperbolic_pde.utils.runtime import apply_runtime_overrides

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import solve_conservation_fvm
from hyperbolic_pde.data.lax_hopf import solve_lax_hopf
from hyperbolic_pde.models.hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3
from hyperbolic_pde.models.fno import FNO2d

SOLVERS = ["HypNO-ST3", "WENO5", "Godunov", "FNO"]
COLORS = {
    "HypNO-ST3": "tab:blue",
    "WENO5": "tab:orange",
    "Godunov": "tab:green",
    "FNO": "tab:red",
}

# Hard-coded path to the trained FNO checkpoint on CLEPS. The FNO baseline is
# treated as a fixed pretrained model: training happens out-of-band via
# scripts/train_fno.py with configs/hyperbolic_pde_cleps_fno.yaml.
FNO_WEIGHTS_PATH = "/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/fno.pt"
# Architecture matches the default cfg['fno'] block in hyperbolic_pde.yaml
# (width=32, modes_x=16, modes_t=16, layers=4). If you retrain with different
# hyperparameters, update these constants too.
FNO_ARCH = dict(in_channels=3, out_channels=1, width=32, modes_x=16, modes_t=16, layers=4)


def build_fno(device: torch.device, weights_path: str = FNO_WEIGHTS_PATH) -> FNO2d:
    """Load the pretrained FNO baseline. Returns ``None`` if the checkpoint
    is missing (so the comparison can still run on machines without it)."""
    model = FNO2d(**FNO_ARCH).to(device)
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(
            f"FNO checkpoint not found at {p}. Either train it via "
            f"scripts/train_fno.py or update FNO_WEIGHTS_PATH."
        )
    state = torch.load(p, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def run_fno(
    model: FNO2d,
    u0_np: np.ndarray,
    x_grid: torch.Tensor,
    t_grid: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Run the FNO baseline on one IC.

    Mirrors the input layout from FNODataset in scripts/train_fno.py:
    channels = [X, T, u0_broadcast], shape (1, 3, nx, nt). The model emits
    (1, 1, nx, nt), which we transpose to (nt, nx) to match HypNO/FVM.
    """
    nx = x_grid.numel()
    nt = t_grid.numel()
    X, T = torch.meshgrid(x_grid, t_grid, indexing="ij")  # (nx, nt)
    u0_t = torch.as_tensor(u0_np, dtype=torch.float32, device=device)
    u0_grid = u0_t.unsqueeze(1).expand(nx, nt)
    inp = torch.stack([X, T, u0_grid], dim=0).unsqueeze(0)  # (1, 3, nx, nt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(inp)  # (1, 1, nx, nt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return pred[0, 0].cpu().numpy().T, elapsed  # -> (nt, nx)


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred - truth).mean())


def per_t_mae(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(pred - truth).mean(axis=1)


def load_ood_dataset(path: Path):
    """Load the grouped OOD .npz. Returns x, t, u, u0, ic, num_segments, ic_type."""
    from hyperbolic_pde import resolve_data_path

    path = resolve_data_path(path)
    data = np.load(path, allow_pickle=False)
    if "num_segments" not in data or "ic_type" not in data:
        raise KeyError(
            f"{path} is missing 'num_segments'/'ic_type' arrays. Regenerate it "
            f"with the updated generate_ood_data.py."
        )
    return (
        data["x"],
        data["t"],
        data["u"],
        data["u0"],
        data["ic"],
        data["num_segments"].astype(int),
        np.array([str(s) for s in data["ic_type"]]),
    )


def build_model(model_cfg: dict, device: torch.device) -> HypNO_ST3:
    _dhn = model_cfg.get("d_hidden_nonadj", None)
    d_hidden_nonadj = int(_dhn) if _dhn is not None else None
    model = HypNO_ST3(
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
        dilated_spatial=bool(model_cfg.get("dilated_spatial", False)),
        normalize_edge_offsets=bool(model_cfg.get("normalize_edge_offsets", False)),
        decoder_depth=int(model_cfg.get("decoder_depth", 3)),
    ).to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final OOD comparison of HypNO-ST3 vs WENO5 vs Godunov."
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Run directory containing config.yaml and model_final.pt.",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Override the OOD dataset path (defaults to cfg['ood_data']['path']).",
    )
    parser.add_argument(
        "--n_per_group", type=int, default=None,
        help="Cap on samples evaluated per num_segments group (default: all).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_runtime_overrides(cfg)
    print(f"Using config from {run_dir / 'config.yaml'}")

    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st")))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # ---- dataset ----
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        ood_cfg = cfg.get("ood_data")
        if not ood_cfg or "path" not in ood_cfg:
            raise KeyError(
                "No --data_path given and config has no 'ood_data: path:'. "
                "Pass --data_path explicitly."
            )
        data_path = Path(ood_cfg["path"])
    print(f"Loading OOD dataset from {data_path}")
    x_np, t_np, u_gt, u0_all, _ic, seg_all, ictype_all = load_ood_dataset(data_path)

    x_min = float(x_np[0] - 0.5 * (x_np[1] - x_np[0]))
    x_max = float(x_np[-1] + 0.5 * (x_np[1] - x_np[0]))
    t_max = float(t_np[-1])
    nt = len(t_np)
    data_cfg = cfg["data"]
    cfl = float(data_cfg.get("cfl", 0.25))
    boundary = str(data_cfg.get("boundary", "ghost"))

    # ---- model ----
    model = build_model(model_cfg, device)
    weights_path = run_dir / "model_final.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"No model_final.pt in {run_dir}")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded HypNO-ST3 from {weights_path}")

    # FNO baseline — hardcoded checkpoint path; see FNO_WEIGHTS_PATH.
    fno_model = build_fno(device)
    print(f"Loaded FNO from {FNO_WEIGHTS_PATH}")

    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)
    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"

    plot_dir = run_dir / "plots_final_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)

    seg_values = sorted(set(int(s) for s in seg_all))
    print(f"num_segments groups: {seg_values}")

    # Per-group accumulators.
    # mae_by_seg[seg][solver]   -> list of per-sample MAE
    # time_by_seg[seg][solver]  -> list of per-sample wall times (seconds)
    mae_by_seg = {seg: {s: [] for s in SOLVERS} for seg in seg_values}
    time_by_seg = {seg: {s: [] for s in SOLVERS} for seg in seg_values}

    def run_hypno(u0_np: np.ndarray) -> tuple[np.ndarray, float]:
        u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device).unsqueeze(0)
        if not skip_ef:
            ef_adj, ef_nonadj = precompute_lwr_edge_features_v3(u0_t, x_grid, stencil_k_x)
            ef_adj = ef_adj.to(device)
            ef_nonadj = ef_nonadj.to(device) if ef_nonadj.numel() > 0 else None
        else:
            ef_adj, ef_nonadj = None, None
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_t, _, _, _ = model(
                u0_t, x_grid, t_grid,
                edge_feats_adj=ef_adj, edge_feats_nonadj=ef_nonadj,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return pred_t[0].cpu().numpy(), elapsed

    # For the per-group qualitative plot, keep one representative sample.
    rep_by_seg: dict[int, dict] = {}

    n_total = u0_all.shape[0]
    for idx in range(n_total):
        seg = int(seg_all[idx])
        ic_type = str(ictype_all[idx])
        # how many of this group already done?
        done = len(mae_by_seg[seg]["HypNO-ST3"])
        if args.n_per_group is not None and done >= args.n_per_group:
            continue

        u0_np = u0_all[idx]
        lh = u_gt[idx]  # ground truth already stored in the OOD npz

        print(f"[{idx + 1}/{n_total}] num_segments={seg}, ic_type={ic_type} ...", flush=True)

        # WENO5
        t0 = time.perf_counter()
        _, _, weno = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="weno5"
        )
        t_weno = time.perf_counter() - t0

        # Godunov
        t0 = time.perf_counter()
        _, _, godunov = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="godunov"
        )
        t_godu = time.perf_counter() - t0

        # HypNO-ST3
        hypno_np, t_hypno = run_hypno(u0_np)

        # FNO
        fno_np, t_fno = run_fno(fno_model, u0_np, x_grid, t_grid, device)

        results = {
            "HypNO-ST3": (hypno_np, t_hypno),
            "WENO5": (weno, t_weno),
            "Godunov": (godunov, t_godu),
            "FNO": (fno_np, t_fno),
        }
        for name, (pred, elapsed) in results.items():
            mae_by_seg[seg][name].append(mae(pred, lh))
            time_by_seg[seg][name].append(elapsed)

        if seg not in rep_by_seg:
            rep_by_seg[seg] = {
                "idx": idx,
                "ic_type": ic_type,
                "lh": lh,
                "preds": {name: pred for name, (pred, _) in results.items()},
            }

    # ---------------------------------------------------------------- #
    # Plot 1: average MAE vs num_segments, per method
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name in SOLVERS:
        means = [np.mean(mae_by_seg[seg][name]) for seg in seg_values]
        stds = [np.std(mae_by_seg[seg][name]) for seg in seg_values]
        ax.errorbar(
            seg_values, means, yerr=stds, marker="o", capsize=4,
            label=name, color=COLORS[name],
        )
    ax.set_xlabel("num_segments (discontinuities in IC)")
    ax.set_ylabel("average MAE vs Lax-Hopf exact")
    ax.set_yscale("log")
    ax.set_xticks(seg_values)
    ax.set_title("OOD accuracy vs IC complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(plot_dir / "mae_vs_num_segments.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # Plot 2: average wall-time vs num_segments, per method
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name in SOLVERS:
        means = [np.mean(time_by_seg[seg][name]) for seg in seg_values]
        stds = [np.std(time_by_seg[seg][name]) for seg in seg_values]
        ax.errorbar(
            seg_values, means, yerr=stds, marker="o", capsize=4,
            label=name, color=COLORS[name],
        )
    ax.set_xlabel("num_segments (discontinuities in IC)")
    ax.set_ylabel("average wall time per sample [s]")
    ax.set_yscale("log")
    ax.set_xticks(seg_values)
    ax.set_title("OOD per-sample runtime vs IC complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(plot_dir / "time_vs_num_segments.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # Plot 3: eval_vs_numerical-style comparison, one per num_segments
    # ---------------------------------------------------------------- #
    for seg in seg_values:
        rep = rep_by_seg.get(seg)
        if rep is None:
            continue
        lh = rep["lh"]
        preds = rep["preds"]
        ic_type = rep["ic_type"]
        idx = rep["idx"]
        vmin, vmax = float(lh.min()), float(lh.max())
        panels = [("Lax-Hopf (GT)", lh)] + [(n, preds[n]) for n in SOLVERS]

        fig, axes = plt.subplots(
            2, len(panels), figsize=(4 * len(panels), 8), constrained_layout=True
        )
        for c, (name, sol) in enumerate(panels):
            im = axes[0, c].pcolormesh(
                x_np, t_np, sol, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
            )
            axes[0, c].set_title(name)
            axes[0, c].set_xlabel("x")
            axes[0, c].set_ylabel("t")
            fig.colorbar(im, ax=axes[0, c], label="u")

        err_vmax = None
        for c, (name, sol) in enumerate(panels[1:], start=1):
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

        mae_str = "  ".join(
            f"{n}={mae(preds[n], lh):.3e}" for n in SOLVERS
        )
        fig.suptitle(
            f"OOD sample {idx} — IC: {ic_type}, num_segments={seg} "
            f"({seg} discontinuities)\nMAE: {mae_str}"
        )
        fig.savefig(plot_dir / f"compare_num_segments_{seg}.png", dpi=150)
        plt.close(fig)

    # ---------------------------------------------------------------- #
    # Summary table
    # ---------------------------------------------------------------- #
    summary_path = plot_dir / "summary.txt"
    lines = ["OOD final comparison — MAE vs Lax-Hopf, wall time per sample\n"]
    for seg in seg_values:
        n = len(mae_by_seg[seg]["HypNO-ST3"])
        lines.append(f"num_segments = {seg}  ({n} samples)")
        for name in SOLVERS:
            m = mae_by_seg[seg][name]
            tt = time_by_seg[seg][name]
            lines.append(
                f"  {name:12s}: MAE mean={np.mean(m):.4e} std={np.std(m):.4e}  |  "
                f"time mean={np.mean(tt):.4e}s std={np.std(tt):.4e}s"
            )
        lines.append("")
    summary = "\n".join(lines)
    print("\n" + summary)
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Saved plots and summary to {plot_dir}")


if __name__ == "__main__":
    main()
