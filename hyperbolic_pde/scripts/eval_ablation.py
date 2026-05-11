"""Unified numerical + visual evaluation across HypNO-ST3 ablation variants.

Loads 4 HypNO-ST3 runs (auto-identified by config flags), evaluates each on
the same N random test samples vs. precomputed Lax-Hopf ground truth (from
the dataset), plus WENO5 and Godunov as references. Emits a CSV table of
aggregate metrics and side-by-side comparison plots for selected samples.

Run directories are mapped to variant labels via their config.yaml flags:
    baseline       : include_flux=true,  temporal_gate_type=cfl,  pure_pairwise=false
    no_flux        : include_flux=false, temporal_gate_type=cfl,  pure_pairwise=false
    temporal_time  : include_flux=*,     temporal_gate_type=time, pure_pairwise=false
    pure_pairwise  : include_flux=*,     temporal_gate_type=*,    pure_pairwise=true
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset, solve_conservation_fvm
from hyperbolic_pde.models.hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3


# ─── variant identification ─────────────────────────────────────────────── #
def identify_variant(cfg: dict) -> str:
    """Map a run config to a variant label by inspecting the model flags."""
    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st", {})))
    include_flux = bool(model_cfg.get("include_flux", True))
    temporal_gate = str(model_cfg.get("temporal_gate_type", "cfl"))
    pure_pairwise = bool(model_cfg.get("pure_pairwise_edges", False))
    if pure_pairwise:
        return "pure_pairwise"
    if temporal_gate == "time":
        return "temporal_time"
    if not include_flux:
        return "no_flux"
    return "baseline"


# ─── model loading ──────────────────────────────────────────────────────── #
def build_model_from_cfg(model_cfg: dict, device: torch.device) -> HypNO_ST3:
    """Instantiate HypNO_ST3 from a run's model config, mirroring eval_vs_numerical."""
    _dhn = model_cfg.get("d_hidden_nonadj", None)
    return HypNO_ST3(
        stencil_k_x=int(model_cfg.get("stencil_k_x", 3)),
        stencil_k_t=int(model_cfg.get("stencil_k_t", 2)),
        d_latent=int(model_cfg.get("d_latent", 128)),
        d_hidden=int(model_cfg.get("d_hidden", 128)),
        n_layers=int(model_cfg.get("n_layers", 6)),
        activation=str(model_cfg.get("activation", "gelu")),
        causal_temporal=bool(model_cfg.get("causal_temporal", True)),
        radius_x=None,
        radius_t=None,
        readout=str(model_cfg.get("readout", "gelu")),
        encoder_scaling=str(model_cfg.get("encoder_scaling", "physics")),
        encoder_type=str(model_cfg.get("encoder_type", "gnn")),
        skip=bool(model_cfg.get("skip", False)),
        use_char_cone=bool(model_cfg.get("use_char_cone", False)),
        detector_path=None,
        detector_cfg={},
        d_hidden_nonadj=int(_dhn) if _dhn is not None else None,
        include_flux=bool(model_cfg.get("include_flux", True)),
        mask_same_t_nonadj=bool(model_cfg.get("mask_same_t_nonadj", True)),
        temporal_gate_type=str(model_cfg.get("temporal_gate_type", "cfl")),
        pure_pairwise_edges=bool(model_cfg.get("pure_pairwise_edges", False)),
    ).to(device)


def load_run(run_dir: Path, device: torch.device) -> tuple[str, HypNO_ST3, dict]:
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    variant = identify_variant(cfg)
    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st", {})))
    model = build_model_from_cfg(model_cfg, device)

    weights = run_dir / "model_final.pt"
    if not weights.exists():
        # Fall back to highest-epoch checkpoint.
        ckpts = sorted(run_dir.glob("checkpoint_epoch*.pt"),
                       key=lambda p: int(p.stem.split("epoch")[1]))
        if not ckpts:
            raise FileNotFoundError(f"No weights found in {run_dir}")
        weights = ckpts[-1]
    state_dict = torch.load(weights, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[{variant}] {run_dir.name} <- {weights.name}")
    return variant, model, cfg


# ─── metrics ────────────────────────────────────────────────────────────── #
def metrics_vs_gt(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    diff = pred - gt
    return {
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "linf": float(np.abs(diff).max()),
    }


def find_shock_positions(u_row: np.ndarray, x: np.ndarray, level: float = 0.5) -> np.ndarray:
    sign = np.sign(u_row - level)
    sign = np.where(sign == 0, 1.0, sign)
    crossings = np.where(sign[:-1] != sign[1:])[0]
    if crossings.size == 0:
        return np.empty(0, dtype=np.float64)
    u_l = u_row[crossings]
    u_r = u_row[crossings + 1]
    x_l = x[crossings]
    x_r = x[crossings + 1]
    alpha = (level - u_l) / (u_r - u_l)
    return x_l + alpha * (x_r - x_l)


def shock_pos_err(pred: np.ndarray, truth: np.ndarray, x: np.ndarray) -> float:
    """Mean continuous shock-position error over timesteps where GT has shocks."""
    nt = pred.shape[0]
    errs = []
    for k in range(nt):
        gt_x = find_shock_positions(truth[k], x)
        pr_x = find_shock_positions(pred[k], x)
        if gt_x.size == 0:
            continue
        if pr_x.size == 0:
            errs.extend([x[-1] - x[0]] * gt_x.size)
            continue
        for g in gt_x:
            errs.append(float(np.min(np.abs(pr_x - g))))
    return float(np.mean(errs)) if errs else float("nan")


# ─── prediction wrappers ────────────────────────────────────────────────── #
def predict_hypno(
    model: HypNO_ST3, model_cfg: dict, u0: np.ndarray,
    x_grid: torch.Tensor, t_grid: torch.Tensor, device: torch.device,
) -> np.ndarray:
    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"
    u0_t = torch.tensor(u0, dtype=torch.float32, device=device).unsqueeze(0)
    if not skip_ef:
        ef_adj, ef_nonadj = precompute_lwr_edge_features_v3(u0_t, x_grid, stencil_k_x)
        ef_adj = ef_adj.to(device)
        ef_nonadj = ef_nonadj.to(device) if ef_nonadj is not None and ef_nonadj.numel() > 0 else None
    else:
        ef_adj, ef_nonadj = None, None
    with torch.no_grad():
        pred, _, _, _ = model(u0_t, x_grid, t_grid, edge_feats_adj=ef_adj, edge_feats_nonadj=ef_nonadj)
    return pred[0].cpu().numpy()


# ─── main ───────────────────────────────────────────────────────────────── #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs_root", type=str,
        default=str(ROOT / "runs" / "hypno_st3"),
        help="Directory containing the four ablation run subdirectories.",
    )
    parser.add_argument(
        "--run_dirs", type=str, nargs="*", default=None,
        help="Explicit list of run dirs. If omitted, autodetect 4 variants under runs_root.",
    )
    parser.add_argument(
        "--data_path", type=str,
        default=str(ROOT / "data" / "hyperbolic_dataset.npz"),
    )
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--plot_idx", type=str, default="1547,1753",
                        help="Comma-separated dataset indices to ALWAYS plot, in addition to N random.")
    parser.add_argument("--n_random_plots", type=int, default=5)
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory. Defaults to runs/ablation_eval/<timestamp>/.",
    )
    parser.add_argument(
        "--cfl", type=float, default=0.25,
        help="CFL for the WENO/Godunov reference solves.",
    )
    parser.add_argument(
        "--boundary", type=str, default="ghost",
        help="Boundary condition for the FVM reference solves.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # ── locate runs ────────────────────────────────────────────────────── #
    runs_root = Path(args.runs_root)
    if args.run_dirs:
        run_dirs = [Path(p) for p in args.run_dirs]
    else:
        candidates = [
            d for d in sorted(runs_root.iterdir())
            if d.is_dir() and (d / "config.yaml").exists() and "old" not in d.name.lower()
        ]
        run_dirs = candidates
    if not run_dirs:
        raise SystemExit(f"No run dirs found under {runs_root}")

    # Load all runs and dedupe by variant (keep first occurrence).
    loaded: dict[str, tuple[Path, HypNO_ST3, dict]] = {}
    for d in run_dirs:
        try:
            variant, model, cfg = load_run(d, device)
        except FileNotFoundError as e:
            print(f"  skipping {d.name}: {e}")
            continue
        if variant in loaded:
            print(f"  duplicate variant '{variant}' in {d.name}; keeping {loaded[variant][0].name}")
            continue
        loaded[variant] = (d, model, cfg)

    variants_order = ["baseline", "no_flux", "temporal_time", "pure_pairwise"]
    variants_present = [v for v in variants_order if v in loaded]
    if not variants_present:
        raise SystemExit("No recognised variants among run dirs.")
    print(f"variants found: {variants_present}")

    # ── dataset + GT (from precomputed dataset.u) ──────────────────────── #
    data_path = Path(args.data_path)
    print(f"Loading dataset from {data_path}")
    dataset = load_dataset(data_path)
    x_np = dataset.x
    t_np = dataset.t
    x_min, x_max = float(x_np[0]), float(x_np[-1])
    # The dataset's t-grid is built with linspace(0, t_max, nt) so t[-1] == t_max.
    t_max = float(t_np[-1])
    nt = len(t_np)
    nx = len(x_np)

    # ── sample selection ──────────────────────────────────────────────── #
    # Use the same train/test split convention as eval_vs_numerical.
    base_cfg = next(iter(loaded.values()))[2]
    train_fraction = float(base_cfg["data"].get("train_fraction", 0.8))
    split_seed = int(base_cfg.get("seed", 42))
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(dataset.u.shape[0])
    n_train = int(train_fraction * len(perm))
    test_pool = perm[n_train:]

    # Random N from test pool with our own seed (independent of split seed).
    pick_rng = np.random.default_rng(args.seed)
    n_metric = min(args.n_samples, len(test_pool))
    metric_idx = pick_rng.choice(test_pool, size=n_metric, replace=False)

    # Plot indices: random subset of metric_idx + always-plot indices.
    always_plot = [int(s) for s in args.plot_idx.split(",") if s.strip()]
    n_rand_plot = min(args.n_random_plots, n_metric)
    rand_plot_idx = pick_rng.choice(metric_idx, size=n_rand_plot, replace=False).tolist()
    plot_idx = sorted(set(rand_plot_idx + always_plot))
    print(f"metric samples: {n_metric}, plot samples: {plot_idx}")

    # ── output dir ────────────────────────────────────────────────────── #
    out_dir = (
        Path(args.out_dir) if args.out_dir
        else ROOT / "runs" / "ablation_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    # ── prep torch grids ──────────────────────────────────────────────── #
    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)

    # ── eval loop ─────────────────────────────────────────────────────── #
    # Methods to compare: each variant + WENO5 + Godunov.
    method_names = variants_present + ["WENO5", "Godunov"]
    metric_keys = ["mae", "rmse", "linf", "shock_pos_dx"]
    per_sample: dict[str, dict[str, list[float]]] = {
        m: {k: [] for k in metric_keys} for m in method_names
    }

    dx = x_np[1] - x_np[0]

    all_idx = sorted(set(list(metric_idx) + plot_idx))
    for i, idx in enumerate(all_idx):
        idx = int(idx)
        is_metric = idx in metric_idx
        is_plot = idx in plot_idx
        u0 = dataset.u0[idx]
        gt = dataset.u[idx]   # already on the (nt, nx) grid, Lax-Hopf precomputed

        # WENO5 + Godunov reference solves (fresh each sample).
        _, _, weno = solve_conservation_fvm(
            u0, x_min, x_max, t_max, nt, cfl=args.cfl, boundary=args.boundary, method="weno5",
        )
        _, _, godu = solve_conservation_fvm(
            u0, x_min, x_max, t_max, nt, cfl=args.cfl, boundary=args.boundary, method="godunov",
        )

        # HypNO predictions for each variant.
        preds: dict[str, np.ndarray] = {}
        for variant in variants_present:
            _, model, cfg_v = loaded[variant]
            model_cfg = cfg_v.get("hypno_st3", {})
            preds[variant] = predict_hypno(model, model_cfg, u0, x_grid, t_grid, device)
        preds["WENO5"] = weno
        preds["Godunov"] = godu

        if is_metric:
            for m, pred in preds.items():
                row = metrics_vs_gt(pred, gt)
                row["shock_pos_dx"] = shock_pos_err(pred, gt, x_np) / dx
                for k in metric_keys:
                    per_sample[m][k].append(row[k])

        if is_plot:
            _make_sample_plot(idx, x_np, t_np, gt, preds, variants_present, out_dir)

        if (i + 1) % 10 == 0 or i + 1 == len(all_idx):
            print(f"  processed {i + 1}/{len(all_idx)} samples")

    # ── aggregate and write CSV ───────────────────────────────────────── #
    csv_path = out_dir / "ablation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "mean", "std", "min", "max", "n"])
        for m in method_names:
            for k in metric_keys:
                vals = np.array(per_sample[m][k], dtype=np.float64)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    w.writerow([m, k, "nan", "nan", "nan", "nan", 0])
                    continue
                w.writerow([m, k, f"{vals.mean():.6e}", f"{vals.std():.6e}",
                            f"{vals.min():.6e}", f"{vals.max():.6e}", vals.size])

    # Pretty print to stdout too.
    print(f"\n=== Aggregate metrics over {n_metric} test samples ===")
    header = f"{'method':<16}" + "".join(f"{k:>16}" for k in metric_keys)
    print(header)
    print("-" * len(header))
    for m in method_names:
        cells = [f"{m:<16}"]
        for k in metric_keys:
            vals = np.array(per_sample[m][k], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            cells.append(f"{vals.mean():>16.4e}" if vals.size else f"{'nan':>16}")
        print("".join(cells))
    print(f"\nCSV: {csv_path}")
    print(f"Plots in: {out_dir}")


def _make_sample_plot(
    idx: int,
    x_np: np.ndarray, t_np: np.ndarray,
    gt: np.ndarray,
    preds: dict[str, np.ndarray],
    variants_present: list[str],
    out_dir: Path,
) -> None:
    """Two-row figure: GT + all methods (top), |pred - GT| for each method (bottom)."""
    method_order = variants_present + ["WENO5", "Godunov"]
    n_methods = len(method_order)
    vmin, vmax = float(gt.min()), float(gt.max())
    err_vmax = max(np.abs(preds[m] - gt).max() for m in method_order)

    fig, axes = plt.subplots(
        2, n_methods + 1,
        figsize=(3.0 * (n_methods + 1), 6.5),
        constrained_layout=True,
    )
    # Row 0 col 0: GT.
    im_gt = axes[0, 0].pcolormesh(x_np, t_np, gt, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Lax-Hopf (GT)")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("t")
    fig.colorbar(im_gt, ax=axes[0, 0])
    axes[1, 0].axis("off")
    # Row 0 cols 1..: predictions; Row 1 cols 1..: errors.
    for c, m in enumerate(method_order, start=1):
        im_p = axes[0, c].pcolormesh(x_np, t_np, preds[m], shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        axes[0, c].set_title(m)
        axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
        fig.colorbar(im_p, ax=axes[0, c])
        im_e = axes[1, c].pcolormesh(x_np, t_np, np.abs(preds[m] - gt), shading="auto",
                                     cmap="magma", vmin=0, vmax=err_vmax)
        axes[1, c].set_title(f"|{m} - GT|")
        axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
        fig.colorbar(im_e, ax=axes[1, c])

    # MAE in the suptitle.
    mae_str = "  ".join(
        f"{m}={np.abs(preds[m] - gt).mean():.2e}" for m in method_order
    )
    fig.suptitle(f"Sample {idx} — MAE: {mae_str}", fontsize=9)
    fig.savefig(out_dir / f"ablation_sample_{idx}.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
