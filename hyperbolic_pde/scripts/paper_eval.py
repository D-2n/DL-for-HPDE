"""Paper evaluation: HypNO-ST3 vs WENO5 vs Godunov on the paper_eval grouped set.

Per (ic_type, num_segments) cell, computes mean +/- std MAE vs Lax-Hopf exact
across all samples, and emits:

  * one qualitative figure per cell with a u0(x) strip on top, a row of
    space-time heatmaps (Lax-Hopf | HypNO-ST3 | WENO5 | Godunov), and a row of
    |solver - GT| heatmaps. Title carries ic_type, num_segments, per-segment
    values for discrete ICs.
  * mae_vs_num_segments_<ic_type>.png with mean +/- std error bars per solver.
  * summary.csv (long format) with mae mean/std per (ic_type, num_segments, solver).
  * summary.tex with one tabular per ic_type, columns = solvers (mean +/- std).

No wall-clock measurement.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from hyperbolic_pde.utils.runtime import apply_runtime_overrides

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import solve_conservation_fvm
from hyperbolic_pde.models.hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3
from hyperbolic_pde.models.fno import FNO2d

# Solver order. FNO is appended at runtime only if its checkpoint is found, so
# that runs without a trained FNO still produce HypNO-ST3 / WENO5 / Godunov.
BASE_SOLVERS = ["HypNO-ST3", "WENO5", "Godunov"]
COLORS = {
    "HypNO-ST3": "tab:blue",
    "WENO5": "tab:orange",
    "Godunov": "tab:green",
    "FNO": "tab:red",
}

# Maximum number of segment values to spell out in the figure title before
# truncation; beyond this just write "[v0, v1, ..., v_last]".
_MAX_SEG_VALUES_IN_TITLE = 8


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred - truth).mean())


def load_paper_eval_dataset(path: Path):
    from hyperbolic_pde import resolve_data_path

    path = resolve_data_path(path)
    data = np.load(path, allow_pickle=False)
    if "num_segments" not in data or "ic_type" not in data:
        raise KeyError(
            f"{path} is missing 'num_segments'/'ic_type' arrays. Regenerate it "
            f"with generate_ood_data.py --block paper_eval_data."
        )
    return (
        data["x"],
        data["t"],
        data["u"],
        data["u0"],
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


def make_fno_input(
    u0_np: np.ndarray, x_np: np.ndarray, t_np: np.ndarray
) -> torch.Tensor:
    """Build the (1, 3, nx, nt) input tensor the FNO expects (mirrors FNODataset)."""
    x_t = torch.tensor(x_np, dtype=torch.float32)
    t_t = torch.tensor(t_np, dtype=torch.float32)
    u0_t = torch.tensor(u0_np, dtype=torch.float32)
    X, T = torch.meshgrid(x_t, t_t, indexing="ij")
    u0_grid = u0_t.unsqueeze(1).repeat(1, t_t.numel())
    inp = torch.stack([X, T, u0_grid], dim=0).unsqueeze(0)  # (1, 3, nx, nt)
    return inp


def load_fno(fno_cfg: dict, weights_path: Path, device: torch.device) -> FNO2d:
    model = FNO2d(
        in_channels=3,
        out_channels=1,
        width=int(fno_cfg["width"]),
        modes_x=int(fno_cfg["modes_x"]),
        modes_t=int(fno_cfg["modes_t"]),
        layers=int(fno_cfg["layers"]),
    ).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_segment_values(u0: np.ndarray, atol: float = 1e-6) -> list[float]:
    """Return the sequence of distinct adjacent segment values in u0.

    Works for piecewise-constant outputs (piecewise_constant, sine_staircase,
    riemann). For piecewise-sine the result is meaningless but the caller
    doesn't ask for it in that case.
    """
    values: list[float] = [float(u0[0])]
    for v in u0[1:]:
        if abs(float(v) - values[-1]) > atol:
            values.append(float(v))
    return values


def format_ic_label(u0: np.ndarray, ic_type: str, num_segments: int) -> str:
    """One-line label for the figure title."""
    if ic_type in ("piecewise_constant", "sine_staircase", "riemann"):
        vals = extract_segment_values(u0)
        if len(vals) <= _MAX_SEG_VALUES_IN_TITLE:
            val_str = "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"
        else:
            head = ", ".join(f"{v:.2f}" for v in vals[:3])
            tail = ", ".join(f"{v:.2f}" for v in vals[-2:])
            val_str = f"[{head}, ..., {tail}] (n_distinct={len(vals)})"
        return f"{ic_type}, num_segments={num_segments}, values={val_str}"
    return f"{ic_type}, num_segments={num_segments}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper evaluation: HypNO-ST3 vs WENO5 vs Godunov on the paper_eval set."
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Run directory containing config.yaml and model_final.pt.",
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Override the paper_eval dataset path (defaults to cfg['paper_eval_data']['path']).",
    )
    parser.add_argument(
        "--n_per_cell", type=int, default=None,
        help="Cap on samples evaluated per (ic_type, num_segments) cell (default: all).",
    )
    parser.add_argument(
        "--out_dir", type=str, default="paper_eval",
        help="Output subdirectory under run-dir (default: paper_eval).",
    )
    parser.add_argument(
        "--fno-weights", type=str, default=None,
        help="Path to trained FNO checkpoint. Defaults to cfg['fno']['save_path']. "
             "On CLEPS the trained weights are at "
             "hyperbolic_pde/runs/fno_epoch200.pt. If neither is found, FNO is "
             "skipped and only HypNO-ST3 / WENO5 / Godunov are reported.",
    )
    parser.add_argument(
        "--no-fno", action="store_true",
        help="Skip the FNO baseline even if a checkpoint exists.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_runtime_overrides(cfg)
    print(f"Using config from {run_dir / 'config.yaml'}")

    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st")))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    if args.data_path:
        data_path = Path(args.data_path)
    else:
        pe_cfg = cfg.get("paper_eval_data")
        if not pe_cfg or "path" not in pe_cfg:
            raise KeyError(
                "No --data_path given and config has no 'paper_eval_data: path:'. "
                "Pass --data_path explicitly or add the block to the config."
            )
        data_path = Path(pe_cfg["path"])
    print(f"Loading paper-eval dataset from {data_path}")
    x_np, t_np, u_gt, u0_all, seg_all, ictype_all = load_paper_eval_dataset(data_path)

    x_min = float(x_np[0] - 0.5 * (x_np[1] - x_np[0]))
    x_max = float(x_np[-1] + 0.5 * (x_np[1] - x_np[0]))
    t_max = float(t_np[-1])
    nt = len(t_np)
    data_cfg = cfg["data"]
    cfl = float(data_cfg.get("cfl", 0.25))
    boundary = str(data_cfg.get("boundary", "ghost"))

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

    # ---- Optional FNO baseline --------------------------------------- #
    # Resolve the FNO checkpoint: explicit --fno-weights wins, else
    # cfg['fno']['save_path']. The CLEPS-trained weights live at
    # hyperbolic_pde/runs/fno_epoch200.pt. If nothing is found (or --no-fno),
    # FNO is dropped from the solver list and the script behaves as before.
    fno_model = None
    fno_cfg = cfg.get("fno")
    if not args.no_fno and fno_cfg is not None:
        if args.fno_weights is not None:
            fno_weights = Path(args.fno_weights)
        else:
            fno_weights = Path(fno_cfg.get("save_path", "hyperbolic_pde/runs/fno.pt"))
        if fno_weights.exists():
            fno_model = load_fno(fno_cfg, fno_weights, device)
            print(f"Loaded FNO from {fno_weights}")
        else:
            print(
                f"FNO checkpoint not found at {fno_weights}; skipping FNO baseline. "
                f"Pass --fno-weights to point at it (CLEPS: "
                f"hyperbolic_pde/runs/fno_epoch200.pt)."
            )
    elif args.no_fno:
        print("--no-fno given; skipping FNO baseline.")

    SOLVERS = list(BASE_SOLVERS) + (["FNO"] if fno_model is not None else [])

    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)
    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"

    out_dir = run_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ic_types = sorted(set(ictype_all.tolist()))
    seg_values = sorted(set(int(s) for s in seg_all))
    print(f"ic_types: {ic_types}")
    print(f"num_segments groups: {seg_values}")

    # ID/OOD classification: a (ic_type, num_segments) cell is in-distribution
    # iff its ic_type appears in cfg['data']['ic_types'] AND its num_segments
    # appears in cfg['data']['num_segments'] from the run config.
    train_ic_types: set[str] = set(str(s) for s in cfg["data"].get("ic_types", []))
    train_segs: set[int] = set(int(s) for s in cfg["data"].get("num_segments", []))

    def cell_is_id(ic_type: str, seg: int) -> bool:
        return (ic_type in train_ic_types) and (seg in train_segs)

    id_cells = [(ic, seg) for ic in ic_types for seg in seg_values if cell_is_id(ic, seg)]
    ood_cells = [(ic, seg) for ic in ic_types for seg in seg_values if not cell_is_id(ic, seg)]
    print(f"train ic_types: {sorted(train_ic_types)}")
    print(f"train num_segments: {sorted(train_segs)}")
    print(f"ID cells ({len(id_cells)}):  {id_cells}")
    print(f"OOD cells ({len(ood_cells)}): {ood_cells}")

    # mae_by_cell[(ic_type, seg)][solver] -> list of per-sample MAE
    mae_by_cell: dict[tuple[str, int], dict[str, list[float]]] = {
        (ic, seg): {s: [] for s in SOLVERS}
        for ic in ic_types for seg in seg_values
    }
    rep_by_cell: dict[tuple[str, int], dict] = {}

    def run_hypno(u0_np: np.ndarray) -> np.ndarray:
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
        return pred_t[0].cpu().numpy()

    def run_fno(u0_np: np.ndarray) -> np.ndarray:
        inp = make_fno_input(u0_np, x_np, t_np).to(device)
        with torch.no_grad():
            pred = fno_model(inp)  # (1, 1, nx, nt)
        # FNO predicts in (nx, nt) layout; transpose to (nt, nx) like the GT.
        return pred[0, 0].cpu().numpy().T

    n_total = u0_all.shape[0]
    for idx in range(n_total):
        seg = int(seg_all[idx])
        ic_type = str(ictype_all[idx])
        cell = (ic_type, seg)
        if cell not in mae_by_cell:
            continue
        done = len(mae_by_cell[cell]["HypNO-ST3"])
        if args.n_per_cell is not None and done >= args.n_per_cell:
            continue

        u0_np = u0_all[idx]
        lh = u_gt[idx]

        print(f"[{idx + 1}/{n_total}] ic_type={ic_type}, num_segments={seg} ...", flush=True)

        _, _, weno = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="weno5"
        )
        _, _, godunov = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="godunov"
        )
        hypno_np = run_hypno(u0_np)

        preds = {"HypNO-ST3": hypno_np, "WENO5": weno, "Godunov": godunov}
        if fno_model is not None:
            preds["FNO"] = run_fno(u0_np)
        for name, pred in preds.items():
            mae_by_cell[cell][name].append(mae(pred, lh))

        if cell not in rep_by_cell:
            rep_by_cell[cell] = {
                "idx": idx,
                "u0": u0_np,
                "lh": lh,
                "preds": preds,
            }

    # ---------------------------------------------------------------- #
    # Qualitative figures: one per (ic_type, num_segments) cell
    # ---------------------------------------------------------------- #
    plots_dir = out_dir / "figures"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for (ic_type, seg), rep in rep_by_cell.items():
        u0_np = rep["u0"]
        lh = rep["lh"]
        preds = rep["preds"]
        idx = rep["idx"]
        vmin, vmax = float(lh.min()), float(lh.max())
        panels = [("Lax-Hopf (GT)", lh)] + [(n, preds[n]) for n in SOLVERS]

        # 3-row layout: IC strip (thin) + heatmap row + error row.
        fig = plt.figure(figsize=(4 * len(panels), 9), constrained_layout=True)
        gs = fig.add_gridspec(
            nrows=3, ncols=len(panels),
            height_ratios=[0.6, 3.0, 3.0],
        )

        # Row 0: u0(x) strip spanning all columns.
        ax_ic = fig.add_subplot(gs[0, :])
        ax_ic.plot(x_np, u0_np, color="black", linewidth=1.4)
        ax_ic.set_ylabel("u0(x)")
        ax_ic.set_xlabel("x")
        ax_ic.set_xlim(float(x_np[0]), float(x_np[-1]))
        u_pad = 0.05 * (vmax - vmin + 1e-6)
        ax_ic.set_ylim(vmin - u_pad, vmax + u_pad)
        ax_ic.grid(True, alpha=0.3)

        # Row 1: space-time heatmaps.
        sol_axes = [fig.add_subplot(gs[1, c]) for c in range(len(panels))]
        for c, (name, sol) in enumerate(panels):
            im = sol_axes[c].pcolormesh(
                x_np, t_np, sol, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
            )
            sol_axes[c].set_title(name)
            sol_axes[c].set_xlabel("x")
            sol_axes[c].set_ylabel("t")
            fig.colorbar(im, ax=sol_axes[c], label="u")

        # Row 2: error heatmaps for solver columns (col 0 stays blank).
        err_axes = [fig.add_subplot(gs[2, c]) for c in range(len(panels))]
        err_vmax = max(np.abs(preds[n] - lh).max() for n in SOLVERS)
        for c, (name, sol) in enumerate(panels):
            if c == 0:
                err_axes[c].axis("off")
                continue
            err = np.abs(sol - lh)
            im = err_axes[c].pcolormesh(
                x_np, t_np, err, shading="auto", cmap="magma", vmin=0, vmax=err_vmax
            )
            err_axes[c].set_title(f"|{name} - GT|")
            err_axes[c].set_xlabel("x")
            err_axes[c].set_ylabel("t")
            fig.colorbar(im, ax=err_axes[c])

        mae_str = "  ".join(f"{n}={mae(preds[n], lh):.3e}" for n in SOLVERS)
        ic_label = format_ic_label(u0_np, ic_type, seg)
        fig.suptitle(
            f"paper_eval sample {idx} | {ic_label}\nMAE: {mae_str}",
            fontsize=11,
        )
        fig.savefig(plots_dir / f"compare_{ic_type}_seg{seg:02d}.png", dpi=150)
        plt.close(fig)

    # ---------------------------------------------------------------- #
    # MAE vs num_segments, one plot per ic_type
    # ---------------------------------------------------------------- #
    for ic_type in ic_types:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for name in SOLVERS:
            means = [np.mean(mae_by_cell[(ic_type, seg)][name]) for seg in seg_values]
            stds = [np.std(mae_by_cell[(ic_type, seg)][name]) for seg in seg_values]
            ax.errorbar(
                seg_values, means, yerr=stds, marker="o", capsize=4,
                label=name, color=COLORS[name],
            )
        ax.set_xlabel("num_segments")
        ax.set_ylabel("MAE vs Lax-Hopf exact (mean +/- std)")
        ax.set_yscale("log")
        ax.set_xticks(seg_values)
        ax.set_title(f"MAE vs num_segments: {ic_type}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / f"mae_vs_num_segments_{ic_type}.png", dpi=150)
        plt.close(fig)

    # ---------------------------------------------------------------- #
    # Pooled aggregates: grand total + per-num_segments marginals
    # ---------------------------------------------------------------- #
    # pooled_all[name] -> list of per-sample MAEs across the whole eval set
    pooled_all: dict[str, list[float]] = {name: [] for name in SOLVERS}
    # pooled_by_seg[seg][name] -> per-sample MAEs across all ic_types for this seg
    pooled_by_seg: dict[int, dict[str, list[float]]] = {
        seg: {name: [] for name in SOLVERS} for seg in seg_values
    }
    # Pooled per ID/OOD group.
    pooled_id: dict[str, list[float]] = {name: [] for name in SOLVERS}
    pooled_ood: dict[str, list[float]] = {name: [] for name in SOLVERS}
    for (ic_type, seg), solver_dict in mae_by_cell.items():
        is_id = cell_is_id(ic_type, seg)
        for name, vals in solver_dict.items():
            pooled_all[name].extend(vals)
            pooled_by_seg[seg][name].extend(vals)
            (pooled_id if is_id else pooled_ood)[name].extend(vals)

    # ---------------------------------------------------------------- #
    # summary.csv (long format) -- per-cell + pooled rows
    # ---------------------------------------------------------------- #
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ic_type", "num_segments", "group", "solver",
            "mae_mean", "mae_std", "n_samples",
        ])
        for ic_type in ic_types:
            for seg in seg_values:
                cell = (ic_type, seg)
                group = "ID" if cell_is_id(ic_type, seg) else "OOD"
                for name in SOLVERS:
                    vals = mae_by_cell[cell][name]
                    if not vals:
                        continue
                    writer.writerow([
                        ic_type, seg, group, name,
                        f"{np.mean(vals):.6e}", f"{np.std(vals):.6e}", len(vals),
                    ])
        # Per-num_segments marginals (pooled across ic_types).
        for seg in seg_values:
            for name in SOLVERS:
                vals = pooled_by_seg[seg][name]
                if not vals:
                    continue
                writer.writerow([
                    "ALL", seg, "ALL", name,
                    f"{np.mean(vals):.6e}", f"{np.std(vals):.6e}", len(vals),
                ])
        # ID and OOD pooled aggregates.
        for group_name, pool in [("ID", pooled_id), ("OOD", pooled_ood)]:
            for name in SOLVERS:
                vals = pool[name]
                if not vals:
                    continue
                writer.writerow([
                    "ALL", "ALL", group_name, name,
                    f"{np.mean(vals):.6e}", f"{np.std(vals):.6e}", len(vals),
                ])
        # Grand total.
        for name in SOLVERS:
            vals = pooled_all[name]
            if not vals:
                continue
            writer.writerow([
                "ALL", "ALL", "ALL", name,
                f"{np.mean(vals):.6e}", f"{np.std(vals):.6e}", len(vals),
            ])
    print(f"Wrote {csv_path}")

    # ---------------------------------------------------------------- #
    # summary.tex (one tabular per ic_type)
    # ---------------------------------------------------------------- #
    tex_path = out_dir / "summary.tex"
    lines: list[str] = []
    lines.append("% Paper evaluation: MAE vs Lax-Hopf exact (mean +/- std).")
    lines.append("% One tabular per IC type. Rows = num_segments, columns = solvers.")
    lines.append("")
    for ic_type in ic_types:
        lines.append(f"% --- {ic_type} ---")
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{MAE vs Lax-Hopf exact, mean $\pm$ std over 20 samples per cell. "
            f"IC type: \\texttt{{{ic_type}}}.}}"
        )
        lines.append(f"\\label{{tab:paper_eval_{ic_type}}}")
        lines.append(r"\begin{tabular}{l" + "c" * len(SOLVERS) + "}")
        lines.append(r"\hline")
        header = "num\\_segments & " + " & ".join(SOLVERS) + r" \\"
        lines.append(header)
        lines.append(r"\hline")
        for seg in seg_values:
            cell = (ic_type, seg)
            row_cells = [str(seg)]
            for name in SOLVERS:
                vals = mae_by_cell[cell][name]
                if not vals:
                    row_cells.append("--")
                else:
                    row_cells.append(
                        f"${np.mean(vals):.2e} \\pm {np.std(vals):.2e}$"
                    )
            lines.append(" & ".join(row_cells) + r" \\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    # --- HEADLINE table for the main text: ID / OOD / all (3 rows) ----------
    lines.append("% --- headline summary (main text): ID / OOD / all ---")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Headline result: MAE vs Lax-Hopf exact (mean $\pm$ std). "
        r"ID = (ic\_type, num\_segments) cells whose ic\_type and num\_segments "
        r"both appear in the training configuration; OOD = the rest of the "
        r"evaluation set.}"
    )
    lines.append(r"\label{tab:paper_eval_headline}")
    lines.append(r"\begin{tabular}{l" + "c" * len(SOLVERS) + "}")
    lines.append(r"\hline")
    lines.append("Subset & " + " & ".join(SOLVERS) + r" \\")
    lines.append(r"\hline")
    for group_label, pool in [("ID", pooled_id), ("OOD", pooled_ood), (r"\textbf{all}", pooled_all)]:
        row_cells = [group_label]
        for name in SOLVERS:
            vals = pool[name]
            if not vals:
                row_cells.append("--")
            elif group_label.startswith(r"\textbf"):
                row_cells.append(
                    f"$\\mathbf{{{np.mean(vals):.2e} \\pm {np.std(vals):.2e}}}$"
                )
            else:
                row_cells.append(f"${np.mean(vals):.2e} \\pm {np.std(vals):.2e}$")
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Appendix aggregate table: per-num_segments + ID/OOD + grand total ---
    lines.append("% --- appendix aggregate (per-num_segments breakdown) ---")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Appendix: per-num\_segments MAE breakdown, mean $\pm$ std, "
        r"pooled across all IC types. ID and OOD rows match Table~\ref{tab:paper_eval_headline}; "
        r"the final row pools the entire evaluation set.}"
    )
    lines.append(r"\label{tab:paper_eval_appendix}")
    lines.append(r"\begin{tabular}{l" + "c" * len(SOLVERS) + "}")
    lines.append(r"\hline")
    lines.append("num\\_segments & " + " & ".join(SOLVERS) + r" \\")
    lines.append(r"\hline")
    for seg in seg_values:
        row_cells = [str(seg)]
        for name in SOLVERS:
            vals = pooled_by_seg[seg][name]
            if not vals:
                row_cells.append("--")
            else:
                row_cells.append(f"${np.mean(vals):.2e} \\pm {np.std(vals):.2e}$")
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\hline")
    # ID / OOD pooled rows.
    for group_label, pool in [("ID", pooled_id), ("OOD", pooled_ood)]:
        row_cells = [group_label]
        for name in SOLVERS:
            vals = pool[name]
            if not vals:
                row_cells.append("--")
            else:
                row_cells.append(f"${np.mean(vals):.2e} \\pm {np.std(vals):.2e}$")
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\hline")
    # Grand total row.
    row_cells = [r"\textbf{all}"]
    for name in SOLVERS:
        vals = pooled_all[name]
        if not vals:
            row_cells.append("--")
        else:
            row_cells.append(
                f"$\\mathbf{{{np.mean(vals):.2e} \\pm {np.std(vals):.2e}}}$"
            )
    lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {tex_path}")

    # ---------------------------------------------------------------- #
    # Plain-text summary (handy for terminal scanning)
    # ---------------------------------------------------------------- #
    txt_path = out_dir / "summary.txt"
    txt_lines = ["paper_eval -- MAE vs Lax-Hopf, mean +/- std\n"]
    for ic_type in ic_types:
        txt_lines.append(f"ic_type = {ic_type}")
        for seg in seg_values:
            cell = (ic_type, seg)
            n = len(mae_by_cell[cell]["HypNO-ST3"])
            txt_lines.append(f"  num_segments = {seg}  ({n} samples)")
            for name in SOLVERS:
                vals = mae_by_cell[cell][name]
                if not vals:
                    continue
                txt_lines.append(
                    f"    {name:12s}: mean={np.mean(vals):.4e}  std={np.std(vals):.4e}"
                )
        txt_lines.append("")

    # Pooled marginals + ID/OOD + grand total.
    txt_lines.append("=" * 60)
    txt_lines.append("AGGREGATE (pooled across ic_types)")
    for seg in seg_values:
        n = len(pooled_by_seg[seg]["HypNO-ST3"])
        txt_lines.append(f"  num_segments = {seg}  ({n} samples)")
        for name in SOLVERS:
            vals = pooled_by_seg[seg][name]
            if not vals:
                continue
            txt_lines.append(
                f"    {name:12s}: mean={np.mean(vals):.4e}  std={np.std(vals):.4e}"
            )
    txt_lines.append("")
    txt_lines.append(
        f"ID cells ({len(id_cells)}): {id_cells}"
    )
    txt_lines.append(
        f"  ({len(pooled_id['HypNO-ST3'])} samples)"
    )
    for name in SOLVERS:
        vals = pooled_id[name]
        if not vals:
            continue
        txt_lines.append(
            f"  {name:12s}: mean={np.mean(vals):.4e}  std={np.std(vals):.4e}"
        )
    txt_lines.append("")
    txt_lines.append(
        f"OOD cells ({len(ood_cells)}): {ood_cells}"
    )
    txt_lines.append(
        f"  ({len(pooled_ood['HypNO-ST3'])} samples)"
    )
    for name in SOLVERS:
        vals = pooled_ood[name]
        if not vals:
            continue
        txt_lines.append(
            f"  {name:12s}: mean={np.mean(vals):.4e}  std={np.std(vals):.4e}"
        )
    txt_lines.append("")
    txt_lines.append(
        f"GRAND TOTAL  ({len(pooled_all['HypNO-ST3'])} samples across all cells)"
    )
    for name in SOLVERS:
        vals = pooled_all[name]
        if not vals:
            continue
        txt_lines.append(
            f"  {name:12s}: mean={np.mean(vals):.4e}  std={np.std(vals):.4e}"
        )
    txt_lines.append("")

    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"Wrote {txt_path}")

    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
