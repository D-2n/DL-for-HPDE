"""Shock-localized OOD comparison: HypNO-ST3 vs WENO5 vs Godunov.

Companion to ``final_comparison.py``. In addition to the regular full-domain
metrics, this script isolates shock neighborhoods on the Lax-Hopf ground
truth and reports MAE there, so accuracy *around shocks* can be compared
directly across methods.

What it produces, per run-dir ``plots_shock_comparison/``:

  * ``shock_mae_vs_num_segments.png`` — average shock-region MAE per method,
    binned by ``num_segments`` (the OOD complexity axis).
  * ``shock_mae_ratio_vs_num_segments.png`` — shock-MAE / full-domain MAE
    per method, exposing whether the shock region dominates the error.
  * One ``shock_compare_num_segments_{seg}.png`` per OOD group: solution +
    shock-masked absolute-error fields for every method, with the detected
    shock band overlaid on the ground truth.
  * ``shock_summary.txt`` — full + shock-only MAE table per (seg, method).

Shock detection runs on the Lax-Hopf field:

    grad_x[k, i] ~ |u[k, i+1] - u[k, i-1]| / (2*dx)
    raw_mask     = (grad_x * dx) > jump_threshold
    shock_mask   = binary_dilate(raw_mask, band_cells) along x

That is: any cell with a (cell-scale) jump above ``--jump-threshold`` flags
the region, then the mask is widened by ``--band-cells`` cells on each side
along x to form a shock *neighborhood*. The same mask is reused for every
method, so the comparison is apples-to-apples.
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
from matplotlib.colors import ListedColormap
from hyperbolic_pde.utils.runtime import apply_runtime_overrides

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import solve_conservation_fvm
from hyperbolic_pde.models.hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3
from hyperbolic_pde.scripts.final_comparison import (
    SOLVERS,
    COLORS,
    load_ood_dataset,
    build_model,
    mae,
)


# ----------------------------------------------------------------------------
# Shock-mask utilities
# ----------------------------------------------------------------------------
def detect_shock_mask(
    u: np.ndarray,
    dx: float,
    jump_threshold: float,
    band_cells: int,
    tv_gate: bool = True,
    tv_multiplier: float = 2.0,
) -> np.ndarray:
    """Return a boolean (nt, nx) mask of shock neighborhoods on ``u``.

    Gate 1 (per-cell jump): half-central difference
    ``|u[i+1] - u[i-1]| / 2 > jump_threshold``. This is the dimensionless
    per-cell u-jump magnitude — ``dx`` is accepted only for downstream
    reporting (band width in physical units) and is intentionally not used
    in the detection itself.

    Gate 2 (local TV, optional): a cell is kept only if the sum of
    ``|Δu|`` in a ±``band_cells`` window exceeds
    ``tv_multiplier * jump_threshold``. This suppresses isolated smooth-region
    blips that happen to clear gate 1: a real shock has *several* large
    neighbouring deltas, smoothly-varying noise typically does not.

    Step 3 (dilate): 1D binary dilation along x by ``band_cells`` cells on
    either side, so we report the *neighborhood* around each detected
    discontinuity rather than a single-cell edge.
    """
    if u.ndim != 2:
        raise ValueError(f"detect_shock_mask expects (nt, nx), got {u.shape}")
    _ = dx  # see docstring

    # ---- Gate 1: per-cell jump magnitude ------------------------------- #
    u_pad = np.pad(u, ((0, 0), (1, 1)), mode="edge")
    half_diff = (u_pad[:, 2:] - u_pad[:, :-2]) * 0.5
    raw = np.abs(half_diff) > float(jump_threshold)

    # ---- Gate 2: local total variation -------------------------------- #
    if tv_gate:
        # |Δu| at face i+1/2 lives at index i in `du` (length nx-1 along x);
        # pad to length nx so a windowed sum lines up with cell indices.
        du = np.abs(u[:, 1:] - u[:, :-1])
        du_pad = np.pad(du, ((0, 0), (0, 1)), mode="edge")  # (nt, nx)
        # Sliding sum over a (2*band_cells+1) window via cumulative sum.
        w = max(int(band_cells), 1)
        cumsum = np.cumsum(du_pad, axis=1)
        # window sum at column i = cumsum[i+w] - cumsum[i-w-1] (with clipping)
        nx = du_pad.shape[1]
        lo = np.clip(np.arange(nx) - w - 1, -1, nx - 1)
        hi = np.clip(np.arange(nx) + w, 0, nx - 1)
        # cumsum[-1] would wrap; treat lo == -1 as 0
        lo_vals = np.where(lo[None, :] >= 0, cumsum[:, lo.clip(min=0)], 0.0)
        hi_vals = cumsum[:, hi]
        local_tv = hi_vals - lo_vals
        tv_pass = local_tv > float(tv_multiplier) * float(jump_threshold)
        raw = raw & tv_pass

    if band_cells <= 0:
        return raw

    # ---- Step 3: dilate by band_cells along x -------------------------- #
    band = int(band_cells)
    out = raw.copy()
    for shift in range(1, band + 1):
        out[:, shift:] |= raw[:, :-shift]
        out[:, :-shift] |= raw[:, shift:]
    return out


def masked_mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    """MAE restricted to ``mask``. Returns NaN if mask is empty."""
    if not mask.any():
        return float("nan")
    return float(np.abs(pred[mask] - truth[mask]).mean())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Shock-localized OOD comparison of HypNO-ST3 vs WENO5 vs Godunov. "
            "Reports MAE on shock neighborhoods detected from the Lax-Hopf "
            "ground truth, alongside full-domain MAE."
        )
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
    parser.add_argument(
        "--jump-threshold", type=float, default=0.15,
        help="Cell-scale jump magnitude that flags a shock cell "
             "(|u[i+1]-u[i-1]|/2 > threshold). Default 0.15.",
    )
    parser.add_argument(
        "--band-cells", type=int, default=2,
        help="Dilate the raw shock mask by this many cells on each side "
             "to form a shock *neighborhood*. Default 2 (=> 5-cell band).",
    )
    parser.add_argument(
        "--no-tv-gate", action="store_true",
        help="Disable the local-TV second gate (kept on by default).",
    )
    parser.add_argument(
        "--tv-multiplier", type=float, default=2.0,
        help="Local TV must exceed (tv_multiplier * jump_threshold) for a "
             "cell to pass the TV gate. Default 2.0.",
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
    dx = float(x_np[1] - x_np[0])
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

    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)
    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"

    plot_dir = run_dir / "plots_shock_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)

    seg_values = sorted(set(int(s) for s in seg_all))
    print(f"num_segments groups: {seg_values}")
    tv_gate = not args.no_tv_gate
    print(
        f"Shock detector: jump_threshold={args.jump_threshold}, "
        f"band_cells={args.band_cells} (neighborhood width {2*args.band_cells+1} cells), "
        f"tv_gate={'on' if tv_gate else 'off'}"
        + (f" (multiplier={args.tv_multiplier})" if tv_gate else "")
    )

    # Per-group accumulators (full + shock-restricted MAE)
    mae_full = {seg: {s: [] for s in SOLVERS} for seg in seg_values}
    mae_shock = {seg: {s: [] for s in SOLVERS} for seg in seg_values}
    shock_frac = {seg: [] for seg in seg_values}  # fraction of cells in shock band

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

    # Per-group representative sample for qualitative plots.
    rep_by_seg: dict[int, dict] = {}

    n_total = u0_all.shape[0]
    for idx in range(n_total):
        seg = int(seg_all[idx])
        ic_type = str(ictype_all[idx])
        done = len(mae_full[seg]["HypNO-ST3"])
        if args.n_per_group is not None and done >= args.n_per_group:
            continue

        u0_np = u0_all[idx]
        lh = u_gt[idx]

        # Shock band on the GROUND TRUTH — same mask reused for every method.
        mask = detect_shock_mask(
            lh, dx, args.jump_threshold, args.band_cells,
            tv_gate=tv_gate, tv_multiplier=args.tv_multiplier,
        )
        shock_frac[seg].append(float(mask.mean()))

        print(
            f"[{idx + 1}/{n_total}] num_segments={seg}, ic_type={ic_type}, "
            f"shock_band={mask.mean()*100:.2f}% of cells ...",
            flush=True,
        )

        _, _, weno = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="weno5"
        )
        _, _, godunov = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="godunov"
        )
        hypno_np = run_hypno(u0_np)

        preds = {"HypNO-ST3": hypno_np, "WENO5": weno, "Godunov": godunov}
        for name, pred in preds.items():
            mae_full[seg][name].append(mae(pred, lh))
            mae_shock[seg][name].append(masked_mae(pred, lh, mask))

        if seg not in rep_by_seg:
            rep_by_seg[seg] = {
                "idx": idx, "ic_type": ic_type,
                "lh": lh, "preds": preds, "mask": mask,
            }

    # ---------------------------------------------------------------- #
    # Plot 1: shock-MAE vs num_segments
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name in SOLVERS:
        means = [np.nanmean(mae_shock[seg][name]) for seg in seg_values]
        stds = [np.nanstd(mae_shock[seg][name]) for seg in seg_values]
        ax.errorbar(
            seg_values, means, yerr=stds, marker="o", capsize=4,
            label=name, color=COLORS[name],
        )
    ax.set_xlabel("num_segments (discontinuities in IC)")
    ax.set_ylabel(
        f"average shock-region MAE  "
        f"(band ±{args.band_cells} cells, jump>{args.jump_threshold})"
    )
    ax.set_yscale("log")
    ax.set_xticks(seg_values)
    ax.set_title("OOD accuracy on shock neighborhoods")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(plot_dir / "shock_mae_vs_num_segments.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # Plot 2: shock-MAE / full-MAE ratio vs num_segments
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name in SOLVERS:
        ratios = []
        for seg in seg_values:
            sm = np.nanmean(mae_shock[seg][name])
            fm = np.nanmean(mae_full[seg][name])
            ratios.append(sm / fm if fm > 0 else np.nan)
        ax.plot(seg_values, ratios, marker="o", label=name, color=COLORS[name])
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="parity")
    ax.set_xlabel("num_segments (discontinuities in IC)")
    ax.set_ylabel("shock-MAE / full-MAE")
    ax.set_xticks(seg_values)
    ax.set_title("How concentrated is the error around shocks?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(plot_dir / "shock_mae_ratio_vs_num_segments.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # Plot 3: per-group qualitative comparison + shock band overlay
    # ---------------------------------------------------------------- #
    band_cmap = ListedColormap([(0, 0, 0, 0), (1, 1, 1, 0.35)])  # transparent / white-translucent
    for seg in seg_values:
        rep = rep_by_seg.get(seg)
        if rep is None:
            continue
        lh = rep["lh"]
        preds = rep["preds"]
        mask = rep["mask"]
        ic_type = rep["ic_type"]
        idx = rep["idx"]
        vmin, vmax = float(lh.min()), float(lh.max())

        # Layout:
        #   row 0: GT + each method's solution (with shock band overlay)
        #   row 1: empty, then |method - GT| over full domain
        #   row 2: empty, then |method - GT| masked to shock band only
        ncols = 1 + len(SOLVERS)
        fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 11), constrained_layout=True)

        # Top-left: GT + band
        im = axes[0, 0].pcolormesh(
            x_np, t_np, lh, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
        )
        axes[0, 0].pcolormesh(x_np, t_np, mask.astype(float), shading="auto", cmap=band_cmap)
        axes[0, 0].set_title("Lax-Hopf (GT) + shock band")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("t")
        fig.colorbar(im, ax=axes[0, 0], label="u")
        axes[1, 0].axis("off")
        axes[2, 0].axis("off")

        # Determine shared error vmax across methods for fair comparison.
        full_err_vmax = max(np.abs(preds[n] - lh).max() for n in SOLVERS)
        masked_errs = [np.abs(preds[n] - lh) * mask for n in SOLVERS]
        shock_err_vmax = max(e.max() for e in masked_errs) if mask.any() else full_err_vmax

        for c, name in enumerate(SOLVERS, start=1):
            sol = preds[name]
            err = np.abs(sol - lh)
            err_masked = err * mask

            im = axes[0, c].pcolormesh(
                x_np, t_np, sol, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
            )
            axes[0, c].pcolormesh(x_np, t_np, mask.astype(float), shading="auto", cmap=band_cmap)
            axes[0, c].set_title(name)
            axes[0, c].set_xlabel("x")
            axes[0, c].set_ylabel("t")
            fig.colorbar(im, ax=axes[0, c], label="u")

            im = axes[1, c].pcolormesh(
                x_np, t_np, err, shading="auto", cmap="magma",
                vmin=0, vmax=full_err_vmax,
            )
            axes[1, c].set_title(f"|{name} - GT|  (full)")
            axes[1, c].set_xlabel("x")
            axes[1, c].set_ylabel("t")
            fig.colorbar(im, ax=axes[1, c])

            im = axes[2, c].pcolormesh(
                x_np, t_np, err_masked, shading="auto", cmap="magma",
                vmin=0, vmax=shock_err_vmax,
            )
            axes[2, c].set_title(f"|{name} - GT|  (shock band only)")
            axes[2, c].set_xlabel("x")
            axes[2, c].set_ylabel("t")
            fig.colorbar(im, ax=axes[2, c])

        full_str = "  ".join(f"{n}={mae(preds[n], lh):.3e}" for n in SOLVERS)
        sh_str = "  ".join(
            f"{n}={masked_mae(preds[n], lh, mask):.3e}" for n in SOLVERS
        )
        fig.suptitle(
            f"OOD sample {idx} — IC: {ic_type}, num_segments={seg}\n"
            f"full MAE:  {full_str}\n"
            f"shock MAE: {sh_str}  (band covers {mask.mean()*100:.2f}% of cells)"
        )
        fig.savefig(plot_dir / f"shock_compare_num_segments_{seg}.png", dpi=150)
        plt.close(fig)

    # ---------------------------------------------------------------- #
    # Plot 4: zoomed band-only views, per group
    #
    # Same representative sample, but every panel is cropped to the bounding
    # box of the shock band and non-band cells are blanked out with NaN so
    # the eye is drawn to the band itself. Color scales are computed *over
    # the band only*, so faint within-band errors that get washed out in
    # the full-domain plot become visible.
    #
    # Also a 1D slice plot through the time row with the largest GT shock
    # band, comparing each method's u(x) to the GT around the shock.
    # ---------------------------------------------------------------- #
    for seg in seg_values:
        rep = rep_by_seg.get(seg)
        if rep is None or not rep["mask"].any():
            continue
        lh = rep["lh"]
        preds = rep["preds"]
        mask = rep["mask"]
        ic_type = rep["ic_type"]
        idx = rep["idx"]

        # Bounding box of the band in (t, x) index space.
        t_any = mask.any(axis=1)
        x_any = mask.any(axis=0)
        t_lo, t_hi = int(np.argmax(t_any)), int(len(t_any) - np.argmax(t_any[::-1]))
        x_lo, x_hi = int(np.argmax(x_any)), int(len(x_any) - np.argmax(x_any[::-1]))
        # Tiny pad of 1 cell on each side for context, clipped to grid.
        t_lo = max(t_lo - 1, 0); t_hi = min(t_hi + 1, lh.shape[0])
        x_lo = max(x_lo - 1, 0); x_hi = min(x_hi + 1, lh.shape[1])

        x_z = x_np[x_lo:x_hi]
        t_z = t_np[t_lo:t_hi]
        mask_z = mask[t_lo:t_hi, x_lo:x_hi]
        lh_z = lh[t_lo:t_hi, x_lo:x_hi]

        # Blank non-band cells so they render transparent.
        def blanked(arr2d: np.ndarray) -> np.ndarray:
            out = arr2d.astype(float).copy()
            out[~mask_z] = np.nan
            return out

        # Color scales computed strictly inside the band.
        u_in_band = lh[mask]
        vmin = float(u_in_band.min())
        vmax = float(u_in_band.max())
        err_in_band_max = max(
            float(np.abs(preds[n] - lh)[mask].max()) for n in SOLVERS
        )

        ncols = 1 + len(SOLVERS)
        fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8), constrained_layout=True)

        # Top-left: GT cropped + band-only
        im = axes[0, 0].pcolormesh(
            x_z, t_z, blanked(lh_z), shading="auto", cmap="jet", vmin=vmin, vmax=vmax
        )
        axes[0, 0].set_title("Lax-Hopf (GT) — band only")
        axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("t")
        fig.colorbar(im, ax=axes[0, 0], label="u")
        axes[1, 0].axis("off")

        for c, name in enumerate(SOLVERS, start=1):
            sol_z = preds[name][t_lo:t_hi, x_lo:x_hi]
            err_z = np.abs(preds[name] - lh)[t_lo:t_hi, x_lo:x_hi]

            im = axes[0, c].pcolormesh(
                x_z, t_z, blanked(sol_z), shading="auto", cmap="jet",
                vmin=vmin, vmax=vmax,
            )
            axes[0, c].set_title(f"{name} — band only")
            axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
            fig.colorbar(im, ax=axes[0, c], label="u")

            im = axes[1, c].pcolormesh(
                x_z, t_z, blanked(err_z), shading="auto", cmap="magma",
                vmin=0, vmax=err_in_band_max,
            )
            axes[1, c].set_title(f"|{name} - GT|  (band only)")
            axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
            fig.colorbar(im, ax=axes[1, c])

        sh_str = "  ".join(
            f"{n}={masked_mae(preds[n], lh, mask):.3e}" for n in SOLVERS
        )
        fig.suptitle(
            f"OOD sample {idx} — IC: {ic_type}, num_segments={seg}  (zoom to shock band)\n"
            f"shock MAE: {sh_str}"
        )
        fig.savefig(plot_dir / f"shock_zoom_num_segments_{seg}.png", dpi=150)
        plt.close(fig)

        # ---- 1D slice through the time row with the widest band -------- #
        k_star = int(np.argmax(mask.sum(axis=1)))  # row with most band cells
        # x-range to plot: just the band on that row, padded by 3 cells.
        row_mask = mask[k_star]
        if row_mask.any():
            i_lo = max(int(np.argmax(row_mask)) - 3, 0)
            i_hi = min(int(len(row_mask) - np.argmax(row_mask[::-1])) + 3, lh.shape[1])
            xs = x_np[i_lo:i_hi]

            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            ax.plot(xs, lh[k_star, i_lo:i_hi], color="black", lw=2.0, label="Lax-Hopf (GT)")
            for name in SOLVERS:
                ax.plot(
                    xs, preds[name][k_star, i_lo:i_hi],
                    color=COLORS[name], lw=1.4, label=name,
                )
            # Shade the band cells on this row for context.
            band_xs = xs[row_mask[i_lo:i_hi]]
            if band_xs.size:
                ax.axvspan(
                    band_xs[0] - 0.5 * dx, band_xs[-1] + 0.5 * dx,
                    color="orange", alpha=0.15, label="shock band",
                )
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.set_title(
                f"OOD sample {idx} — IC: {ic_type}, num_segments={seg}\n"
                f"slice at t={t_np[k_star]:.3f} (widest band row, "
                f"{int(row_mask.sum())} cells)"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(plot_dir / f"shock_slice_num_segments_{seg}.png", dpi=150)
            plt.close(fig)

    # ---------------------------------------------------------------- #
    # Summary table
    # ---------------------------------------------------------------- #
    summary_path = plot_dir / "shock_summary.txt"
    lines = [
        "OOD shock-localized comparison",
        f"  jump_threshold = {args.jump_threshold}",
        f"  band_cells     = {args.band_cells}  (neighborhood width "
        f"{2*args.band_cells+1} cells, dx={dx:.4g})",
        f"  tv_gate        = {'on' if tv_gate else 'off'}"
        + (f"  (multiplier={args.tv_multiplier})" if tv_gate else ""),
        "",
    ]
    for seg in seg_values:
        n = len(mae_full[seg]["HypNO-ST3"])
        sf = float(np.mean(shock_frac[seg])) if shock_frac[seg] else float("nan")
        lines.append(
            f"num_segments = {seg}  ({n} samples, "
            f"avg shock-band coverage {sf*100:.2f}%)"
        )
        for name in SOLVERS:
            mf = mae_full[seg][name]
            ms = mae_shock[seg][name]
            lines.append(
                f"  {name:12s}: full MAE mean={np.nanmean(mf):.4e} std={np.nanstd(mf):.4e}  |  "
                f"shock MAE mean={np.nanmean(ms):.4e} std={np.nanstd(ms):.4e}"
            )
        lines.append("")
    summary = "\n".join(lines)
    print("\n" + summary)
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Saved plots and summary to {plot_dir}")


if __name__ == "__main__":
    main()
