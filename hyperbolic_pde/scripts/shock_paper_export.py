"""Paper-ready export of shock-localized OOD comparison results.

Re-runs the same data path as ``shock_comparison.py`` (HypNO-ST3 vs WENO5
vs Godunov on the grouped OOD set, with shock-band detection on the
Lax-Hopf ground truth), then writes:

    <run-dir>/paper_shock/
        figs/
            shock_mae_vs_num_segments.png
            shock_compare_num_segments_{seg}.png   (one per seg)
            shock_zoom_num_segments_{seg}.png      (one per seg)
            shock_slice_num_segments_{seg}.png     (one per seg)
        shock_results.tex     -- standalone \\input-able section
        shock_table.tex       -- table snippet only (for fine-grained \\input)
        shock_summary.txt     -- machine-readable per-method per-seg MAE

The .tex emits one wide table with rows = num_segments and columns =
(full MAE, shock MAE) × method, plus \\begin{figure} blocks for the
headline curve and per-segment qualitative panels. Figure paths are
relative to ``figs/`` so the section can be \\input{}-ed from any
location that puts ``figs/`` next to the .tex file.

The detector, model loader, run_hypno, and the per-group qualitative
plots are imported from ``shock_comparison`` so any change to detection
or rendering propagates here automatically.
"""
from __future__ import annotations

import argparse
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
from hyperbolic_pde.models.hypno_st3 import precompute_lwr_edge_features_v3
from hyperbolic_pde.scripts.final_comparison import (
    SOLVERS, COLORS, load_ood_dataset, build_model, mae,
    build_fno, run_fno, FNO_WEIGHTS_PATH,
)
from hyperbolic_pde.scripts.shock_comparison import (
    detect_shock_mask, masked_mae,
)


# sine_staircase is a subset of the general piecewise-constant family (finite
# discontinuities), so for the paper we report only two IC families: riemann and
# piecewise constant. The shock export pools by num_segments, so this only
# affects the IC label shown in the representative-sample captions/logs.
_FAMILY_ALIASES = {
    "sine_staircase": "piecewise_constant",
    "piecewise_sine": "piecewise_constant",
}


def canonical_family(name: str) -> str:
    return _FAMILY_ALIASES.get(name, name)


# ----------------------------------------------------------------------------
# LaTeX-name mapping
# ----------------------------------------------------------------------------
TEX_NAME = {
    "HypNO-ST3": r"HypNO-ST3",
    "WENO5": r"WENO5",
    "Godunov": r"Godunov",
    "FNO": r"FNO",
}


def fmt_mae(x: float) -> str:
    """Format a MAE value as a LaTeX-friendly scientific literal, e.g. 1.23e-03."""
    if not np.isfinite(x):
        return r"\text{n/a}"
    return f"\\num{{{x:.3e}}}"


# ----------------------------------------------------------------------------
# Plot helpers (paper-tuned defaults; PNG only as requested)
# ----------------------------------------------------------------------------
def configure_paper_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def draw_band_outline(ax, mask: np.ndarray, x_np: np.ndarray, t_np: np.ndarray) -> None:
    """Bright pink (#ff1493) outline of the band, matching shock_comparison.py."""
    m = np.pad(mask.astype(float), 1, mode="constant", constant_values=0.0)
    x_pad = np.concatenate([
        [x_np[0] - (x_np[1] - x_np[0])], x_np,
        [x_np[-1] + (x_np[-1] - x_np[-2])],
    ])
    t_pad = np.concatenate([
        [t_np[0] - (t_np[1] - t_np[0])], t_np,
        [t_np[-1] + (t_np[-1] - t_np[-2])],
    ])
    ax.contour(x_pad, t_pad, m, levels=[0.5], colors="#ff1493", linewidths=1.2)


def plot_mae_vs_segments(
    seg_values, mae_full, mae_shock, fig_path: Path, band_cells: int, jump: float
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for name in SOLVERS:
        means = [np.nanmean(mae_shock[seg][name]) for seg in seg_values]
        stds = [np.nanstd(mae_shock[seg][name]) for seg in seg_values]
        ax.errorbar(
            seg_values, means, yerr=stds, marker="o", capsize=3,
            label=name, color=COLORS[name], linewidth=1.4,
        )
    ax.set_xlabel("num.\\ IC segments")
    ax.set_ylabel("shock-band MAE")
    ax.set_yscale("log")
    ax.set_xticks(seg_values)
    ax.set_title(
        f"OOD shock-band MAE  "
        f"(band $\\pm${band_cells} cells, signed jump $>${jump})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_path)
    plt.close(fig)


def plot_compare_full(
    rep: dict, x_np, t_np, fig_path: Path, dx: float
) -> None:
    """Per-seg full-domain comparison with shock-band outline."""
    lh = rep["lh"]; preds = rep["preds"]; mask = rep["mask"]
    ic_type = rep["ic_type"]; idx = rep["idx"]; seg = rep["seg"]
    vmin, vmax = float(lh.min()), float(lh.max())
    ncols = 1 + len(SOLVERS)
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 6.4), constrained_layout=True)

    im = axes[0, 0].pcolormesh(x_np, t_np, lh, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    draw_band_outline(axes[0, 0], mask, x_np, t_np)
    axes[0, 0].set_title("Lax-Hopf (GT) + shock band")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("t")
    fig.colorbar(im, ax=axes[0, 0], label="u")
    axes[1, 0].axis("off")

    full_err_vmax = max(np.abs(preds[n] - lh).max() for n in SOLVERS)
    for c, name in enumerate(SOLVERS, start=1):
        sol = preds[name]
        err = np.abs(sol - lh)
        im = axes[0, c].pcolormesh(x_np, t_np, sol, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        draw_band_outline(axes[0, c], mask, x_np, t_np)
        axes[0, c].set_title(name)
        axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
        fig.colorbar(im, ax=axes[0, c], label="u")

        im = axes[1, c].pcolormesh(x_np, t_np, err, shading="auto", cmap="magma", vmin=0, vmax=full_err_vmax)
        axes[1, c].set_title(f"|{name} $-$ GT|")
        axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
        fig.colorbar(im, ax=axes[1, c])

    sh_str = "  ".join(f"{n}={masked_mae(preds[n], lh, mask):.2e}" for n in SOLVERS)
    fig.suptitle(
        f"Sample {idx} -- IC: {ic_type}, num_segments={seg}.  "
        f"Shock MAE: {sh_str}"
    )
    fig.savefig(fig_path)
    plt.close(fig)


def plot_zoom(rep: dict, x_np, t_np, fig_path: Path) -> None:
    """Band-only zoom (cropped to the band bbox; non-band cells blanked)."""
    lh = rep["lh"]; preds = rep["preds"]; mask = rep["mask"]
    seg = rep["seg"]; idx = rep["idx"]; ic_type = rep["ic_type"]
    if not mask.any():
        return
    t_any = mask.any(axis=1); x_any = mask.any(axis=0)
    t_lo, t_hi = int(np.argmax(t_any)), int(len(t_any) - np.argmax(t_any[::-1]))
    x_lo, x_hi = int(np.argmax(x_any)), int(len(x_any) - np.argmax(x_any[::-1]))
    t_lo = max(t_lo - 1, 0); t_hi = min(t_hi + 1, lh.shape[0])
    x_lo = max(x_lo - 1, 0); x_hi = min(x_hi + 1, lh.shape[1])
    x_z = x_np[x_lo:x_hi]; t_z = t_np[t_lo:t_hi]
    mask_z = mask[t_lo:t_hi, x_lo:x_hi]

    def blanked(arr2d: np.ndarray) -> np.ndarray:
        out = arr2d.astype(float).copy(); out[~mask_z] = np.nan; return out

    u_in = lh[mask]; vmin = float(u_in.min()); vmax = float(u_in.max())
    err_max = max(float(np.abs(preds[n] - lh)[mask].max()) for n in SOLVERS)
    ncols = 1 + len(SOLVERS)
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 6.0), constrained_layout=True)

    im = axes[0, 0].pcolormesh(x_z, t_z, blanked(lh[t_lo:t_hi, x_lo:x_hi]),
                                shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Lax-Hopf (GT)"); axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("t")
    fig.colorbar(im, ax=axes[0, 0], label="u")
    axes[1, 0].axis("off")
    for c, name in enumerate(SOLVERS, start=1):
        sol_z = preds[name][t_lo:t_hi, x_lo:x_hi]
        err_z = np.abs(preds[name] - lh)[t_lo:t_hi, x_lo:x_hi]
        im = axes[0, c].pcolormesh(x_z, t_z, blanked(sol_z), shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        axes[0, c].set_title(name); axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
        fig.colorbar(im, ax=axes[0, c], label="u")
        im = axes[1, c].pcolormesh(x_z, t_z, blanked(err_z), shading="auto", cmap="magma", vmin=0, vmax=err_max)
        axes[1, c].set_title(f"|{name} $-$ GT|"); axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
        fig.colorbar(im, ax=axes[1, c])

    sh_str = "  ".join(
        f"{n}={masked_mae(preds[n], lh, mask):.2e}" for n in SOLVERS
    )
    fig.suptitle(
        f"Sample {idx} -- IC: {ic_type}, num_segments={seg} (band only)\n"
        f"shock MAE: {sh_str}  (band {mask.mean()*100:.2f}% of cells)"
    )
    fig.savefig(fig_path)
    plt.close(fig)


def plot_slice(rep: dict, x_np, t_np, fig_path: Path, dx: float) -> None:
    lh = rep["lh"]; preds = rep["preds"]; mask = rep["mask"]
    seg = rep["seg"]; idx = rep["idx"]; ic_type = rep["ic_type"]
    if not mask.any():
        return
    # Pick a *random* time row in [0.15, 0.8] * T_max that has band cells.
    # Argmax-of-band-width consistently lands at the first eligible row
    # (band is widest right after the IC and narrows over time), which made
    # every slice look identical at t ~ 0.10 * T_max. Random sampling from
    # a mid-time window avoids both the IC region and the end-of-domain.
    T = float(t_np[-1])
    t_lo_slice, t_hi_slice = 0.15 * T, 0.8 * T
    eligible = (t_np >= t_lo_slice) & (t_np <= t_hi_slice) & mask.any(axis=1)
    if not eligible.any():
        # Fallback: any row past t=0 that has band cells.
        eligible = (np.arange(len(t_np)) > 0) & mask.any(axis=1)
        if not eligible.any():
            return
    eligible_idx = np.flatnonzero(eligible)
    # Deterministic given the global seed (set in main()).
    k_star = int(np.random.choice(eligible_idx))
    row_mask = mask[k_star]
    i_lo = max(int(np.argmax(row_mask)) - 3, 0)
    i_hi = min(int(len(row_mask) - np.argmax(row_mask[::-1])) + 3, lh.shape[1])
    xs = x_np[i_lo:i_hi]
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    ax.plot(xs, lh[k_star, i_lo:i_hi], color="black", lw=2.0, label="Lax-Hopf (GT)")
    for name in SOLVERS:
        ax.plot(xs, preds[name][k_star, i_lo:i_hi],
                color=COLORS[name], lw=1.4, label=name)
    band_xs = xs[row_mask[i_lo:i_hi]]
    if band_xs.size:
        ax.axvspan(band_xs[0] - 0.5 * dx, band_xs[-1] + 0.5 * dx,
                   color="orange", alpha=0.15, label="shock band")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.set_title(f"Slice at t={t_np[k_star]:.3f},  num_segments={seg}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(fig_path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# LaTeX emission
# ----------------------------------------------------------------------------
def emit_table(seg_values, mae_full, mae_shock) -> str:
    """Rows = num_segments, columns = (full, shock) MAE per method."""
    n_methods = len(SOLVERS)
    col_spec = "c" + "cc" * n_methods
    headers_top = " & ".join(
        ["num.\\ seg."] +
        [f"\\multicolumn{{2}}{{c}}{{{TEX_NAME[n]}}}" for n in SOLVERS]
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 2*k}-{3 + 2*k}}}" for k in range(n_methods)
    )
    headers_bot = " & ".join([""] + ["full", "shock"] * n_methods)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{OOD comparison on shock neighborhoods. \emph{full} is MAE "
        r"over the entire space-time domain; \emph{shock} is MAE restricted "
        r"to a shock-band mask detected on the Lax-Hopf ground truth via the "
        r"signed entropy condition $u_L<u_R$, with the band dilated to a "
        r"neighborhood of fixed width.}",
        r"\label{tab:shock-mae}",
        r"\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        headers_top + r" \\",
        cmidrules,
        headers_bot + r" \\",
        r"\midrule",
    ]
    for seg in seg_values:
        cells = [str(seg)]
        for name in SOLVERS:
            mf = np.nanmean(mae_full[seg][name])
            ms = np.nanmean(mae_shock[seg][name])
            cells += [fmt_mae(mf), fmt_mae(ms)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def emit_section(
    table_tex: str, seg_values, jump: float, band_cells: int, n_per_group: int | None,
) -> str:
    """Build the standalone \\input-able section."""
    def fig_block(path: str, caption: str, label: str, width: str = r"\linewidth") -> str:
        return "\n".join([
            r"\begin{figure}[t]",
            r"  \centering",
            f"  \\includegraphics[width={width}]{{figs/{path}}}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            r"\end{figure}",
        ])

    out = []
    out.append("% =========================================================================")
    out.append("% HypNO-ST3 OOD accuracy on shock neighborhoods.")
    out.append("% Standalone section file -- \\input{} into the main document.")
    out.append("% Requires: amsmath, booktabs, graphicx, siunitx.")
    out.append("% =========================================================================\n")
    out.append(r"\section{Accuracy on shock neighborhoods}")
    out.append(r"\label{sec:shock-comparison}")
    out.append("")
    out.append(
        r"To isolate accuracy on the discontinuous portion of the solution, "
        r"we evaluate HypNO-ST3, WENO5, Godunov, and an FNO baseline on the "
        r"grouped out-of-distribution set used in Section~\ref{sec:hypno-st3-2d} "
        r"and report MAE restricted to a \emph{shock band} detected on the "
        r"Lax-Hopf ground truth. A grid cell at $(t,x_i)$ is flagged when "
        r"the signed half-central difference "
        r"$(u(t,x_{i+1})-u(t,x_{i-1}))/2$ exceeds "
        f"the threshold $\\tau={jump}$. The sign requirement encodes the "
        r"Lax entropy condition $u_L<u_R$ for the concave LWR flux "
        r"$f(u)=u(1-u)$, so rarefactions are rejected. A local "
        r"total-variation second gate suppresses isolated smooth-region "
        r"blips, and the surviving cells are dilated by "
        f"$\\pm{band_cells}$ cells along $x$ to form a neighborhood. The "
        r"same mask is reused across methods for an apples-to-apples "
        r"comparison."
    )
    out.append("")
    if n_per_group is not None:
        out.append(f"% Each num_segments group capped at {n_per_group} samples.")
        out.append("")
    out.append(table_tex)
    out.append("")
    out.append(fig_block(
        "shock_mae_vs_num_segments.png",
        r"OOD MAE restricted to the shock band, as a function of the number "
        r"of IC discontinuities. HypNO-ST3 is compared against WENO5, "
        r"Godunov, and an FNO baseline on the same Lax-Hopf ground truth.",
        "fig:shock-mae-vs-seg",
    ))
    for seg in seg_values:
        out.append(fig_block(
            f"shock_compare_num_segments_{seg}.png",
            f"Representative OOD sample at num\\_segments={seg}. Top row: "
            r"solution fields with the detected shock band outlined in pink; "
            r"bottom row: absolute error to the Lax-Hopf ground truth.",
            f"fig:shock-compare-{seg}",
        ))
        out.append(fig_block(
            f"shock_zoom_num_segments_{seg}.png",
            f"Same sample as Fig.~\\ref{{fig:shock-compare-{seg}}}, zoomed "
            r"to the shock-band bounding box with non-band cells blanked. "
            r"Color scales are computed inside the band, so within-band "
            r"differences are not washed out by the full-domain range.",
            f"fig:shock-zoom-{seg}",
        ))
        out.append(fig_block(
            f"shock_slice_num_segments_{seg}.png",
            f"$u(x)$ slice through the widest band row for num\\_segments={seg}. "
            r"The shaded region marks the shock band.",
            f"fig:shock-slice-{seg}",
        ))
    out.append("")
    out.append(r"% End of shock_results.tex")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit paper-ready shock-comparison plots and a .tex section."
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--n_per_group", type=int, default=None)
    parser.add_argument(
        "--jump-threshold", type=float, default=0.06,
        help="Signed cell-scale jump threshold: a cell is flagged when "
             "(u[i+1]-u[i-1])/2 > threshold. The signed (not absolute) form "
             "enforces the Lax entropy condition u_L<u_R for LWR's concave "
             "flux, so rarefactions are rejected. Default 0.06.",
    )
    parser.add_argument("--band-cells", type=int, default=2)
    parser.add_argument("--no-tv-gate", action="store_true")
    parser.add_argument("--tv-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for full determinism (torch + numpy + cuDNN). "
             "Default 0; pass any int to vary.",
    )
    parser.add_argument(
        "--min-rep-band-frac", type=float, default=0.05,
        help="Minimum shock-band coverage (fraction of cells) for a sample "
             "to be eligible as the representative plot for its num_segments "
             "group. The first eligible sample in index order wins; if no "
             "sample in a group qualifies, the first sample is used as a "
             "fallback. Default 0.05 (=5%).",
    )
    args = parser.parse_args()

    # --- Full determinism: paper numbers must be bit-stable across reruns. ---
    # Inference still uses cuDNN, so we pin its kernel selection (deterministic=True,
    # benchmark=False) on top of the torch/numpy seeds. WENO5/Godunov/the detector
    # are pure numpy and already deterministic; this guards the HypNO-ST3 forward.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Determinism: seed={args.seed}, cudnn.deterministic=True, benchmark=False")

    configure_paper_style()
    tv_gate = not args.no_tv_gate

    run_dir = Path(args.run_dir)
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_runtime_overrides(cfg)

    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st")))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    if args.data_path:
        data_path = Path(args.data_path)
    else:
        ood_cfg = cfg.get("ood_data") or {}
        if "path" not in ood_cfg:
            raise KeyError("Pass --data_path or set cfg['ood_data']['path'].")
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

    model = build_model(model_cfg, device)
    weights_path = run_dir / "model_final.pt"
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # FNO baseline — hardcoded checkpoint path (see final_comparison.FNO_WEIGHTS_PATH).
    fno_model = build_fno(device)
    print(f"Loaded FNO from {FNO_WEIGHTS_PATH}")

    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)
    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"

    out_dir = run_dir / "paper_shock"
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    seg_values = sorted(set(int(s) for s in seg_all))
    mae_full = {seg: {s: [] for s in SOLVERS} for seg in seg_values}
    mae_shock = {seg: {s: [] for s in SOLVERS} for seg in seg_values}
    rep_by_seg: dict[int, dict] = {}        # qualified picks (band >= threshold)
    fallback_by_seg: dict[int, dict] = {}   # first-seen sample per group
    print(
        f"Representative-sample rule: first sample per num_segments with "
        f"band coverage >= {args.min_rep_band_frac*100:.1f}%. "
        f"Pool restricted to the first {args.n_per_group} samples per group "
        f"(--n_per_group); raise it to widen the pool."
        if args.n_per_group is not None
        else f"Representative-sample rule: first sample per num_segments with "
             f"band coverage >= {args.min_rep_band_frac*100:.1f}%."
    )

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

    n_total = u0_all.shape[0]
    for idx in range(n_total):
        seg = int(seg_all[idx])
        ic_type = canonical_family(str(ictype_all[idx]))
        done = len(mae_full[seg]["HypNO-ST3"])
        if args.n_per_group is not None and done >= args.n_per_group:
            continue
        u0_np = u0_all[idx]; lh = u_gt[idx]
        mask = detect_shock_mask(
            lh, dx, args.jump_threshold, args.band_cells,
            tv_gate=tv_gate, tv_multiplier=args.tv_multiplier,
        )
        print(
            f"[{idx+1}/{n_total}] seg={seg} ic={ic_type} band={mask.mean()*100:.2f}%",
            flush=True,
        )
        _, _, weno = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="weno5"
        )
        _, _, godu = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="godunov"
        )
        hypno = run_hypno(u0_np)
        fno_pred, _ = run_fno(fno_model, u0_np, x_grid, t_grid, device)
        preds = {
            "HypNO-ST3": hypno,
            "WENO5": weno,
            "Godunov": godu,
            "FNO": fno_pred,
        }
        for name, pred in preds.items():
            mae_full[seg][name].append(mae(pred, lh))
            mae_shock[seg][name].append(masked_mae(pred, lh, mask))

        # Pick the first sample whose shock band covers >= min_rep_band_frac
        # of the (t, x) cells. Keep the first-seen sample as a fallback so a
        # group with no qualifying sample still gets a plot.
        record = {
            "idx": idx, "ic_type": ic_type, "seg": seg,
            "lh": lh, "preds": preds, "mask": mask,
        }
        if seg not in fallback_by_seg:
            fallback_by_seg[seg] = record
        if seg not in rep_by_seg and mask.mean() >= args.min_rep_band_frac:
            rep_by_seg[seg] = record

    # ---- Plots ----
    plot_mae_vs_segments(
        seg_values, mae_full, mae_shock,
        fig_dir / "shock_mae_vs_num_segments.png",
        args.band_cells, args.jump_threshold,
    )
    # Fill in any groups that never met the band threshold with the fallback,
    # so every num_segments still gets a plot. Log the substitution loudly.
    for seg in seg_values:
        if seg not in rep_by_seg and seg in fallback_by_seg:
            fb = fallback_by_seg[seg]
            print(
                f"[rep-pick] num_segments={seg}: no sample with band >= "
                f"{args.min_rep_band_frac*100:.1f}%; falling back to first-seen "
                f"sample idx={fb['idx']} (band {fb['mask'].mean()*100:.2f}%)."
            )
            rep_by_seg[seg] = fb
        elif seg in rep_by_seg:
            r = rep_by_seg[seg]
            print(
                f"[rep-pick] num_segments={seg}: chose sample idx={r['idx']} "
                f"(band {r['mask'].mean()*100:.2f}%)."
            )

    for seg, rep in rep_by_seg.items():
        plot_compare_full(rep, x_np, t_np, fig_dir / f"shock_compare_num_segments_{seg}.png", dx)
        plot_zoom(rep, x_np, t_np, fig_dir / f"shock_zoom_num_segments_{seg}.png")
        plot_slice(rep, x_np, t_np, fig_dir / f"shock_slice_num_segments_{seg}.png", dx)

    # ---- LaTeX ----
    table_tex = emit_table(seg_values, mae_full, mae_shock)
    section_tex = emit_section(
        table_tex, seg_values, args.jump_threshold, args.band_cells, args.n_per_group,
    )
    (out_dir / "shock_table.tex").write_text(table_tex + "\n", encoding="utf-8")
    (out_dir / "shock_results.tex").write_text(section_tex + "\n", encoding="utf-8")

    # ---- Machine-readable summary ----
    summary_lines = [
        "# shock_summary.txt",
        f"# jump_threshold={args.jump_threshold} band_cells={args.band_cells} "
        f"tv_gate={'on' if tv_gate else 'off'} tv_multiplier={args.tv_multiplier}",
        "# seg method mae_full_mean mae_full_std mae_shock_mean mae_shock_std n_samples",
    ]
    for seg in seg_values:
        for name in SOLVERS:
            mf = np.array(mae_full[seg][name], dtype=float)
            ms = np.array(mae_shock[seg][name], dtype=float)
            summary_lines.append(
                f"{seg} {name} {np.nanmean(mf):.6e} {np.nanstd(mf):.6e} "
                f"{np.nanmean(ms):.6e} {np.nanstd(ms):.6e} {len(mf)}"
            )
    (out_dir / "shock_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"\nWrote:")
    print(f"  {out_dir / 'shock_results.tex'}")
    print(f"  {out_dir / 'shock_table.tex'}")
    print(f"  {out_dir / 'shock_summary.txt'}")
    print(f"  {fig_dir}/  ({sum(1 for _ in fig_dir.iterdir())} PNGs)")


if __name__ == "__main__":
    main()
