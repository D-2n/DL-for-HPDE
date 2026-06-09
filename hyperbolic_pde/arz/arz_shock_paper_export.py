"""Paper-ready export of ARZ shock-localized comparison results.

The ARZ analogue of scripts/shock_paper_export.py. Re-runs HypNO-ARZ vs WENO5
vs Godunov on a stratified ARZ set, detects two discontinuity bands on the
ground truth (1-shock via the v / lambda1 field, contact via the rho field),
and writes (under <out_dir>, default <ckpt_dir>/arz_paper_shock):

    figs/
        shock_mae_vs_num_segments.png          (rho-channel band MAE)
        shock_compare_num_segments_{seg}.png   (one per seg)
        shock_zoom_num_segments_{seg}.png      (one per seg)
        shock_slice_num_segments_{seg}.png     (one per seg)
    shock_results.tex     -- standalone \\input-able section
    shock_table.tex       -- table snippet only
    shock_summary.txt     -- machine-readable per-method per-seg per-region MAE

The headline channel is rho (LWR parity: u -> rho). The shock table reports
MAE on rho in three regions per method: (full, 1shock, contact) -- the
ARZ-native two-band split is kept rather than collapsed into one column. The
detectors and baseline runner are imported from eval_shock_arz so any change
there propagates here.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.model_arz import load_hypno_arz_from_checkpoint
from hyperbolic_pde.arz.eval_shock_arz import (
    detect_1shock_band, detect_contact_band, _mae, _run_baseline,
)

# rho is the headline channel for all figures and the shock table.
HEADLINE_CHANNEL = "rho"
COLORS = {"model": "tab:blue", "weno5": "tab:orange", "godunov": "tab:green"}
TEX_NAME = {"model": r"HypNO-ARZ", "weno5": r"WENO5", "godunov": r"Godunov"}
# Band outline colours (compare plots): 1-shock pink, contact cyan dashed.
BAND_1SHOCK_COLOR = "#ff1493"
BAND_CONTACT_COLOR = "#00ced1"
REGIONS = ["full", "1shock", "contact"]


def configure_paper_style(plt) -> None:
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


def fmt_mae(x: float) -> str:
    if not np.isfinite(x):
        return r"\text{n/a}"
    return f"\\num{{{x:.3e}}}"


def _locate_config(ckpt_path: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    for cand in (ckpt_path.parent / "config.yaml", ckpt_path.parent.parent / "config.yaml"):
        if cand.exists():
            return cand
    return None


# --------------------------------------------------------------------------- #
# Figure helpers (rho channel; two band outlines)
# --------------------------------------------------------------------------- #
def draw_band_outline(ax, mask: np.ndarray, x_np, t_np, color, dashed=False) -> None:
    """Padded contour outline of a boolean band mask."""
    m = np.pad(mask.astype(float), 1, mode="constant", constant_values=0.0)
    x_pad = np.concatenate([
        [x_np[0] - (x_np[1] - x_np[0])], x_np, [x_np[-1] + (x_np[-1] - x_np[-2])],
    ])
    t_pad = np.concatenate([
        [t_np[0] - (t_np[1] - t_np[0])], t_np, [t_np[-1] + (t_np[-1] - t_np[-2])],
    ])
    ax.contour(x_pad, t_pad, m, levels=[0.5], colors=color, linewidths=1.2,
               linestyles="dashed" if dashed else "solid")


def plot_mae_vs_segments(seg_values, mae_by_region, methods, fig_path, band_half, tau_shock, plt):
    """Three panels (full, 1shock, contact): rho-band MAE vs num_segments."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for ax, region in zip(axes, REGIONS):
        any_curve = False
        for m in methods:
            means = [np.nanmean(mae_by_region[region][seg][m]) for seg in seg_values]
            stds = [np.nanstd(mae_by_region[region][seg][m]) for seg in seg_values]
            if all(not np.isfinite(v) for v in means):
                continue
            ax.errorbar(seg_values, means, yerr=stds, marker="o", capsize=3,
                        label=m, color=COLORS.get(m), linewidth=1.4)
            any_curve = True
        ax.set_xlabel("num.\\ IC segments")
        ax.set_ylabel("rho MAE")
        ax.set_yscale("log")
        ax.set_xticks(seg_values)
        ax.set_title(f"{region} (band $\\pm${band_half}, $\\tau={tau_shock}$)")
        if any_curve:
            ax.legend()
        ax.grid(True, alpha=0.3)
    fig.savefig(fig_path)
    plt.close(fig)


def plot_compare_full(rep, x_np, t_np, fig_path, methods, plt):
    """Per-seg full-domain rho comparison with both band outlines."""
    gt = rep["rho_gt"]; preds = rep["preds"]
    m1 = rep["mask_1shock"]; mc = rep["mask_contact"]
    present = [m for m in methods if m in preds]
    vmin, vmax = float(gt.min()), float(gt.max())
    ncols = 1 + len(present)
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 6.4), constrained_layout=True)

    im = axes[0, 0].pcolormesh(x_np, t_np, gt, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    draw_band_outline(axes[0, 0], m1, x_np, t_np, BAND_1SHOCK_COLOR)
    draw_band_outline(axes[0, 0], mc, x_np, t_np, BAND_CONTACT_COLOR, dashed=True)
    axes[0, 0].set_title("GT rho (pink=1-shock, cyan=contact)")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("t")
    fig.colorbar(im, ax=axes[0, 0], label="rho")
    axes[1, 0].axis("off")

    full_err_vmax = max(np.abs(preds[m]["rho"] - gt).max() for m in present)
    for c, m in enumerate(present, start=1):
        sol = preds[m]["rho"]; err = np.abs(sol - gt)
        im = axes[0, c].pcolormesh(x_np, t_np, sol, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        draw_band_outline(axes[0, c], m1, x_np, t_np, BAND_1SHOCK_COLOR)
        draw_band_outline(axes[0, c], mc, x_np, t_np, BAND_CONTACT_COLOR, dashed=True)
        axes[0, c].set_title(TEX_NAME.get(m, m))
        axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
        fig.colorbar(im, ax=axes[0, c], label="rho")

        im = axes[1, c].pcolormesh(x_np, t_np, err, shading="auto", cmap="magma", vmin=0, vmax=full_err_vmax)
        axes[1, c].set_title(f"|{TEX_NAME.get(m, m)} $-$ GT|")
        axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
        fig.colorbar(im, ax=axes[1, c])

    sh = "  ".join(
        f"{m}: 1sh={_mae(preds[m]['rho'], gt, m1):.2e} con={_mae(preds[m]['rho'], gt, mc):.2e}"
        for m in present
    )
    fig.suptitle(f"Sample {rep['idx']} -- {rep['fam']}, num_segments={rep['seg']}\nrho MAE: {sh}")
    fig.savefig(fig_path)
    plt.close(fig)


def plot_zoom(rep, x_np, t_np, fig_path, methods, plt):
    """Band-only zoom (crop to combined-band bbox; non-band cells blanked)."""
    gt = rep["rho_gt"]; preds = rep["preds"]
    mask = rep["mask_combined"]
    present = [m for m in methods if m in preds]
    if not mask.any():
        return
    t_any = mask.any(axis=1); x_any = mask.any(axis=0)
    t_lo, t_hi = int(np.argmax(t_any)), int(len(t_any) - np.argmax(t_any[::-1]))
    x_lo, x_hi = int(np.argmax(x_any)), int(len(x_any) - np.argmax(x_any[::-1]))
    t_lo = max(t_lo - 1, 0); t_hi = min(t_hi + 1, gt.shape[0])
    x_lo = max(x_lo - 1, 0); x_hi = min(x_hi + 1, gt.shape[1])
    x_z = x_np[x_lo:x_hi]; t_z = t_np[t_lo:t_hi]
    mask_z = mask[t_lo:t_hi, x_lo:x_hi]

    def blanked(arr2d):
        out = arr2d.astype(float).copy(); out[~mask_z] = np.nan; return out

    u_in = gt[mask]; vmin = float(u_in.min()); vmax = float(u_in.max())
    err_max = max(float(np.abs(preds[m]["rho"] - gt)[mask].max()) for m in present)
    ncols = 1 + len(present)
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 6.0), constrained_layout=True)

    im = axes[0, 0].pcolormesh(x_z, t_z, blanked(gt[t_lo:t_hi, x_lo:x_hi]),
                                shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("GT rho"); axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("t")
    fig.colorbar(im, ax=axes[0, 0], label="rho")
    axes[1, 0].axis("off")
    for c, m in enumerate(present, start=1):
        sol_z = preds[m]["rho"][t_lo:t_hi, x_lo:x_hi]
        err_z = np.abs(preds[m]["rho"] - gt)[t_lo:t_hi, x_lo:x_hi]
        im = axes[0, c].pcolormesh(x_z, t_z, blanked(sol_z), shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        axes[0, c].set_title(TEX_NAME.get(m, m)); axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
        fig.colorbar(im, ax=axes[0, c], label="rho")
        im = axes[1, c].pcolormesh(x_z, t_z, blanked(err_z), shading="auto", cmap="magma", vmin=0, vmax=err_max)
        axes[1, c].set_title(f"|{TEX_NAME.get(m, m)} $-$ GT|"); axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
        fig.colorbar(im, ax=axes[1, c])

    fig.suptitle(
        f"Sample {rep['idx']} -- {rep['fam']}, num_segments={rep['seg']} (combined band only)\n"
        f"band {mask.mean()*100:.2f}% of cells"
    )
    fig.savefig(fig_path)
    plt.close(fig)


def plot_slice(rep, x_np, t_np, fig_path, methods, plt):
    gt = rep["rho_gt"]; preds = rep["preds"]
    mask = rep["mask_combined"]
    present = [m for m in methods if m in preds]
    if not mask.any():
        return
    T = float(t_np[-1])
    t_lo_s, t_hi_s = 0.15 * T, 0.8 * T
    eligible = (t_np >= t_lo_s) & (t_np <= t_hi_s) & mask.any(axis=1)
    if not eligible.any():
        eligible = (np.arange(len(t_np)) > 0) & mask.any(axis=1)
        if not eligible.any():
            return
    k_star = int(np.random.choice(np.flatnonzero(eligible)))
    row_mask = mask[k_star]
    i_lo = max(int(np.argmax(row_mask)) - 3, 0)
    i_hi = min(int(len(row_mask) - np.argmax(row_mask[::-1])) + 3, gt.shape[1])
    xs = x_np[i_lo:i_hi]
    dx = float(x_np[1] - x_np[0])
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    ax.plot(xs, gt[k_star, i_lo:i_hi], color="black", lw=2.0, label="GT")
    for m in present:
        ax.plot(xs, preds[m]["rho"][k_star, i_lo:i_hi], color=COLORS.get(m), lw=1.4, label=TEX_NAME.get(m, m))
    band_xs = xs[row_mask[i_lo:i_hi]]
    if band_xs.size:
        ax.axvspan(band_xs[0] - 0.5 * dx, band_xs[-1] + 0.5 * dx, color="orange", alpha=0.15, label="shock band")
    ax.set_xlabel("x"); ax.set_ylabel("rho")
    ax.set_title(f"rho slice at t={t_np[k_star]:.3f}, num_segments={rep['seg']}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(fig_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# LaTeX emission
# --------------------------------------------------------------------------- #
def emit_table(seg_values, mae_by_region, methods) -> str:
    """Rows = num_segments, columns = (full, 1shock, contact) MAE per method (rho)."""
    n_methods = len(methods)
    ncols_per = len(REGIONS)
    col_spec = "c" + "c" * (ncols_per * n_methods)
    headers_top = " & ".join(
        ["num.\\ seg."] +
        [f"\\multicolumn{{{ncols_per}}}{{c}}{{{TEX_NAME.get(m, m)}}}" for m in methods]
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + ncols_per*k}-{1 + ncols_per*(k+1)}}}" for k in range(n_methods)
    )
    headers_bot = " & ".join([""] + REGIONS * n_methods)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{ARZ accuracy on shock neighborhoods (rho channel). \emph{full} "
        r"is MAE over the whole space-time domain; \emph{1shock} and "
        r"\emph{contact} restrict to the genuinely-nonlinear $1$-shock band "
        r"(detected on the $\lambda_1$/$v$ field via the Lax condition "
        r"$\lambda_{1,L}>\lambda_{1,R}$) and the $2$-contact band (detected on "
        r"$\rho$ with $v$ continuity), both dilated to a fixed-width "
        r"neighborhood. The same masks are reused across methods.}",
        r"\label{tab:arz-shock-mae}",
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
        for m in methods:
            for region in REGIONS:
                cells.append(fmt_mae(np.nanmean(mae_by_region[region][seg][m])))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def emit_section(table_tex, seg_values, tau_shock, band_half, p_form) -> str:
    def fig_block(path, caption, label, width=r"\linewidth"):
        return "\n".join([
            r"\begin{figure}[t]", r"  \centering",
            f"  \\includegraphics[width={width}]{{figs/{path}}}",
            f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
            r"\end{figure}",
        ])

    p_tex = r"p(\rho)=\rho" if p_form == "rho" else r"p(\rho)=\rho+\rho^2"
    out = []
    out.append("% =========================================================================")
    out.append("% HypNO-ARZ accuracy on shock neighborhoods.")
    out.append("% Standalone section file -- \\input{} into the main document.")
    out.append("% Requires: amsmath, booktabs, graphicx, siunitx.")
    out.append("% =========================================================================\n")
    out.append(r"\section{Accuracy on shock neighborhoods (ARZ)}")
    out.append(r"\label{sec:arz-shock-comparison}")
    out.append("")
    out.append(
        r"To isolate accuracy on the discontinuous portion of the ARZ solution, "
        r"we evaluate HypNO-ARZ, WENO5, and Godunov on a stratified set and "
        r"report MAE on $\rho$ restricted to two discontinuity bands detected "
        r"on the ground truth. Because ARZ carries two wave families, we detect "
        r"them separately: a \emph{1-shock} band on the genuinely-nonlinear "
        rf"field (the eigenvalue $\lambda_1=v-\rho\,p'(\rho)$ with ${p_tex}$, "
        r"flagged where $\lambda_1$ and $v$ drop across an interface, the Lax-1 "
        r"condition $\lambda_{1,L}>\lambda_{1,R}$), and a \emph{2-contact} band "
        r"on $\rho$ where the density jumps but $v$ stays continuous. A "
        r"total-variation gate suppresses smooth-region blips, and the "
        rf"surviving cells are dilated by $\pm{band_half}$ cells "
        rf"(threshold $\tau={tau_shock}$). The same masks are reused across "
        r"methods."
    )
    out.append("")
    out.append(table_tex)
    out.append("")
    out.append(fig_block(
        "shock_mae_vs_num_segments.png",
        r"ARZ $\rho$ MAE in the full domain, the 1-shock band, and the contact "
        r"band, as a function of the number of IC discontinuities.",
        "fig:arz-shock-mae-vs-seg",
    ))
    for seg in seg_values:
        out.append(fig_block(
            f"shock_compare_num_segments_{seg}.png",
            rf"Representative ARZ sample at num\_segments={seg}. Top row: $\rho$ "
            r"fields with the 1-shock band (pink) and contact band (cyan dashed) "
            r"outlined; bottom row: absolute error to the ground truth.",
            f"fig:arz-shock-compare-{seg}",
        ))
        out.append(fig_block(
            f"shock_zoom_num_segments_{seg}.png",
            rf"Same sample as Fig.~\ref{{fig:arz-shock-compare-{seg}}}, zoomed to "
            r"the combined-band bounding box with non-band cells blanked.",
            f"fig:arz-shock-zoom-{seg}",
        ))
        out.append(fig_block(
            f"shock_slice_num_segments_{seg}.png",
            rf"$\rho(x)$ slice through a band row for num\_segments={seg}. "
            r"The shaded region marks the combined shock band.",
            f"fig:arz-shock-slice-{seg}",
        ))
    out.append("")
    out.append(r"% End of shock_results.tex")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Emit paper-ready ARZ shock-comparison plots and a .tex section."
    )
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--baselines", type=str, default="godunov,weno5")
    ap.add_argument("--n_per_group", type=int, default=None)
    ap.add_argument("--tau-shock", type=float, default=0.06)
    ap.add_argument("--band-halfwidth", type=int, default=2)
    ap.add_argument("--tv-mult", type=float, default=1.5)
    ap.add_argument("--tau", type=float, default=None,
                    help="Override the relaxation tau for baselines (defaults to bundle.tau).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rep-band-frac", type=float, default=0.05)
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Output directory (default: <ckpt_dir>/arz_paper_shock).")
    args = ap.parse_args()

    # Full determinism (the slice picks a random eligible band row).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Determinism: seed={args.seed}, cudnn.deterministic=True, benchmark=False")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_paper_style(plt)

    bundle = load_arz_dataset(args.data)
    tau = float(args.tau) if args.tau is not None else float(bundle.tau)
    N = bundle.rho.shape[0]

    from hyperbolic_pde.arz.physics_arz import set_pressure_form, get_pressure_form
    set_pressure_form(bundle.p_form)
    print(f"[arz_shock_paper_export] pressure_form={get_pressure_form()}")

    config_path = _locate_config(args.ckpt, args.config)
    model, _ck_tau = load_hypno_arz_from_checkpoint(
        args.ckpt, device=args.device, config_path=config_path,
    )
    print(f"[arz_shock_paper_export] loaded {args.ckpt}  tau_eval={tau}")

    out_dir = args.out_dir or (args.ckpt.parent / "arz_paper_shock")
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x_np = bundle.x.astype(np.float64)
    t_np = bundle.t.astype(np.float64)
    x_t = torch.tensor(bundle.x, dtype=torch.float32, device=args.device)
    t_t = torch.tensor(bundle.t, dtype=torch.float32, device=args.device)

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    methods = ["model"] + baselines
    baseline_tau = tau if np.isfinite(tau) else 1e6

    seg_values = sorted(set(int(s) for s in bundle.num_segments))
    # mae_by_region[region][seg][method] -> list of per-sample rho MAE.
    mae_by_region = {
        r: {seg: {m: [] for m in methods} for seg in seg_values} for r in REGIONS
    }
    rep_by_seg: dict = {}
    fallback_by_seg: dict = {}

    print(
        f"Representative rule: first sample per num_segments with combined-band "
        f"coverage >= {args.min_rep_band_frac*100:.1f}%."
    )

    for i in range(N):
        fam = str(bundle.ic_type[i])
        seg = int(bundle.num_segments[i])
        done = len(mae_by_region["full"][seg]["model"])
        if args.n_per_group is not None and done >= args.n_per_group:
            continue

        rho_gt = bundle.rho[i].astype(np.float64)
        w_gt = bundle.w[i].astype(np.float64)
        v_gt = w_gt - P.pressure(rho_gt)
        rho0 = bundle.rho0[i].astype(np.float64)
        w0 = bundle.w0[i].astype(np.float64)

        mask_1shock = detect_1shock_band(rho_gt, w_gt, v_gt, args.tau_shock, args.band_halfwidth, args.tv_mult)
        mask_contact = detect_contact_band(rho_gt, w_gt, v_gt, args.tau_shock, args.band_halfwidth, args.tv_mult)
        mask_combined = mask_1shock | mask_contact
        region_masks = {"full": None, "1shock": mask_1shock, "contact": mask_contact}

        print(
            f"[{i+1}/{N}] seg={seg} fam={fam} "
            f"1sh={mask_1shock.mean()*100:.2f}% con={mask_contact.mean()*100:.2f}%",
            flush=True,
        )

        with torch.no_grad():
            rho_p, w_p, _ = model(
                torch.tensor(rho0[None], dtype=torch.float32, device=args.device),
                torch.tensor(w0[None], dtype=torch.float32, device=args.device),
                x_t, t_t,
            )
        rho_p = rho_p[0].cpu().numpy(); w_p = w_p[0].cpu().numpy()
        preds = {"model": {"rho": rho_p, "w": w_p, "v": w_p - P.pressure(rho_p)}}
        for m in baselines:
            try:
                rho_b, w_b = _run_baseline(m, rho0, w0, bundle, baseline_tau, args.boundary)
            except Exception as e:
                print(f"  [warn] {m} failed on sample {i}: {e}", flush=True)
                continue
            preds[m] = {"rho": rho_b, "w": w_b, "v": w_b - P.pressure(rho_b)}

        for region, mask in region_masks.items():
            for m in preds:
                mae_by_region[region][seg][m].append(_mae(preds[m]["rho"], rho_gt, mask))

        record = {
            "idx": i, "fam": fam, "seg": seg, "rho_gt": rho_gt, "preds": preds,
            "mask_1shock": mask_1shock, "mask_contact": mask_contact,
            "mask_combined": mask_combined,
        }
        if seg not in fallback_by_seg:
            fallback_by_seg[seg] = record
        if seg not in rep_by_seg and mask_combined.mean() >= args.min_rep_band_frac:
            rep_by_seg[seg] = record

    # ---- Plots ----
    plot_mae_vs_segments(
        seg_values, mae_by_region, methods,
        fig_dir / "shock_mae_vs_num_segments.png",
        args.band_halfwidth, args.tau_shock, plt,
    )
    for seg in seg_values:
        if seg not in rep_by_seg and seg in fallback_by_seg:
            fb = fallback_by_seg[seg]
            print(
                f"[rep-pick] num_segments={seg}: no sample with band >= "
                f"{args.min_rep_band_frac*100:.1f}%; falling back to idx={fb['idx']} "
                f"(band {fb['mask_combined'].mean()*100:.2f}%)."
            )
            rep_by_seg[seg] = fb
        elif seg in rep_by_seg:
            r = rep_by_seg[seg]
            print(f"[rep-pick] num_segments={seg}: chose idx={r['idx']} "
                  f"(band {r['mask_combined'].mean()*100:.2f}%).")

    for seg, rep in rep_by_seg.items():
        plot_compare_full(rep, x_np, t_np, fig_dir / f"shock_compare_num_segments_{seg}.png", methods, plt)
        plot_zoom(rep, x_np, t_np, fig_dir / f"shock_zoom_num_segments_{seg}.png", methods, plt)
        plot_slice(rep, x_np, t_np, fig_dir / f"shock_slice_num_segments_{seg}.png", methods, plt)

    # ---- LaTeX ----
    table_tex = emit_table(seg_values, mae_by_region, methods)
    section_tex = emit_section(table_tex, seg_values, args.tau_shock, args.band_halfwidth, bundle.p_form)
    (out_dir / "shock_table.tex").write_text(table_tex + "\n", encoding="utf-8")
    (out_dir / "shock_results.tex").write_text(section_tex + "\n", encoding="utf-8")

    # ---- Machine-readable summary ----
    lines = [
        "# arz_shock_summary.txt  (channel=rho)",
        f"# tau_shock={args.tau_shock} band_halfwidth={args.band_halfwidth} tv_mult={args.tv_mult}",
        "# seg method region mae_mean mae_std n_samples",
    ]
    for seg in seg_values:
        for m in methods:
            for region in REGIONS:
                arr = np.array(mae_by_region[region][seg][m], dtype=float)
                lines.append(
                    f"{seg} {m} {region} {np.nanmean(arr):.6e} {np.nanstd(arr):.6e} {len(arr)}"
                )
    (out_dir / "shock_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote:")
    print(f"  {out_dir / 'shock_results.tex'}")
    print(f"  {out_dir / 'shock_table.tex'}")
    print(f"  {out_dir / 'shock_summary.txt'}")
    print(f"  {fig_dir}/  ({sum(1 for _ in fig_dir.iterdir())} PNGs)")


if __name__ == "__main__":
    main()
