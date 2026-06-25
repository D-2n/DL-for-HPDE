"""ARZ mark-0 (orig) paper evaluation: HypNO-ARZ-orig vs FNO vs WENO5 vs
Godunov, stratified by (ic_type, num_segments) -- the mark-0 analogue of
scripts/paper_eval.py (and arz_paper_eval.py, which targets the mark-1 model).

Identical workflow to the LWR paper-eval so ARZ-mark0 and LWR results are
directly comparable. Per (ic_type, num_segments) "cell" it computes per-channel
MAE (rho, w, v) vs the dataset's stored ground truth, classifies cells as ID/OOD
against the training config, and emits:

  * one per-cell qualitative figure per channel (rho, w): a rho0 strip on top,
    a row of space-time heatmaps (GT | model | FNO | baselines), and a row of
    |solver - GT| heatmaps.
  * mae_vs_num_segments_<ic_type>_<channel>.png with mean +/- std error bars.
  * summary.csv (long format): family, num_segments, group, method, channel,
    mae_mean, mae_std, n_samples.
  * summary.tex: rho headline tables (per-family + ID/OOD/all) + appendix
    (per-num_segments rho + per-channel pooled).
  * summary.txt: plain-text scan.

rho is the headline channel (main tables/figures); rho/w/v all appear in the
CSV and the per-channel appendix. The only differences vs arz_paper_eval.py are
(a) the model loader (HypNO_ARZ_Orig / load_hypno_arz_orig_from_checkpoint) and
(b) the optional FNO baseline column.

Usage:
    python -m hyperbolic_pde.arz.arz_mark0_paper_eval \\
        --ckpt /path/to/checkpoint_epoch170.pt \\
        --data /path/to/arz_stratified_eval.npz \\
        --fno-weights /path/to/fno_arz.pt --fno-config config.yaml \\
        --out_dir /path/to/results/arz_mark0_paper_eval
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.model_arz_orig import load_hypno_arz_orig_from_checkpoint
from hyperbolic_pde.arz.eval_vs_numerical_arz import _run_baseline
from hyperbolic_pde.models.competitive_architectures.fno import FNO2d

# Headline channel for the main-text tables/figures (LWR parity: u -> rho).
HEADLINE_CHANNEL = "rho"
CHANNELS = ["rho", "w", "v"]
COLORS = {
    "model": "tab:blue",
    "FNO": "tab:purple",
    "weno5": "tab:orange",
    "godunov": "tab:green",
}
# Maximum number of segment values to spell out in a figure title.
_MAX_SEG_VALUES_IN_TITLE = 8


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred - truth).mean())


def _locate_config(ckpt_path: Path, explicit: Path | None) -> Path | None:
    """Mirror load_hypno_arz_orig_from_checkpoint's config auto-location."""
    if explicit is not None:
        return explicit
    for cand in (
        ckpt_path.parent / "config.yaml",
        ckpt_path.parent.parent / "config.yaml",
    ):
        if cand.exists():
            return cand
    return None


def _load_train_id_sets(
    config_path: Path | None, train_section: str
) -> tuple[set[str], set[int]]:
    """Read the training ic_types / num_segments for ID/OOD classification.

    Returns (train_ic_types, train_segs). Empty sets if the config or section
    is unavailable -- in that case every cell reads as OOD.
    """
    if config_path is None or not Path(config_path).exists():
        print(f"[arz_mark0_paper_eval] no config for ID/OOD; all cells -> OOD")
        return set(), set()
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    sec = cfg.get(train_section)
    if not sec:
        print(
            f"[arz_mark0_paper_eval] config has no '{train_section}:' section; "
            f"all cells -> OOD"
        )
        return set(), set()
    ic_types = set(str(s) for s in sec.get("ic_types", []))
    segs = set(int(s) for s in sec.get("num_segments", []))
    print(f"[arz_mark0_paper_eval] train ic_types={sorted(ic_types)}  segs={sorted(segs)}")
    return ic_types, segs


def extract_segment_values(field0: np.ndarray, atol: float = 1e-6) -> list[float]:
    """Distinct adjacent segment values in a 1D initial profile."""
    values: list[float] = [float(field0[0])]
    for val in field0[1:]:
        if abs(float(val) - values[-1]) > atol:
            values.append(float(val))
    return values


def format_ic_label(rho0: np.ndarray, ic_type: str, num_segments: int) -> str:
    if ic_type in ("piecewise_constant_stratified", "riemann_stratified"):
        vals = extract_segment_values(rho0)
        if len(vals) <= _MAX_SEG_VALUES_IN_TITLE:
            val_str = "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"
        else:
            head = ", ".join(f"{v:.2f}" for v in vals[:3])
            tail = ", ".join(f"{v:.2f}" for v in vals[-2:])
            val_str = f"[{head}, ..., {tail}] (n_distinct={len(vals)})"
        return f"{ic_type}, num_segments={num_segments}, rho0={val_str}"
    return f"{ic_type}, num_segments={num_segments}"


# --------------------------------------------------------------------------- #
# FNO baseline (optional)
# --------------------------------------------------------------------------- #

def _make_fno_input(rho0_np, w0_np, x_np, t_np) -> torch.Tensor:
    """Build the (1, 4, nx, nt) input tensor the ARZ FNO expects."""
    x_t = torch.tensor(x_np, dtype=torch.float32)
    t_t = torch.tensor(t_np, dtype=torch.float32)
    X, T = torch.meshgrid(x_t, t_t, indexing="ij")
    nt = t_t.numel()
    rho0_g = torch.tensor(rho0_np, dtype=torch.float32).unsqueeze(1).repeat(1, nt)
    w0_g = torch.tensor(w0_np, dtype=torch.float32).unsqueeze(1).repeat(1, nt)
    return torch.stack([X, T, rho0_g, w0_g], dim=0).unsqueeze(0)  # (1, 4, nx, nt)


def _load_fno(weights: Path, cfg_section: dict, device: str) -> FNO2d:
    model = FNO2d(
        in_channels=4, out_channels=2,
        width=int(cfg_section["width"]),
        modes_x=int(cfg_section["modes_x"]),
        modes_t=int(cfg_section["modes_t"]),
        layers=int(cfg_section["layers"]),
    ).to(device)
    # Accept either a bare state_dict or a wrapped training checkpoint
    # ({"model"|"state_dict"|"model_state_dict": ...}, possibly with optimizer
    # state -> needs weights_only=False).
    state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ARZ mark-0 (orig) paper evaluation: HypNO-ARZ-orig vs FNO "
                    "vs WENO5 vs Godunov, stratified by (ic_type, num_segments)."
    )
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="HypNO_ARZ_Orig checkpoint (legacy dict OR bare state_dict).")
    ap.add_argument("--config", type=Path, default=None,
                    help="config.yaml for the architecture AND for ID/OOD "
                         "training sets. Auto-located next to the ckpt if omitted.")
    ap.add_argument("--model-section", type=str, default="hypno_arz_orig",
                    help="Config section for architecture kwargs (default hypno_arz_orig).")
    ap.add_argument("--data", type=Path, required=True,
                    help="Stratified eval dataset .npz (must carry ic_type, "
                         "num_segments).")
    ap.add_argument("--baselines", type=str, default="godunov,weno5")
    ap.add_argument("--n_per_cell", type=int, default=None,
                    help="Cap on samples per (ic_type, num_segments) cell.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--tau", type=float, default=None,
                    help="Override the relaxation tau for baselines "
                         "(defaults to bundle.tau).")
    ap.add_argument("--train-section", type=str, default="arz_data",
                    help="Config section whose ic_types/num_segments define "
                         "the ID set (default arz_data).")
    ap.add_argument("--fno-weights", type=Path, default=None,
                    help="FNO checkpoint (.pt). FNO column omitted if not given.")
    ap.add_argument("--fno-config", type=Path, default=None,
                    help="config.yaml containing the fno_arz section. "
                         "Defaults to --config if omitted.")
    ap.add_argument("--fno-section", type=str, default="fno_arz",
                    help="Config section for FNO architecture (default fno_arz).")
    ap.add_argument("--no-fno", action="store_true",
                    help="Skip the FNO baseline even if weights are provided.")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Output directory (default: <ckpt_dir>/arz_mark0_paper_eval).")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle = load_arz_dataset(args.data)
    tau = float(args.tau) if args.tau is not None else float(bundle.tau)
    N = bundle.rho.shape[0]

    # Pressure closure must match the dataset before any physics runs.
    from hyperbolic_pde.arz.physics_arz import set_pressure_form, get_pressure_form
    set_pressure_form(bundle.p_form)
    print(f"[arz_mark0_paper_eval] pressure_form={get_pressure_form()}")

    config_path = _locate_config(args.ckpt, args.config)
    model, _ck_tau = load_hypno_arz_orig_from_checkpoint(
        args.ckpt, device=args.device, config_path=config_path,
        model_section=args.model_section,
    )
    print(f"[arz_mark0_paper_eval] loaded {args.ckpt}  tau_eval={tau}")

    # ---- Optional FNO baseline --------------------------------------- #
    fno_model = None
    if args.no_fno:
        print("[arz_mark0_paper_eval] --no-fno given; skipping FNO baseline.")
    elif args.fno_weights is not None:
        if not args.fno_weights.exists():
            print(f"[arz_mark0_paper_eval] FNO weights not found at "
                  f"{args.fno_weights}; skipping FNO column.")
        else:
            fno_cfg_path = args.fno_config or config_path
            if fno_cfg_path is None or not Path(fno_cfg_path).exists():
                ap.error("--fno-weights requires --fno-config (or a locatable "
                         "--config) to find the fno section.")
            with open(fno_cfg_path, encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f) or {}
            fno_sec = full_cfg.get(args.fno_section)
            if fno_sec is None:
                ap.error(f"Section '{args.fno_section}' not found in {fno_cfg_path}.")
            fno_model = _load_fno(args.fno_weights, fno_sec, args.device)
            print(f"[arz_mark0_paper_eval] loaded FNO: {args.fno_weights}")

    train_ic_types, train_segs = _load_train_id_sets(config_path, args.train_section)

    def cell_is_id(fam: str, seg: int) -> bool:
        return (fam in train_ic_types) and (seg in train_segs)

    out_dir = args.out_dir or (args.ckpt.parent / "arz_mark0_paper_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    x_np = bundle.x.astype(np.float64)
    t_np = bundle.t.astype(np.float64)

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    # Method order: model, FNO (if present), then numerical baselines.
    methods = ["model"] + (["FNO"] if fno_model is not None else []) + baselines

    ic_types = sorted(set(str(s) for s in bundle.ic_type))
    seg_values = sorted(set(int(s) for s in bundle.num_segments))
    print(f"[arz_mark0_paper_eval] ic_types={ic_types}  num_segments={seg_values}")

    id_cells = [(ic, s) for ic in ic_types for s in seg_values if cell_is_id(ic, s)]
    ood_cells = [(ic, s) for ic in ic_types for s in seg_values if not cell_is_id(ic, s)]
    print(f"[arz_mark0_paper_eval] ID cells ({len(id_cells)}): {id_cells}")
    print(f"[arz_mark0_paper_eval] OOD cells ({len(ood_cells)}): {ood_cells}")

    # mae_by_cell[(fam, seg)][method][channel] -> list of per-sample MAE.
    mae_by_cell: dict = {
        (ic, s): {m: {ch: [] for ch in CHANNELS} for m in methods}
        for ic in ic_types for s in seg_values
    }
    # Representative sample per cell for the qualitative figure.
    rep_by_cell: dict = {}

    x_t = torch.tensor(bundle.x, dtype=torch.float32, device=args.device)
    t_t = torch.tensor(bundle.t, dtype=torch.float32, device=args.device)
    baseline_tau = tau if np.isfinite(tau) else 1e6

    for i in range(N):
        fam = str(bundle.ic_type[i])
        seg = int(bundle.num_segments[i])
        cell = (fam, seg)
        if cell not in mae_by_cell:
            continue
        done = len(mae_by_cell[cell]["model"]["rho"])
        if args.n_per_cell is not None and done >= args.n_per_cell:
            continue

        rho_gt = bundle.rho[i].astype(np.float64)
        w_gt = bundle.w[i].astype(np.float64)
        v_gt = w_gt - P.pressure(rho_gt)
        rho0 = bundle.rho0[i].astype(np.float64)
        w0 = bundle.w0[i].astype(np.float64)
        gt = {"rho": rho_gt, "w": w_gt, "v": v_gt}

        print(f"[{i + 1}/{N}] fam={fam} seg={seg} ...", flush=True)

        # Model (mark-0 / orig).
        with torch.no_grad():
            rho_p, w_p, _ = model(
                torch.tensor(rho0[None], dtype=torch.float32, device=args.device),
                torch.tensor(w0[None], dtype=torch.float32, device=args.device),
                x_t, t_t,
            )
        rho_p = rho_p[0].cpu().numpy(); w_p = w_p[0].cpu().numpy()
        preds = {"model": {"rho": rho_p, "w": w_p, "v": w_p - P.pressure(rho_p)}}

        # FNO.
        if fno_model is not None:
            inp = _make_fno_input(rho0, w0, x_np, t_np).to(args.device)
            with torch.no_grad():
                out = fno_model(inp)  # (1, 2, nx, nt)
            rho_f = out[0, 0].cpu().numpy().T  # (nt, nx)
            w_f = out[0, 1].cpu().numpy().T
            preds["FNO"] = {"rho": rho_f, "w": w_f, "v": w_f - P.pressure(rho_f)}

        # Numerical baselines.
        for m in baselines:
            try:
                rho_b, w_b = _run_baseline(m, rho0, w0, bundle, baseline_tau, args.boundary)
            except Exception as e:
                print(f"  [warn] {m} failed on sample {i}: {e}", flush=True)
                continue
            preds[m] = {"rho": rho_b, "w": w_b, "v": w_b - P.pressure(rho_b)}

        for m, pdict in preds.items():
            for ch in CHANNELS:
                mae_by_cell[cell][m][ch].append(mae(pdict[ch], gt[ch]))

        if cell not in rep_by_cell:
            rep_by_cell[cell] = {"idx": i, "rho0": rho0, "gt": gt, "preds": preds}

    # ----------------------------------------------------------------- #
    # Per-cell qualitative figures (one per channel in {rho, w}).
    # ----------------------------------------------------------------- #
    _render_cell_figures(rep_by_cell, methods, x_np, t_np, figs_dir, plt)

    # ----------------------------------------------------------------- #
    # MAE vs num_segments, per (ic_type, channel in {rho, w}).
    # ----------------------------------------------------------------- #
    _render_mae_vs_segments(
        mae_by_cell, ic_types, seg_values, methods, out_dir, plt
    )

    # ----------------------------------------------------------------- #
    # Pooled aggregates.
    # ----------------------------------------------------------------- #
    pooled_all = {m: {ch: [] for ch in CHANNELS} for m in methods}
    pooled_by_seg = {
        s: {m: {ch: [] for ch in CHANNELS} for m in methods} for s in seg_values
    }
    pooled_id = {m: {ch: [] for ch in CHANNELS} for m in methods}
    pooled_ood = {m: {ch: [] for ch in CHANNELS} for m in methods}
    for (fam, seg), mdict in mae_by_cell.items():
        is_id = cell_is_id(fam, seg)
        for m in methods:
            for ch in CHANNELS:
                vals = mdict[m][ch]
                pooled_all[m][ch].extend(vals)
                pooled_by_seg[seg][m][ch].extend(vals)
                (pooled_id if is_id else pooled_ood)[m][ch].extend(vals)

    _write_csv(out_dir, mae_by_cell, ic_types, seg_values, methods,
               cell_is_id, pooled_by_seg, pooled_id, pooled_ood, pooled_all)
    _write_tex(out_dir, mae_by_cell, ic_types, seg_values, methods,
               pooled_id, pooled_ood, pooled_all, pooled_by_seg)
    _write_txt(out_dir, mae_by_cell, ic_types, seg_values, methods,
               cell_is_id, id_cells, ood_cells,
               pooled_by_seg, pooled_id, pooled_ood, pooled_all)

    print(f"\n[arz_mark0_paper_eval] all outputs in {out_dir}")


# ===================================================================== #
# Figure helpers
# ===================================================================== #
def _render_cell_figures(rep_by_cell, methods, x_np, t_np, figs_dir, plt):
    """3-row gridspec per cell per channel: init strip / fields / |err|."""
    panel_channels = ["rho", "w"]
    for (fam, seg), rep in rep_by_cell.items():
        rho0 = rep["rho0"]
        gt = rep["gt"]
        preds = rep["preds"]
        idx = rep["idx"]
        present = [m for m in methods if m in preds]
        for ch in panel_channels:
            gt_arr = gt[ch]
            vmin, vmax = float(gt_arr.min()), float(gt_arr.max())
            panels = [("GT", gt_arr)] + [(m, preds[m][ch]) for m in present]
            ncols = len(panels)

            fig = plt.figure(figsize=(4 * ncols, 9), constrained_layout=True)
            gs = fig.add_gridspec(nrows=3, ncols=ncols, height_ratios=[0.6, 3.0, 3.0])

            # Row 0: rho0 (or the channel-specific) initial profile strip.
            ax_ic = fig.add_subplot(gs[0, :])
            ax_ic.plot(x_np, rho0, color="black", linewidth=1.4)
            ax_ic.set_ylabel("rho0(x)")
            ax_ic.set_xlabel("x")
            ax_ic.set_xlim(float(x_np[0]), float(x_np[-1]))
            ax_ic.grid(True, alpha=0.3)

            # Row 1: space-time fields (jet, shared scale).
            sol_axes = [fig.add_subplot(gs[1, c]) for c in range(ncols)]
            for c, (name, sol) in enumerate(panels):
                im = sol_axes[c].pcolormesh(
                    x_np, t_np, sol, shading="nearest", cmap="jet", vmin=vmin, vmax=vmax
                )
                sol_axes[c].set_title(name)
                sol_axes[c].set_xlabel("x"); sol_axes[c].set_ylabel("t")
                fig.colorbar(im, ax=sol_axes[c], label=ch)

            # Row 2: |solver - GT| (magma, shared scale).
            err_axes = [fig.add_subplot(gs[2, c]) for c in range(ncols)]
            err_vmax = max(np.abs(preds[m][ch] - gt_arr).max() for m in present)
            for c, (name, sol) in enumerate(panels):
                if c == 0:
                    err_axes[c].axis("off")
                    continue
                err = np.abs(sol - gt_arr)
                im = err_axes[c].pcolormesh(
                    x_np, t_np, err, shading="nearest", cmap="magma", vmin=0, vmax=err_vmax
                )
                err_axes[c].set_title(f"|{name} - GT|")
                err_axes[c].set_xlabel("x"); err_axes[c].set_ylabel("t")
                fig.colorbar(im, ax=err_axes[c])

            mae_str = "  ".join(
                f"{m}={mae(preds[m][ch], gt_arr):.3e}" for m in present
            )
            fig.suptitle(
                f"arz_mark0_paper_eval sample {idx} | {format_ic_label(rho0, fam, seg)} "
                f"| channel={ch}\nMAE: {mae_str}",
                fontsize=11,
            )
            fig.savefig(figs_dir / f"compare_{fam}_seg{seg:02d}_{ch}.png", dpi=150)
            plt.close(fig)


def _render_mae_vs_segments(mae_by_cell, ic_types, seg_values, methods, out_dir, plt):
    for ch in ["rho", "w"]:
        for fam in ic_types:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            any_curve = False
            for m in methods:
                means, stds, segs_plot = [], [], []
                for seg in seg_values:
                    vals = mae_by_cell[(fam, seg)][m][ch]
                    if not vals:
                        continue
                    means.append(np.mean(vals)); stds.append(np.std(vals))
                    segs_plot.append(seg)
                if not segs_plot:
                    continue
                ax.errorbar(segs_plot, means, yerr=stds, marker="o", capsize=4,
                            label=m, color=COLORS.get(m))
                any_curve = True
            if not any_curve:
                plt.close(fig)
                continue
            ax.set_xlabel("num_segments")
            ax.set_ylabel(f"MAE on {ch} (mean +/- std)")
            ax.set_yscale("log")
            ax.set_xticks(seg_values)
            ax.set_title(f"MAE vs num_segments: {fam} ({ch})")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.savefig(out_dir / f"mae_vs_num_segments_{fam}_{ch}.png", dpi=150)
            plt.close(fig)


# ===================================================================== #
# Table / text helpers
# ===================================================================== #
def _msd(vals):
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def _write_csv(out_dir, mae_by_cell, ic_types, seg_values, methods,
               cell_is_id, pooled_by_seg, pooled_id, pooled_ood, pooled_all):
    path = out_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["family", "num_segments", "group", "method", "channel",
                     "mae_mean", "mae_std", "n_samples"])
        for fam in ic_types:
            for seg in seg_values:
                cell = (fam, seg)
                group = "ID" if cell_is_id(fam, seg) else "OOD"
                for m in methods:
                    for ch in CHANNELS:
                        msd = _msd(mae_by_cell[cell][m][ch])
                        if msd is None:
                            continue
                        wr.writerow([fam, seg, group, m, ch,
                                     f"{msd[0]:.6e}", f"{msd[1]:.6e}",
                                     len(mae_by_cell[cell][m][ch])])
        for seg in seg_values:
            for m in methods:
                for ch in CHANNELS:
                    msd = _msd(pooled_by_seg[seg][m][ch])
                    if msd is None:
                        continue
                    wr.writerow(["ALL", seg, "ALL", m, ch,
                                 f"{msd[0]:.6e}", f"{msd[1]:.6e}",
                                 len(pooled_by_seg[seg][m][ch])])
        for group_name, pool in (("ID", pooled_id), ("OOD", pooled_ood)):
            for m in methods:
                for ch in CHANNELS:
                    msd = _msd(pool[m][ch])
                    if msd is None:
                        continue
                    wr.writerow(["ALL", "ALL", group_name, m, ch,
                                 f"{msd[0]:.6e}", f"{msd[1]:.6e}",
                                 len(pool[m][ch])])
        for m in methods:
            for ch in CHANNELS:
                msd = _msd(pooled_all[m][ch])
                if msd is None:
                    continue
                wr.writerow(["ALL", "ALL", "ALL", m, ch,
                             f"{msd[0]:.6e}", f"{msd[1]:.6e}",
                             len(pooled_all[m][ch])])
    print(f"[arz_mark0_paper_eval] wrote {path}")


def _cell_tex(vals):
    msd = _msd(vals)
    if msd is None:
        return "--"
    return f"${msd[0]:.2e} \\pm {msd[1]:.2e}$"


def _write_tex(out_dir, mae_by_cell, ic_types, seg_values, methods,
               pooled_id, pooled_ood, pooled_all, pooled_by_seg):
    path = out_dir / "summary.tex"
    ch = HEADLINE_CHANNEL
    lines: list[str] = []
    lines.append(f"% ARZ mark-0 paper evaluation: MAE on {ch} (mean +/- std).")
    lines.append(f"% Headline channel = {ch}; see summary.csv for rho/w/v.")
    lines.append("")

    # Per-family tables (headline channel).
    for fam in ic_types:
        lines.append(f"% --- {fam} ---")
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(
            rf"\caption{{ARZ mark-0 MAE on {ch}, mean $\pm$ std per cell. "
            rf"IC type: \texttt{{{fam}}}.}}"
        )
        lines.append(rf"\label{{tab:arz_mark0_paper_eval_{fam}}}")
        lines.append(r"\begin{tabular}{l" + "c" * len(methods) + "}")
        lines.append(r"\hline")
        lines.append("num\\_segments & " + " & ".join(methods) + r" \\")
        lines.append(r"\hline")
        for seg in seg_values:
            cells = [str(seg)] + [_cell_tex(mae_by_cell[(fam, seg)][m][ch]) for m in methods]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    # Headline ID/OOD/all table.
    lines.append("% --- headline (ID / OOD / all) ---")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{ARZ mark-0 headline: MAE on {ch} (mean $\pm$ std). "
        r"ID = cells whose ic\_type and num\_segments both appear in training.}"
    )
    lines.append(r"\label{tab:arz_mark0_paper_eval_headline}")
    lines.append(r"\begin{tabular}{l" + "c" * len(methods) + "}")
    lines.append(r"\hline")
    lines.append("Subset & " + " & ".join(methods) + r" \\")
    lines.append(r"\hline")
    for label, pool in (("ID", pooled_id), ("OOD", pooled_ood), (r"\textbf{all}", pooled_all)):
        row = [label]
        for m in methods:
            msd = _msd(pool[m][ch])
            if msd is None:
                row.append("--")
            elif label.startswith(r"\textbf"):
                row.append(f"$\\mathbf{{{msd[0]:.2e} \\pm {msd[1]:.2e}}}$")
            else:
                row.append(f"${msd[0]:.2e} \\pm {msd[1]:.2e}$")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # Appendix: per-num_segments breakdown (headline channel).
    lines.append("% --- appendix: per-num_segments (headline channel) ---")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{ARZ mark-0 appendix: per-num\_segments MAE on {ch}, pooled "
        r"across IC types. ID/OOD rows match Table~\ref{tab:arz_mark0_paper_eval_headline}.}"
    )
    lines.append(r"\label{tab:arz_mark0_paper_eval_appendix_seg}")
    lines.append(r"\begin{tabular}{l" + "c" * len(methods) + "}")
    lines.append(r"\hline")
    lines.append("num\\_segments & " + " & ".join(methods) + r" \\")
    lines.append(r"\hline")
    for seg in seg_values:
        cells = [str(seg)] + [_cell_tex(pooled_by_seg[seg][m][ch]) for m in methods]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\hline")
    for label, pool in (("ID", pooled_id), ("OOD", pooled_ood)):
        cells = [label] + [_cell_tex(pool[m][ch]) for m in methods]
        lines.append(" & ".join(cells) + r" \\")
    row = [r"\textbf{all}"]
    for m in methods:
        msd = _msd(pooled_all[m][ch])
        row.append("--" if msd is None else f"$\\mathbf{{{msd[0]:.2e} \\pm {msd[1]:.2e}}}$")
    lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # Appendix: per-channel pooled table (rho/w/v x ID/OOD/all).
    lines.append("% --- appendix: per-channel pooled (rho/w/v) ---")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{ARZ mark-0 appendix: per-channel MAE (mean $\pm$ std), pooled "
        r"by ID/OOD and overall.}"
    )
    lines.append(r"\label{tab:arz_mark0_paper_eval_appendix_channels}")
    lines.append(r"\begin{tabular}{ll" + "c" * len(methods) + "}")
    lines.append(r"\hline")
    lines.append("channel & subset & " + " & ".join(methods) + r" \\")
    lines.append(r"\hline")
    for cc in CHANNELS:
        for label, pool in (("ID", pooled_id), ("OOD", pooled_ood), ("all", pooled_all)):
            cells = [cc, label] + [_cell_tex(pool[m][cc]) for m in methods]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[arz_mark0_paper_eval] wrote {path}")


def _write_txt(out_dir, mae_by_cell, ic_types, seg_values, methods,
               cell_is_id, id_cells, ood_cells,
               pooled_by_seg, pooled_id, pooled_ood, pooled_all):
    path = out_dir / "summary.txt"
    lines = ["arz_mark0_paper_eval -- MAE vs stored GT, mean +/- std (all channels)\n"]
    for fam in ic_types:
        lines.append(f"ic_type = {fam}")
        for seg in seg_values:
            n = len(mae_by_cell[(fam, seg)]["model"]["rho"])
            grp = "ID" if cell_is_id(fam, seg) else "OOD"
            lines.append(f"  num_segments = {seg}  [{grp}]  ({n} samples)")
            for m in methods:
                parts = []
                for ch in CHANNELS:
                    msd = _msd(mae_by_cell[(fam, seg)][m][ch])
                    if msd is None:
                        continue
                    parts.append(f"{ch}={msd[0]:.3e}+/-{msd[1]:.1e}")
                if parts:
                    lines.append(f"    {m:8s}: " + "  ".join(parts))
        lines.append("")

    lines.append("=" * 60)
    lines.append("AGGREGATE (pooled across ic_types)")
    for seg in seg_values:
        lines.append(f"  num_segments = {seg}")
        for m in methods:
            parts = []
            for ch in CHANNELS:
                msd = _msd(pooled_by_seg[seg][m][ch])
                if msd is None:
                    continue
                parts.append(f"{ch}={msd[0]:.3e}")
            if parts:
                lines.append(f"    {m:8s}: " + "  ".join(parts))
    lines.append("")
    lines.append(f"ID cells ({len(id_cells)}): {id_cells}")
    lines.append(f"OOD cells ({len(ood_cells)}): {ood_cells}")
    for label, pool in (("ID", pooled_id), ("OOD", pooled_ood), ("GRAND TOTAL", pooled_all)):
        lines.append(f"  {label}:")
        for m in methods:
            parts = []
            for ch in CHANNELS:
                msd = _msd(pool[m][ch])
                if msd is None:
                    continue
                parts.append(f"{ch}={msd[0]:.4e}")
            if parts:
                lines.append(f"    {m:8s}: " + "  ".join(parts))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[arz_mark0_paper_eval] wrote {path}")


if __name__ == "__main__":
    main()
