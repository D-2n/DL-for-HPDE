"""Mark-0 (orig) ARZ shock-band paper export: HypNO-ARZ-orig vs FNO vs the
numerical baselines (Godunov / HLL / WENO5), with MAE isolated to the two
discontinuity bands.

The mark-0 analogue of arz_shock_paper_export.py (which targets the mark-1
model), exactly mirroring how arz_mark0_paper_eval.py extends arz_paper_eval.py.
Only two things change vs the mark-1 export:
  (a) the model loader is load_hypno_arz_orig_from_checkpoint (+ --model-section);
  (b) an optional FNO column (4-in/2-out FNO2d) is woven into the methods.
Everything else -- the band detectors (1-shock on lambda_1/v, contact on rho),
the figure/table/section emitters, the rho-headline convention -- is REUSED from
arz_shock_paper_export so any change there propagates here.

Outputs (under <out_dir>, default <ckpt_dir>/arz_mark0_paper_shock):
    figs/shock_mae_vs_num_segments.png, shock_{compare,zoom,slice}_num_segments_{seg}.png
    shock_results.tex, shock_table.tex, shock_summary.txt

Example:
    python -m hyperbolic_pde.arz.arz_mark0_shock_paper_export \\
        --ckpt   hyperbolic_pde/runs/checkpoint_epoch62.pt \\
        --config hyperbolic_pde/runs/config.yaml \\
        --model-section hypno_arz_orig \\
        --data   hyperbolic_pde/arz/data/arz_evaluation_prho.npz \\
        --baselines godunov,hll,weno5 \\
        --fno-weights hyperbolic_pde/arz/runs/fno_arz_local/fno_arz_bigger.pt \\
        --fno-config  hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml \\
        --fno-section fno_arz_bigger \\
        --out_dir hyperbolic_pde/arz/runs/arz_evaluation_paper_shock
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
from hyperbolic_pde.arz.model_arz_orig import load_hypno_arz_orig_from_checkpoint
from hyperbolic_pde.arz.eval_shock_arz import (
    detect_1shock_band, detect_contact_band, _mae,
)
# Use the eval_vs_numerical_arz baseline runner (NOT eval_shock_arz's, which only
# knows godunov/weno5 and raises on 'hll'). This one supports godunov/hll/hllc/
# weno5 and is the SAME runner arz_mark0_paper_eval uses -> table & shock baselines
# stay consistent.
from hyperbolic_pde.arz.eval_vs_numerical_arz import _run_baseline
from hyperbolic_pde.arz.arz_baseline_cache import load_baseline_cache, cached_sample
# Reuse the FNO helpers and ALL figure/table/section emitters from the existing
# scripts so the mark-0 export is byte-for-byte consistent with the mark-1 one.
from hyperbolic_pde.arz.arz_mark0_paper_eval import _make_fno_input, _load_fno
from hyperbolic_pde.arz import arz_shock_paper_export as S

# Extend the shared colour / label maps with the FNO and HLL methods (the
# mark-1 export only knew model/weno5/godunov). The plot/table helpers read
# these module-level dicts via .get(), so updating them in place is enough.
S.COLORS.update({"FNO": "tab:purple", "hll": "tab:red"})
S.TEX_NAME.update({"FNO": r"FNO", "hll": r"HLL"})


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mark-0 (orig) ARZ shock-band export: HypNO-orig vs FNO vs "
                    "Godunov/HLL/WENO5, MAE isolated to shock + contact bands."
    )
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None,
                    help="Run config with the hypno_arz_orig section (auto-located "
                         "beside the ckpt if omitted).")
    ap.add_argument("--model-section", type=str, default="hypno_arz_orig")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--baselines", type=str, default="godunov,hll,weno5")
    ap.add_argument("--baselines-npz", type=Path, default=None,
                    help="Precomputed baseline cache (arz_precompute_baselines). "
                         "Cached methods are loaded instead of re-solved; "
                         "uncached methods still solve on the fly.")
    ap.add_argument("--n_per_group", type=int, default=None)
    ap.add_argument("--tau-shock", type=float, default=0.06)
    ap.add_argument("--band-halfwidth", type=int, default=2)
    ap.add_argument("--tv-mult", type=float, default=1.5)
    ap.add_argument("--tau", type=float, default=None,
                    help="Override the relaxation tau for baselines (defaults to bundle.tau).")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rep-band-frac", type=float, default=0.05)
    # FNO column (optional) -- same flags as arz_mark0_paper_eval.
    ap.add_argument("--fno-weights", type=Path, default=None,
                    help="FNO checkpoint (.pt). FNO column omitted if not given.")
    ap.add_argument("--fno-config", type=Path, default=None,
                    help="config.yaml containing the FNO architecture section.")
    ap.add_argument("--fno-section", type=str, default="fno_arz",
                    help="Config section for FNO architecture (default fno_arz).")
    ap.add_argument("--no-fno", action="store_true",
                    help="Skip the FNO column even if weights are provided.")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Output directory (default: <ckpt_dir>/arz_mark0_paper_shock).")
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
    S.configure_paper_style(plt)

    bundle = load_arz_dataset(args.data)
    tau = float(args.tau) if args.tau is not None else float(bundle.tau)
    N = bundle.rho.shape[0]

    from hyperbolic_pde.arz.physics_arz import set_pressure_form, get_pressure_form
    set_pressure_form(bundle.p_form)
    print(f"[arz_mark0_shock_paper_export] pressure_form={get_pressure_form()}")

    config_path = S._locate_config(args.ckpt, args.config)
    model, _ck_tau = load_hypno_arz_orig_from_checkpoint(
        args.ckpt, device=args.device, config_path=config_path,
        model_section=args.model_section,
    )
    print(f"[arz_mark0_shock_paper_export] loaded {args.ckpt}  tau_eval={tau}")

    # ---- Optional precomputed baseline cache ------------------------- #
    baseline_cache = None
    if args.baselines_npz is not None:
        baseline_cache = load_baseline_cache(args.baselines_npz, bundle)
        print(f"[arz_mark0_shock_paper_export] using baseline cache "
              f"{args.baselines_npz} (methods: {sorted(baseline_cache)})")

    # ---- Optional FNO column ----------------------------------------- #
    fno_model = None
    if args.no_fno:
        print("[arz_mark0_shock_paper_export] --no-fno given; skipping FNO column.")
    elif args.fno_weights is not None:
        if not args.fno_weights.exists():
            print(f"[arz_mark0_shock_paper_export] FNO weights not found at "
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
            print(f"[arz_mark0_shock_paper_export] loaded FNO: {args.fno_weights}")

    out_dir = args.out_dir or (args.ckpt.parent / "arz_mark0_paper_shock")
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x_np = bundle.x.astype(np.float64)
    t_np = bundle.t.astype(np.float64)
    x_t = torch.tensor(bundle.x, dtype=torch.float32, device=args.device)
    t_t = torch.tensor(bundle.t, dtype=torch.float32, device=args.device)

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    # Method order: model, FNO (if present), then numerical baselines.
    methods = ["model"] + (["FNO"] if fno_model is not None else []) + baselines
    baseline_tau = tau if np.isfinite(tau) else 1e6

    seg_values = sorted(set(int(s) for s in bundle.num_segments))
    # mae_by_region[region][seg][method] -> list of per-sample rho MAE.
    mae_by_region = {
        r: {seg: {m: [] for m in methods} for seg in seg_values} for r in S.ACCUM_REGIONS
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
        region_masks = {"full": None, "1shock": mask_1shock, "contact": mask_contact,
                        "combined": mask_combined}

        print(
            f"[{i+1}/{N}] seg={seg} fam={fam} "
            f"1sh={mask_1shock.mean()*100:.2f}% con={mask_contact.mean()*100:.2f}%",
            flush=True,
        )

        # Model (mark-0 / orig): same forward signature as mark-1.
        with torch.no_grad():
            rho_p, w_p, _ = model(
                torch.tensor(rho0[None], dtype=torch.float32, device=args.device),
                torch.tensor(w0[None], dtype=torch.float32, device=args.device),
                x_t, t_t,
            )
        rho_p = rho_p[0].cpu().numpy(); w_p = w_p[0].cpu().numpy()
        preds = {"model": {"rho": rho_p, "w": w_p, "v": w_p - P.pressure(rho_p)}}

        # FNO column.
        if fno_model is not None:
            inp = _make_fno_input(rho0, w0, x_np, t_np).to(args.device)
            with torch.no_grad():
                out = fno_model(inp)  # (1, 2, nx, nt)
            rho_f = out[0, 0].cpu().numpy().T  # (nt, nx)
            w_f = out[0, 1].cpu().numpy().T
            preds["FNO"] = {"rho": rho_f, "w": w_f, "v": w_f - P.pressure(rho_f)}

        # Numerical baselines (cache hit -> no re-solve; cached NaN -> skip).
        for m in baselines:
            if baseline_cache is not None and m in baseline_cache:
                hit = cached_sample(baseline_cache, m, i)
                if hit is None:
                    continue  # cache says this method failed on this sample
                rho_b, w_b = hit
            else:
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

    # ---- Plots (reused from arz_shock_paper_export) ----
    S.plot_mae_vs_segments(
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
        S.plot_compare_full(rep, x_np, t_np, fig_dir / f"shock_compare_num_segments_{seg}.png", methods, plt)
        S.plot_zoom(rep, x_np, t_np, fig_dir / f"shock_zoom_num_segments_{seg}.png", methods, plt)
        S.plot_slice(rep, x_np, t_np, fig_dir / f"shock_slice_num_segments_{seg}.png", methods, plt)

    # ---- LaTeX ----
    table_tex = S.emit_table(seg_values, mae_by_region, methods)
    section_tex = S.emit_section(table_tex, seg_values, args.tau_shock, args.band_halfwidth, bundle.p_form)
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
            for region in S.REGIONS:
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
