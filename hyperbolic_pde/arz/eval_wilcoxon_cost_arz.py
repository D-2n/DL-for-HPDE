"""ARZ Wilcoxon + MAE eval (HypNO + FNO + numerical baselines).

Checkpoint / data locations (override with CLI flags as needed):
  * Data:          hyperbolic_pde/arz/data/arz_evaluation_local.npz
  * Config:        hyperbolic_pde/arz/config_arz_hll_wft.yaml
  * HypNO WFT:     hyperbolic_pde/arz/arz_wft.pt
  * HypNO HLL+WFT: hyperbolic_pde/arz/arz_hll_wft.pt
  * FNO:           hyperbolic_pde/arz/fno_arz_bigger.pt

Outputs in --out-dir (MAE + Wilcoxon only):
  * per_sample_mae.csv   -- per-IC rho MAE
  * summary_stats.csv    -- mean/std MAE per (family, #segs, method)
  * wilcoxon.csv         -- paired Wilcoxon HypNO vs each baseline + Holm p
  * summary.txt          -- printed MAE + Wilcoxon tables

Usage:
    python -m hyperbolic_pde.arz.eval_wilcoxon_cost_arz \\
        --n-per-cell 30 --families piecewise --device cuda \\
        --out-dir hyperbolic_pde/arz/runs/wilcoxon_cost_n30_pwc
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.eval_vs_numerical_arz import _run_baseline
from hyperbolic_pde.arz.model_arz_orig import load_hypno_arz_orig_from_checkpoint
from hyperbolic_pde.arz.reference_arz import solve_arz_weno5
from hyperbolic_pde.models.competitive_architectures.fno import FNO2d

# ---------------------------------------------------------------------------
# Default paths (same as comments above)
# ---------------------------------------------------------------------------
_DEFAULT_DATA = Path("hyperbolic_pde/arz/data/arz_evaluation_local.npz")
_DEFAULT_CONFIG = Path("hyperbolic_pde/arz/config_arz_hll_wft.yaml")
_DEFAULT_CKPT_WFT = Path("hyperbolic_pde/arz/arz_wft.pt")
_DEFAULT_CKPT_HLL_WFT = Path("hyperbolic_pde/arz/arz_hll_wft.pt")
_DEFAULT_FNO = Path("hyperbolic_pde/arz/fno_arz_bigger.pt")

# Match arz_mark0_paper_eval: sine staircase is reported under PWC.
_FAMILY_ALIASES = {"piecewise_sine": "piecewise_constant_stratified"}
_RIEMANN_FAMILY = "riemann_stratified"


def _run_baseline_eval(
    name: str, rho0, w0, bundle, tau: float, boundary: str, weno_pp: str,
):
    """Match eval_evaluation_local WENO settings (clip, CFL 0.2)."""
    if name != "weno5":
        return _run_baseline(name, rho0, w0, bundle, tau, boundary)
    nt = bundle.t.shape[0]
    xb = np.asarray(bundle.x, dtype=np.float64)
    dx = float(xb[1] - xb[0])
    x_min = float(xb[0] - 0.5 * dx)
    x_max = float(xb[-1] + 0.5 * dx)
    t_max = float(bundle.t.max())
    return solve_arz_weno5(
        rho0, w0, x_min, x_max, t_max, nt_out=nt,
        tau=tau, cfl=0.2, boundary=boundary,
        pp_limiter=weno_pp, variant="component",
    )


def canonical_family(name: str) -> str:
    return _FAMILY_ALIASES.get(name, name)


def report_num_segments(raw_family: str, labeled_segments: int) -> int:
    """Physical segment count for tables. Riemann ICs are always 2-piece."""
    if raw_family == _RIEMANN_FAMILY or canonical_family(raw_family) == _RIEMANN_FAMILY:
        return 2
    return int(labeled_segments)


def parse_family_filter(spec: str) -> set[str] | None:
    """Allowed *raw* ic_type names, or None = all.

    ``piecewise`` = PWC + sine (paper merge into piecewise_constant_stratified).
    """
    spec = (spec or "").strip().lower()
    if not spec or spec in ("all", "*"):
        return None
    if spec in ("piecewise", "pwc", "piecewise_only"):
        return {"piecewise_constant_stratified", "piecewise_sine"}
    allowed: set[str] = set()
    for tok in spec.split(","):
        t = tok.strip()
        if not t:
            continue
        if t in ("piecewise", "pwc"):
            allowed |= {"piecewise_constant_stratified", "piecewise_sine"}
        elif t in ("riemann", "riemann_stratified"):
            allowed.add("riemann_stratified")
        else:
            allowed.add(t)
    return allowed or None


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def _make_fno_input(rho0_np, w0_np, x_np, t_np) -> torch.Tensor:
    x_t = torch.tensor(x_np, dtype=torch.float32)
    t_t = torch.tensor(t_np, dtype=torch.float32)
    X, T = torch.meshgrid(x_t, t_t, indexing="ij")
    nt = t_t.numel()
    rho0_g = torch.tensor(rho0_np, dtype=torch.float32).unsqueeze(1).repeat(1, nt)
    w0_g = torch.tensor(w0_np, dtype=torch.float32).unsqueeze(1).repeat(1, nt)
    return torch.stack([X, T, rho0_g, w0_g], dim=0).unsqueeze(0)


def _load_fno(weights: Path, device: str) -> FNO2d:
    model = FNO2d(
        in_channels=4, out_channels=2, width=128, modes_x=24, modes_t=24, layers=12,
    ).to(device)
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


def _wilcoxon_table(
    per_sample: list[dict],
    hypno_methods: list[str],
    baselines: list[str],
) -> list[dict]:
    try:
        from scipy.stats import wilcoxon
    except ImportError as e:
        raise SystemExit("scipy required for Wilcoxon: pip install scipy") from e
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        multipletests = None

    buckets: dict[tuple, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in per_sample:
        key = (row["family"], int(row["num_segments"]))
        buckets[key][row["method"]][int(row["sample_idx"])] = float(row["mae"])

    raw_rows = []
    pvals = []
    for cell, mdict in sorted(buckets.items()):
        fam, seg = cell
        for h in hypno_methods:
            if h not in mdict:
                continue
            for b in baselines:
                if b not in mdict:
                    continue
                common = sorted(set(mdict[h]) & set(mdict[b]))
                if len(common) < 5:
                    continue
                a = np.array([mdict[h][i] for i in common])
                bb = np.array([mdict[b][i] for i in common])
                try:
                    stat, p = wilcoxon(a, bb, alternative="less", zero_method="wilcox")
                except ValueError:
                    stat, p = float("nan"), 1.0
                raw_rows.append(dict(
                    family=fam, num_segments=seg, hypno=h, baseline=b,
                    n=len(common),
                    hypno_mean=float(a.mean()), baseline_mean=float(bb.mean()),
                    hypno_median=float(np.median(a)), baseline_median=float(np.median(bb)),
                    wilcoxon_stat=float(stat) if np.isfinite(stat) else "",
                    p_raw=float(p),
                ))
                pvals.append(float(p))

    if multipletests is not None and pvals:
        _, p_adj, _, _ = multipletests(pvals, method="holm")
        for r, pa in zip(raw_rows, p_adj):
            r["p_holm"] = float(pa)
            r["sig_0.05"] = bool(pa < 0.05)
    else:
        for r in raw_rows:
            r["p_holm"] = r["p_raw"]
            r["sig_0.05"] = bool(r["p_raw"] < 0.05)
            r["note"] = "holm skipped (no statsmodels); p_holm=p_raw"
    return raw_rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ARZ Wilcoxon + rho MAE (no plots / cost tables).",
    )
    # Paths — defaults match the checkpoint comments at the top of this file.
    ap.add_argument("--data", type=Path, default=_DEFAULT_DATA,
                    help=f"Eval NPZ (default: {_DEFAULT_DATA})")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                    help=f"HypNO config YAML (default: {_DEFAULT_CONFIG})")
    ap.add_argument("--ckpt-wft", type=Path, default=_DEFAULT_CKPT_WFT,
                    help=f"HypNO WFT checkpoint (default: {_DEFAULT_CKPT_WFT})")
    ap.add_argument("--ckpt-hll-wft", type=Path, default=_DEFAULT_CKPT_HLL_WFT,
                    help=f"HypNO HLL+WFT checkpoint (default: {_DEFAULT_CKPT_HLL_WFT})")
    ap.add_argument("--fno-weights", type=Path, default=_DEFAULT_FNO,
                    help=f"FNO weights (default: {_DEFAULT_FNO})")
    ap.add_argument("--baselines", type=str, default="godunov,hll,weno5",
                    help="Comma list: godunov,hll,weno5")
    ap.add_argument(
        "--weno-pp", type=str, default="clip", choices=["clip", "zhang_shu", "none"],
        help="WENO positivity limiter; default clip matches eval_evaluation_local",
    )
    ap.add_argument(
        "--n-per-cell", type=int, default=30,
        help="Cap per raw (ic_type, labeled num_segments) NPZ quota cell. "
             "piecewise_sine merges into piecewise_constant_stratified.",
    )
    ap.add_argument(
        "--families", type=str, default="piecewise",
        help="'piecewise' (PWC+sine, skip Riemann), 'all', or comma list of ic_types",
    )
    ap.add_argument("--device", type=str, default="cuda",
                    choices=["cpu", "mps", "cuda"])
    ap.add_argument("--skip-fno", action="store_true")
    ap.add_argument("--skip-hypno", action="store_true")
    ap.add_argument("--hypno-only", type=str, default="",
                    help="Optional: only 'arz_wft' or 'arz_hll_wft'")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("[warn] MPS unavailable; falling back to CPU", flush=True)
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; falling back to CPU", flush=True)
        args.device = "cpu"

    device = args.device
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[paths]", flush=True)
    print(f"  data          = {args.data}", flush=True)
    print(f"  config        = {args.config}", flush=True)
    print(f"  ckpt-wft      = {args.ckpt_wft}", flush=True)
    print(f"  ckpt-hll-wft  = {args.ckpt_hll_wft}", flush=True)
    print(f"  fno-weights   = {args.fno_weights}", flush=True)
    print(f"  out-dir       = {out}", flush=True)

    bundle = load_arz_dataset(args.data)
    P.set_pressure_form(str(getattr(bundle, "p_form", "rho") or "rho"))
    tau = float(bundle.tau) if np.isfinite(float(bundle.tau)) else 1e9
    x_np = np.asarray(bundle.x, dtype=np.float64)
    t_np = np.asarray(bundle.t, dtype=np.float64)
    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_t = torch.tensor(t_np, dtype=torch.float32, device=device)
    N = int(bundle.rho0.shape[0])
    family_allow = parse_family_filter(args.families)
    print(
        f"[families] filter={args.families!r} -> "
        f"{sorted(family_allow) if family_allow else 'ALL'}; "
        f"alias {_FAMILY_ALIASES}",
        flush=True,
    )

    baselines = [s.strip() for s in args.baselines.split(",") if s.strip()]
    hypno_models: dict[str, torch.nn.Module] = {}
    if not args.skip_hypno:
        want = [args.hypno_only] if args.hypno_only else ["arz_wft", "arz_hll_wft"]
        mapping = {
            "arz_wft": args.ckpt_wft,
            "arz_hll_wft": args.ckpt_hll_wft,
        }
        for name in want:
            ckpt = mapping[name]
            if not Path(ckpt).exists():
                raise SystemExit(f"Missing checkpoint for {name}: {ckpt}")
            print(f"[load] {name} from {ckpt} on {device}", flush=True)
            m, _ = load_hypno_arz_orig_from_checkpoint(
                ckpt, config_path=args.config, model_section="hypno_arz_orig",
                device=device,
            )
            m.eval()
            hypno_models[name] = m

    fno_model = None
    if not args.skip_fno:
        if not Path(args.fno_weights).exists():
            print(f"[warn] FNO weights missing ({args.fno_weights}); skipping FNO",
                  flush=True)
        else:
            print(f"[load] FNO from {args.fno_weights} on {device}", flush=True)
            fno_model = _load_fno(args.fno_weights, device)

    methods_order = list(hypno_models.keys())
    if fno_model is not None:
        methods_order.append("FNO")
    methods_order.extend(baselines)

    per_sample: list[dict] = []
    counts: dict[tuple, int] = defaultdict(int)

    for i in range(N):
        raw_fam = str(bundle.ic_type[i])
        if family_allow is not None and raw_fam not in family_allow:
            continue
        fam = canonical_family(raw_fam)
        labeled_seg = int(bundle.num_segments[i])
        seg = report_num_segments(raw_fam, labeled_seg)
        raw_cell = (raw_fam, labeled_seg)
        if counts[raw_cell] >= args.n_per_cell:
            continue
        sample_in_cell = counts[raw_cell]
        counts[raw_cell] += 1

        rho0 = bundle.rho0[i].astype(np.float64)
        w0 = bundle.w0[i].astype(np.float64)
        rho_gt = bundle.rho[i].astype(np.float64)

        tag = fam if raw_fam == fam else f"{fam}←{raw_fam}"
        print(f"[{i+1}/{N}] {tag} seg={seg} ({sample_in_cell+1}/{args.n_per_cell})",
              flush=True)

        preds: dict[str, np.ndarray] = {}

        for name, model in hypno_models.items():
            with torch.no_grad():
                rho_p, _, _ = model(
                    torch.tensor(rho0[None], dtype=torch.float32, device=device),
                    torch.tensor(w0[None], dtype=torch.float32, device=device),
                    x_t, t_t,
                )
            preds[name] = rho_p[0].cpu().numpy()

        if fno_model is not None:
            inp = _make_fno_input(rho0, w0, x_np, t_np).to(device)
            with torch.no_grad():
                fno_out = fno_model(inp)
            preds["FNO"] = fno_out[0, 0].cpu().numpy().T

        for m in baselines:
            try:
                rho_b, _ = _run_baseline_eval(
                    m, rho0, w0, bundle, tau, "ghost", args.weno_pp,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {m} failed: {e}", flush=True)
                continue
            preds[m] = rho_b

        for m, rho_p in preds.items():
            per_sample.append(dict(
                sample_idx=i,
                family=fam,
                num_segments=seg,
                method=m,
                channel="rho",
                mae=mae(rho_p, rho_gt),
            ))

    if not per_sample:
        raise SystemExit("No samples evaluated — check --families / --n-per-cell / data")

    # ---- per-sample MAE ----
    ps_path = out / "per_sample_mae.csv"
    with ps_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        w.writerows(per_sample)
    print(f"[out] {ps_path} ({len(per_sample)} rows)", flush=True)

    # ---- summary MAE (mean ± std) ----
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in per_sample:
        key = (row["family"], row["num_segments"], row["method"])
        groups[key].append(row["mae"])

    stats_rows = []
    for key, vals in sorted(groups.items()):
        fam, seg, method = key
        arr = np.asarray(vals, dtype=np.float64)
        stats_rows.append(dict(
            family=fam, num_segments=seg, method=method, channel="rho",
            n=len(arr),
            mae_mean=float(arr.mean()),
            mae_std=float(arr.std()),
            note="std is across samples (ICs), not SEM",
        ))
    st_path = out / "summary_stats.csv"
    with st_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)
    print(f"[out] {st_path}", flush=True)

    # ---- Wilcoxon ----
    hypno_names = list(hypno_models.keys())
    base_for_test = list(baselines)
    if fno_model is not None:
        base_for_test = ["FNO"] + base_for_test
    wx_rows = _wilcoxon_table(per_sample, hypno_names, base_for_test)
    if len(hypno_names) == 2:
        wx_rows += _wilcoxon_table(per_sample, [hypno_names[0]], [hypno_names[1]])
    wx_path = out / "wilcoxon.csv"
    if wx_rows:
        with wx_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(wx_rows[0].keys()))
            w.writeheader()
            w.writerows(wx_rows)
        print(f"[out] {wx_path}", flush=True)

    # ---- pooled MAE + Wilcoxon text ----
    lines = [
        "ARZ Wilcoxon + MAE (rho)",
        f"device={device}  n_per_cell={args.n_per_cell}  families={args.families}",
        "std = sample std across ICs (not SEM).",
        "",
        "=== MAE (rho, pooled) ===",
    ]
    for method in methods_order:
        maes = [r["mae"] for r in per_sample if r["method"] == method]
        if not maes:
            continue
        lines.append(
            f"  {method:12s}  MAE={float(np.mean(maes)):.3e}±{float(np.std(maes)):.2e}  "
            f"n={len(maes)}"
        )
    lines.append("")
    lines.append("=== MAE (rho, per cell) ===")
    for r in stats_rows:
        lines.append(
            f"  {r['family'][:20]:20s} N={r['num_segments']:<3d}  "
            f"{r['method']:12s}  "
            f"MAE={r['mae_mean']:.3e}±{r['mae_std']:.2e}  n={r['n']}"
        )
    lines.append("")
    lines.append("=== Wilcoxon (rho): HypNO < baseline, Holm-adjusted ===")
    for r in wx_rows:
        flag = "*" if r.get("sig_0.05") else ""
        lines.append(
            f"  {r['family'][:12]:12s} N={r['num_segments']:<3d}  "
            f"{r['hypno']:12s} < {r['baseline']:8s}  "
            f"p_holm={r['p_holm']:.3g}{flag}  n={r['n']}"
        )
    txt = out / "summary.txt"
    txt.write_text("\n".join(lines) + "\n")
    print(f"[out] {txt}", flush=True)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
