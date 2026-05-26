"""ARZ shock-band eval (plan §8b).

The discontinuity detector is run on the ground-truth fields, split by wave
type:

  * 1-shock band   : |dv| large with w approximately continuous.
                     Sign by Lax test: lambda1(U_L) < lambda1(U_R).
  * 2-contact band : |drho| large with v approximately continuous.

Both detectors use a TV-multiplier pass on the appropriate signed-jump
indicator, followed by a +/- b cell dilation. Reports MAE on the full field
and inside each band (and combined).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.model_arz import HypNO_ARZ


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def _tv(field: np.ndarray) -> float:
    """Total variation along axis -1."""
    return float(np.abs(np.diff(field, axis=-1)).sum())


def _dilate_1d(mask: np.ndarray, half: int) -> np.ndarray:
    """+/- half cell dilation along the last axis."""
    if half <= 0:
        return mask
    out = mask.copy()
    for k in range(1, half + 1):
        out[..., k:]    |= mask[..., :-k]
        out[..., :-k]   |= mask[..., k:]
    return out


def detect_1shock_band(
    rho: np.ndarray, w: np.ndarray, v: np.ndarray,
    tau_shock: float, half: int, tv_mult: float,
) -> np.ndarray:
    """Boolean mask of cells in a 1-shock band, computed on GT fields.

    Criterion (per timeslice):
      jump_v = -dv (decreasing v across a Lax 1-shock for concave LWR-like family)
      indicator = relu(-dv); mark cells where indicator > max(tau_shock, tv_mult * TV(v)/nx).
      Then +/- half dilation. Sign of the jump is consistent with lambda1_L < lambda1_R.
    """
    nt, nx = rho.shape
    out = np.zeros_like(rho, dtype=bool)
    dp_rho = 1.0 + 2.0 * rho
    lam1 = v - rho * dp_rho
    for k in range(nt):
        dv = np.diff(v[k])                              # nx-1
        dlam1 = np.diff(lam1[k])                        # nx-1, < 0 implies Lax-1 shock
        # Signed indicator: penalise drops in lambda1 (Lax) and drops in v.
        ind = np.maximum(-dlam1, 0.0) + np.maximum(-dv, 0.0)
        thr = max(tau_shock, tv_mult * _tv(v[k]) / max(nx, 1))
        # Map interface mask back to cells (cell i is "in" if interface i-1/2 or i+1/2 flagged).
        cell = np.zeros(nx, dtype=bool)
        flagged = ind > thr
        cell[1:]  |= flagged
        cell[:-1] |= flagged
        out[k] = cell
    return _dilate_1d(out, half)


def detect_contact_band(
    rho: np.ndarray, w: np.ndarray, v: np.ndarray,
    tau_shock: float, half: int, tv_mult: float,
) -> np.ndarray:
    """Boolean mask of cells in a 2-contact band: |drho| large with v approx continuous."""
    nt, nx = rho.shape
    out = np.zeros_like(rho, dtype=bool)
    for k in range(nt):
        drho = np.diff(rho[k])
        dv   = np.diff(v[k])
        ind  = np.abs(drho) - 0.5 * np.abs(dv)          # prefer drho-only jumps
        thr  = max(tau_shock, tv_mult * _tv(rho[k]) / max(nx, 1))
        cell = np.zeros(nx, dtype=bool)
        flagged = ind > thr
        cell[1:]  |= flagged
        cell[:-1] |= flagged
        out[k] = cell
    return _dilate_1d(out, half)


# --------------------------------------------------------------------------- #
# Eval driver
# --------------------------------------------------------------------------- #
def _mae(arr_pred, arr_gt, mask=None):
    if mask is None:
        return float(np.mean(np.abs(arr_pred - arr_gt)))
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(arr_pred[mask] - arr_gt[mask])))


def _run_baseline(name, rho0, w0, bundle, tau, boundary):
    nt = bundle.t.shape[0]
    x_min = float(bundle.x.min()); x_max = float(bundle.x.max())
    t_max = float(bundle.t.max())
    if name == "godunov":
        _, _, rho_h, w_h = Ref.solve_arz_reference(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=tau, cfl=0.4, boundary=boundary, refine=1,
        )
        return rho_h, w_h
    if name == "weno5":
        from hyperbolic_pde.arz.eval_vs_numerical_arz import solve_arz_weno5
        return solve_arz_weno5(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=tau, cfl=0.2, boundary=boundary,
        )
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--baselines", type=str, default="weno5,godunov")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--tau-shock", type=float, default=0.06)
    ap.add_argument("--band-halfwidth", type=int, default=2)
    ap.add_argument("--tv-mult", type=float, default=1.5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", type=Path, default=None)
    args = ap.parse_args()

    bundle = load_arz_dataset(args.data)
    tau = float(bundle.tau)
    N = bundle.rho.shape[0]
    take = min(args.samples, N)

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model_args = ck["args"]
    model = HypNO_ARZ(
        stencil_k_x=model_args["kx"], stencil_k_t=model_args["kt"],
        d_latent=model_args["d_latent"], d_hidden=model_args["d_hidden"],
        n_layers=model_args["depth"], decoder_depth=model_args["decoder_depth"],
        skip=model_args["skip"], use_checkpoint=False,
        normalize_edge_offsets=model_args.get("normalize_edge_offsets", True),
    ).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()

    x = torch.tensor(bundle.x, dtype=torch.float32, device=args.device)
    t = torch.tensor(bundle.t, dtype=torch.float32, device=args.device)

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    methods = ["model"] + baselines

    # rows: family, num_segments, method, region, channel, mean, std, n
    rows = []
    # accumulate per (fam,seg,method,region,channel) -> list of MAE
    acc: dict = {}

    for i in range(take):
        fam = str(bundle.ic_type[i])
        seg = int(bundle.num_segments[i])
        rho_gt = bundle.rho[i].astype(np.float64)
        w_gt   = bundle.w[i].astype(np.float64)
        v_gt   = w_gt - (rho_gt + rho_gt * rho_gt)
        rho0 = bundle.rho0[i].astype(np.float64)
        w0   = bundle.w0[i].astype(np.float64)

        # Bands on GT.
        mask_1shock = detect_1shock_band(rho_gt, w_gt, v_gt,
                                         args.tau_shock, args.band_halfwidth, args.tv_mult)
        mask_contact = detect_contact_band(rho_gt, w_gt, v_gt,
                                           args.tau_shock, args.band_halfwidth, args.tv_mult)
        mask_combined = mask_1shock | mask_contact

        # Predictions per method.
        preds = {}
        with torch.no_grad():
            rho_p, w_p, _ = model(
                torch.tensor(rho0[None], dtype=torch.float32, device=args.device),
                torch.tensor(w0[None],   dtype=torch.float32, device=args.device),
                x, t,
            )
        preds["model"] = (rho_p[0].cpu().numpy(), w_p[0].cpu().numpy())
        for m in baselines:
            try:
                preds[m] = _run_baseline(m, rho0, w0, bundle, tau, args.boundary)
            except Exception as e:
                print(f"  [warn] {m} sample {i}: {e}", flush=True)

        for m, (rho_b, w_b) in preds.items():
            v_b = w_b - (rho_b + rho_b * rho_b)
            for ch_name, arr_b, arr_gt in (
                ("rho", rho_b, rho_gt), ("w", w_b, w_gt), ("v", v_b, v_gt),
            ):
                regions = {
                    "full":     None,
                    "1shock":   mask_1shock,
                    "contact":  mask_contact,
                    "combined": mask_combined,
                }
                for r_name, mask in regions.items():
                    val = _mae(arr_b, arr_gt, mask)
                    if not np.isfinite(val):
                        continue
                    acc.setdefault((fam, seg, m, r_name, ch_name), []).append(val)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["family", "num_segments", "method", "region", "channel",
                     "mean", "std", "n"])
        for key, vals in sorted(acc.items()):
            arr = np.array(vals, dtype=np.float64)
            wr.writerow([*key, float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)])
    print(f"\n[eval_shock_arz] wrote {args.out}  (N={take} samples)")

    # Pretty print key combined-region rows.
    print("\n  region=combined, channel=rho")
    for key, vals in sorted(acc.items()):
        fam, seg, m, region, ch = key
        if region != "combined" or ch != "rho":
            continue
        arr = np.array(vals)
        print(f"    {fam} seg={seg}  {m:8s}  {arr.mean():.3e} +/- {arr.std():.1e}")


if __name__ == "__main__":
    main()
