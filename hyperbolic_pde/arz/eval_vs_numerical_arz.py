"""ARZ eval: model vs numerical baselines (plan §8a).

Baselines:
  * godunov  : the existing reference solver (HLL + Strang relaxation).
  * weno5    : component-wise WENO5 (global LF) on (rho, y) + SSP-RK3,
               with Strang-split exact relaxation step.

Per-channel MAE on (rho, w, v), mean +/- std stratified by
(ic_type, num_segments). CSV output mirrors the LWR table.
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
from hyperbolic_pde.arz.model_arz import HypNO_ARZ, load_hypno_arz_from_checkpoint
from hyperbolic_pde.data.fvm import _apply_ghost_bc, _weno5_reconstruct_left


# --------------------------------------------------------------------------- #
# WENO5 systemized
# --------------------------------------------------------------------------- #
def _weno5_arz_rhs(rho, y, dx, boundary):
    """Component-wise WENO5 + global LF on (rho, y)."""
    # Use the spectral radius across cells as a global LF coefficient.
    rho_safe = np.maximum(rho, P.RHO_MIN)
    w = y / rho_safe
    v = w - P.pressure(rho)
    lam1 = v - rho * P.dpressure(rho)
    lam2 = v
    alpha = max(float(np.max(np.maximum(np.abs(lam1), np.abs(lam2)))), 1e-8)

    F1 = rho * v
    F2 = y * v

    def _component_rhs(u, f):
        ng = 3
        u_ext = _apply_ghost_bc(u, boundary, ng)
        f_ext = _apply_ghost_bc(f, boundary, ng)
        fp = 0.5 * (f_ext + alpha * u_ext)
        fm = 0.5 * (f_ext - alpha * u_ext)
        fp_s = fp[0:-1]
        fm_s = fm[1:][::-1]
        fhat_p = _weno5_reconstruct_left(fp_s)
        fhat_m = _weno5_reconstruct_left(fm_s)[::-1]
        fhat = fhat_p + fhat_m
        return -(fhat[1:] - fhat[:-1]) / dx

    return _component_rhs(rho, F1), _component_rhs(y, F2)


def _weno5_arz_step(rho, y, dx, dt, boundary):
    """Single SSP-RK3 WENO step on (rho, y)."""
    def L(rho_, y_):
        return _weno5_arz_rhs(rho_, y_, dx, boundary)

    Lr1, Ly1 = L(rho, y)
    rho1 = rho + dt * Lr1
    y1   = y   + dt * Ly1
    rho1 = np.maximum(rho1, P.RHO_MIN)

    Lr2, Ly2 = L(rho1, y1)
    rho2 = 0.75 * rho + 0.25 * (rho1 + dt * Lr2)
    y2   = 0.75 * y   + 0.25 * (y1   + dt * Ly2)
    rho2 = np.maximum(rho2, P.RHO_MIN)

    Lr3, Ly3 = L(rho2, y2)
    rho_n = (1.0/3.0) * rho + (2.0/3.0) * (rho2 + dt * Lr3)
    y_n   = (1.0/3.0) * y   + (2.0/3.0) * (y2   + dt * Ly3)
    return np.maximum(rho_n, P.RHO_MIN), y_n


def solve_arz_weno5(rho0, w0, x_min, x_max, t_max, nt_out, tau, cfl=0.2, boundary="ghost"):
    """Strang-split WENO5(arz) + exact source. No grid refinement here -- the
    point of WENO is to ride the train grid."""
    rho0 = np.asarray(rho0, dtype=np.float64)
    w0   = np.asarray(w0,   dtype=np.float64)
    nx = rho0.size
    dx = (x_max - x_min) / (nx - 1)
    t_out = np.linspace(0.0, t_max, nt_out, dtype=np.float64)
    rho = np.maximum(rho0.copy(), P.RHO_MIN)
    y = rho * w0
    rho_hist = np.empty((nt_out, nx), dtype=np.float32)
    w_hist   = np.empty((nt_out, nx), dtype=np.float32)
    rho_hist[0] = rho.astype(np.float32)
    w_hist[0]   = (y / np.maximum(rho, P.RHO_MIN)).astype(np.float32)

    t = 0.0
    k = 1
    while k < nt_out:
        w_now = y / np.maximum(rho, P.RHO_MIN)
        lam_max = float(P.spectral_radius(rho, w_now).max())
        lam_max = max(lam_max, 1e-6)
        dt = cfl * dx / lam_max
        dt = min(dt, t_out[k] - t)
        if dt <= 0:
            break
        # Strang.
        y = P.y_eq(rho) + (y - P.y_eq(rho)) * np.exp(-0.5 * dt / tau)
        rho, y = _weno5_arz_step(rho, y, dx, dt, boundary)
        y = P.y_eq(rho) + (y - P.y_eq(rho)) * np.exp(-0.5 * dt / tau)
        t += dt
        while k < nt_out and t >= t_out[k] - 1e-12:
            rho_hist[k] = rho.astype(np.float32)
            w_hist[k]   = (y / np.maximum(rho, P.RHO_MIN)).astype(np.float32)
            k += 1
    return rho_hist, w_hist


# --------------------------------------------------------------------------- #
# Eval driver
# --------------------------------------------------------------------------- #
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
        return solve_arz_weno5(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=tau, cfl=0.2, boundary=boundary,
        )
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Checkpoint file (legacy dict OR bare state_dict from the new trainer).")
    ap.add_argument("--config", type=Path, default=None,
                    help="config.yaml describing the architecture (required for "
                         "bare-state_dict checkpoints if it can't be auto-located "
                         "next to the ckpt).")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--baselines", type=str, default="weno5,godunov")
    ap.add_argument("--samples", type=int, default=20,
                    help="Cap on number of eval samples (use all if dataset smaller).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--tau", type=float, default=None,
                    help="Override the relaxation tau used by the baselines "
                         "(defaults to the dataset's bundle.tau).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bundle = load_arz_dataset(args.data)
    tau = float(args.tau) if args.tau is not None else float(bundle.tau)
    N = bundle.rho.shape[0]
    take = min(args.samples, N)

    # Load model (handles both legacy dict and new bare-state_dict formats).
    model, _ck_tau = load_hypno_arz_from_checkpoint(
        args.ckpt, device=args.device, config_path=args.config,
    )
    print(f"[eval_vs_numerical_arz] loaded {args.ckpt}  tau_eval={tau}")

    x = torch.tensor(bundle.x, dtype=torch.float32, device=args.device)
    t = torch.tensor(bundle.t, dtype=torch.float32, device=args.device)

    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    results = []  # rows: family, num_segments, method, channel, mean, std, n

    # Iterate samples, compute MAE per (family, segments, method, channel).
    buckets: dict = {}
    for i in range(take):
        fam = str(bundle.ic_type[i])
        seg = int(bundle.num_segments[i])
        key = (fam, seg)
        if key not in buckets:
            buckets[key] = {m: {"rho": [], "w": [], "v": []}
                            for m in (["model"] + baselines)}

        rho_gt = bundle.rho[i].astype(np.float64)
        w_gt   = bundle.w[i].astype(np.float64)
        v_gt   = w_gt - (rho_gt + rho_gt * rho_gt)
        rho0   = bundle.rho0[i].astype(np.float64)
        w0     = bundle.w0[i].astype(np.float64)

        # Model.
        with torch.no_grad():
            rho_p, w_p, _ = model(
                torch.tensor(rho0[None], dtype=torch.float32, device=args.device),
                torch.tensor(w0[None],   dtype=torch.float32, device=args.device),
                x, t,
            )
        rho_p = rho_p[0].cpu().numpy(); w_p = w_p[0].cpu().numpy()
        v_p = w_p - (rho_p + rho_p * rho_p)
        buckets[key]["model"]["rho"].append(float(np.mean(np.abs(rho_p - rho_gt))))
        buckets[key]["model"]["w"  ].append(float(np.mean(np.abs(w_p   - w_gt  ))))
        buckets[key]["model"]["v"  ].append(float(np.mean(np.abs(v_p   - v_gt  ))))

        # Baselines.
        for m in baselines:
            try:
                rho_b, w_b = _run_baseline(m, rho0, w0, bundle, tau, args.boundary)
            except Exception as e:
                print(f"  [warn] {m} failed on sample {i}: {e}", flush=True)
                continue
            v_b = w_b - (rho_b + rho_b * rho_b)
            buckets[key][m]["rho"].append(float(np.mean(np.abs(rho_b - rho_gt))))
            buckets[key][m]["w"  ].append(float(np.mean(np.abs(w_b   - w_gt  ))))
            buckets[key][m]["v"  ].append(float(np.mean(np.abs(v_b   - v_gt  ))))

    # Emit CSV.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["family", "num_segments", "method", "channel", "mean", "std", "n"])
        for (fam, seg), bymethod in sorted(buckets.items()):
            for m, chans in bymethod.items():
                for ch in ("rho", "w", "v"):
                    arr = np.array(chans[ch], dtype=np.float64)
                    if arr.size == 0:
                        continue
                    wr.writerow([fam, seg, m, ch, float(arr.mean()),
                                 float(arr.std(ddof=0)), int(arr.size)])

    # Pretty print to stdout.
    print(f"\n[eval_vs_numerical_arz] wrote {args.out}  (N={take} samples)")
    for (fam, seg), bymethod in sorted(buckets.items()):
        print(f"  {fam} seg={seg}:")
        for m, chans in bymethod.items():
            cells = []
            for ch in ("rho", "w", "v"):
                arr = np.array(chans[ch])
                if arr.size:
                    cells.append(f"{ch} {arr.mean():.3e}+/-{arr.std():.1e}")
            print(f"    {m:8s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
