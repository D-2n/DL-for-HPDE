"""Quick dataset inspector: shape, ranges, IC counts, pressure-form residual check.

Usage:
    python -m hyperbolic_pde.arz.inspect_dataset --data /path/to/dataset.npz
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()

    data_path = args.data
    print(f"=== {data_path} ===")

    d = np.load(data_path, allow_pickle=True)
    print("Keys:", d.files)

    rho = d["rho"]   # (N, nt, nx)
    w   = d["w"]
    v   = d["v"] if "v" in d else None
    N, nt, nx = rho.shape
    print(f"shape: (N={N}, nt={nt}, nx={nx})")

    for key in ("p_form", "tau", "rare_delta"):
        if key in d:
            print(f"{key}: {d[key]}")

    if "ic_type" in d:
        ic_types = d["ic_type"]
        vals, cnts = np.unique(ic_types, return_counts=True)
        print("ic_types:", dict(zip(vals, cnts.tolist())))
    if "num_segments" in d:
        segs = d["num_segments"].astype(int)
        vals, cnts = np.unique(segs, return_counts=True)
        print("num_segments:", dict(zip(vals.tolist(), cnts.tolist())))

    print()
    print("--- Field ranges ---")
    print(f"rho: [{rho.min():.4f}, {rho.max():.4f}]  mean={rho.mean():.4f}")
    print(f"w  : [{w.min():.4f},  {w.max():.4f}]  mean={w.mean():.4f}")
    if v is not None:
        print(f"v  : [{v.min():.4f},  {v.max():.4f}]  mean={v.mean():.4f}")

    print()
    print("--- Pressure-form check (omega residuals) ---")
    # p=rho:      w = v + rho         => omega_rho   = w - v - rho        ~ 0
    # p=rho+rho2: w = v + rho + rho2  => omega_rho2  = w - v - rho - rho2 ~ 0

    if v is not None:
        omega_rho  = w - v - rho
        omega_rho2 = w - v - rho - rho**2
        print(f"  w - v - rho       (p=rho):        MAE={np.mean(np.abs(omega_rho)):.3e}  max={np.max(np.abs(omega_rho)):.3e}")
        print(f"  w - v - rho - rho² (p=rho+rho2):  MAE={np.mean(np.abs(omega_rho2)):.3e}  max={np.max(np.abs(omega_rho2)):.3e}")
        if np.mean(np.abs(omega_rho)) < 1e-4:
            print("  => CONFIRMED: p = rho")
        elif np.mean(np.abs(omega_rho2)) < 1e-4:
            print("  => CONFIRMED: p = rho + rho²")
        else:
            print("  => UNCLEAR: neither residual is small — data may be corrupted")
    else:
        v_if_rho  = w - rho
        v_if_rho2 = w - rho - rho**2
        print("  No 'v' key — checking v-range plausibility under each pressure form:")
        print(f"    v = w - rho       range: [{v_if_rho.min():.4f}, {v_if_rho.max():.4f}]")
        print(f"    v = w - rho-rho²  range: [{v_if_rho2.min():.4f}, {v_if_rho2.max():.4f}]")
        print("    (physical v should be in ~[0,1])")

    print()
    print("--- Physical sanity ---")
    frac_neg   = float(np.mean(rho < 0))
    frac_over1 = float(np.mean(rho > 1))
    print(f"  rho < 0 : {frac_neg*100:.2f}% of cells")
    print(f"  rho > 1 : {frac_over1*100:.2f}% of cells")
    if v is not None:
        frac_v_neg   = float(np.mean(v < -0.05))
        frac_v_over1 = float(np.mean(v > 1.05))
        print(f"  v < -0.05: {frac_v_neg*100:.2f}%   v > 1.05: {frac_v_over1*100:.2f}%")

    print()
    print("--- IC fields (stored rho0/w0) ---")
    rho0 = d["rho0"] if "rho0" in d else rho[:, 0, :]
    w0   = d["w0"]   if "w0"   in d else w[:, 0, :]
    print(f"  rho0: [{rho0.min():.4f}, {rho0.max():.4f}]")
    print(f"  w0  : [{w0.min():.4f},  {w0.max():.4f}]")
    if "v0" in d:
        v0 = d["v0"]
        print(f"  v0  : [{v0.min():.4f},  {v0.max():.4f}]")

    # Consistency: does rho0 match rho[:,0,:]?
    rho_t0 = rho[:, 0, :]
    ic_err = float(np.mean(np.abs(rho0 - rho_t0)))
    print(f"  |rho0 - rho[:,0,:]| MAE = {ic_err:.3e}  {'OK' if ic_err < 1e-5 else 'WARNING: mismatch!'}")


if __name__ == "__main__":
    main()
