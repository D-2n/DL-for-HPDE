"""Sanity checks for hyperbolic_pde/data/fvm_2d.py.

Two checks:

(a) Rectangle IC relaxation: a high-density rectangle on a low-density
    background. Expected:
      - solution stays in [0, 1]
      - mass is conserved (periodic BC)
      - characteristic structure shows the rect's right/top edges roughly
        stationary (shock with s ~ 0) and the left/bottom edges fan out
        (rarefaction).
    Saves a PNG with snapshots at several times.

(b) Smooth-IC convergence rate. Initialize u0 = 0.5 + 0.1*sin(2*pi*x)*sin(2*pi*y)
    on grids 16, 32, 64, 128. Run to a short time T_check (before shock
    formation). Refine the WENO5 grid by 4x (256) and treat that as the
    reference. Compute L2 error against the reference (resampled by
    block-averaging). Report observed convergence rates: log2(err_h /
    err_{h/2}). Expected: close to 5 for the well-resolved end (16->32 or
    32->64), tapering as the shock starts to form / resolution saturates.

Run from repo root with:
    python hyperbolic_pde/scripts/sanity_fvm_2d.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Anaconda ships duplicate libiomp5md.dll across numpy and matplotlib backends;
# work around the OMP #15 error since this script does no parallel work.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hyperbolic_pde.data.fvm_2d import simulate_lwr_2d


OUT_DIR = ROOT / "hyperbolic_pde" / "scripts" / "_sanity_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# (a) rectangle relaxation
# --------------------------------------------------------------------------- #
def check_rectangle_relaxation() -> None:
    print("\n=== (a) Rectangle relaxation ===")
    nx = ny = 64
    nt = 6
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    t = np.linspace(0.0, 0.3, nt, dtype=np.float32)

    u0 = np.full((ny, nx), 0.2, dtype=np.float32)
    u0[ny // 4 : 3 * ny // 4, nx // 4 : 3 * nx // 4] = 0.8  # central square

    u_hist = simulate_lwr_2d(u0, x, y, t, cfl=0.4, boundary="periodic")

    dxdy = (1.0 / nx) * (1.0 / ny)
    mass_0 = u_hist[0].sum() * dxdy
    mass_T = u_hist[-1].sum() * dxdy
    print(f"  shape          : {u_hist.shape}")
    print(f"  u range t=0    : [{u_hist[0].min():.4f}, {u_hist[0].max():.4f}]")
    print(f"  u range t=end  : [{u_hist[-1].min():.4f}, {u_hist[-1].max():.4f}]")
    print(f"  mass at t=0    : {mass_0:.6f}")
    print(f"  mass at t=end  : {mass_T:.6f}")
    print(f"  relative drift : {(mass_T - mass_0) / mass_0:.2e}")
    in_range = (u_hist.min() >= -1e-3) and (u_hist.max() <= 1.0 + 1e-3)
    print(f"  values in [0,1]: {in_range}")

    fig, axes = plt.subplots(1, nt, figsize=(3 * nt, 3))
    for k in range(nt):
        ax = axes[k]
        ax.imshow(
            u_hist[k], origin="lower", extent=(0, 1, 0, 1),
            vmin=0.0, vmax=1.0, cmap="viridis",
        )
        ax.set_title(f"t={t[k]:.3f}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Rectangle relaxation (WENO5 + Strang + RK3, periodic)")
    fig.tight_layout()
    out = OUT_DIR / "rect_relaxation.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved: {out}")


# --------------------------------------------------------------------------- #
# (b) convergence rate on a smooth IC
# --------------------------------------------------------------------------- #
def _smooth_ic(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Periodic-friendly smooth IC: u0 = 0.5 + 0.1*sin(2 pi x)*sin(2 pi y)
    x = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, endpoint=False, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    u0 = (0.5 + 0.1 * np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)).astype(np.float32)
    return x, y, u0


def _block_average(u: np.ndarray, factor: int) -> np.ndarray:
    """Average a [ny, nx] field over factor x factor blocks (factor must divide both)."""
    ny, nx = u.shape
    assert ny % factor == 0 and nx % factor == 0, "factor must divide ny and nx"
    return u.reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))


def check_convergence_rate() -> None:
    print("\n=== (b) Smooth-IC convergence rate ===")
    T_check = 0.05  # short, well before any shock can sharpen significantly
    nt = 2          # only need t=0 and t=T_check
    grids = [16, 32, 64, 128]
    ref_n = 256

    # Reference solution on the finest grid
    print(f"  computing reference on {ref_n}x{ref_n} ...")
    x_ref, y_ref, u0_ref = _smooth_ic(ref_n, ref_n)
    t_arr = np.linspace(0.0, T_check, nt, dtype=np.float32)
    u_ref = simulate_lwr_2d(u0_ref, x_ref, y_ref, t_arr, cfl=0.4, boundary="periodic")
    u_ref_T = u_ref[-1]  # [ref_n, ref_n]

    errs: list[float] = []
    for n in grids:
        x, y, u0 = _smooth_ic(n, n)
        u_hist = simulate_lwr_2d(u0, x, y, t_arr, cfl=0.4, boundary="periodic")
        u_T = u_hist[-1]
        # downsample reference to this grid by block-averaging
        factor = ref_n // n
        u_ref_down = _block_average(u_ref_T, factor)
        err = float(np.sqrt(np.mean((u_T - u_ref_down) ** 2)))
        errs.append(err)
        print(f"  n={n:4d}   L2 err = {err:.3e}")

    print("  observed rates (log2(err_h / err_{h/2})):")
    for i in range(len(grids) - 1):
        rate = np.log2(errs[i] / max(errs[i + 1], 1e-30))
        print(f"    {grids[i]:3d} -> {grids[i+1]:3d}:  {rate:.2f}")

    # Plot error vs h on log-log
    hs = [1.0 / n for n in grids]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(hs, errs, "o-", label="WENO5 + Strang + RK3")
    ref_line = errs[0] * (np.array(hs) / hs[0]) ** 5
    ax.loglog(hs, ref_line, "k--", label="slope 5 reference")
    ax.set_xlabel("h")
    ax.set_ylabel("L2 error vs 256^2 reference")
    ax.set_title(f"Smooth-IC convergence at T={T_check}")
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    out = OUT_DIR / "convergence.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    check_rectangle_relaxation()
    check_convergence_rate()
    print(f"\nAll outputs in: {OUT_DIR}")
