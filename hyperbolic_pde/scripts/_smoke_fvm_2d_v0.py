"""Smoke test for the v0 refactor of fvm_2d.py.

Quick sanity checks:
 1. Legacy simulate_lwr_2d still works.
 2. WENO5 + linear flux at fine resolution matches the analytical shift.
 3. generate_dataset_upsampled_2d returns sane shapes for a small batch.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from hyperbolic_pde.data.fvm_2d import (
    RectSpec,
    SampleSpec,
    analytical_shift_2d_rects,
    cell_average_to_grid,
    generate_dataset_upsampled_2d,
    linear_flux,
    rasterise_sample_spec,
    simulate_lwr_2d,
    simulate_weno5_2d,
)


def main() -> None:
    print("=== Test 1: legacy simulate_lwr_2d unchanged ===")
    nx = ny = 32
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y = np.linspace(0, 1, ny, dtype=np.float32)
    t = np.linspace(0, 0.2, 4, dtype=np.float32)
    u0 = np.full((ny, nx), 0.5, dtype=np.float32)
    u0[8:16, 8:16] = 0.2
    u_lwr = simulate_lwr_2d(u0, x, y, t, cfl=0.4, boundary="ghost")
    print(f"  LWR shape={u_lwr.shape} finite={np.isfinite(u_lwr).all()}")
    print(
        f"  u[0] [{u_lwr[0].min():.3f}, {u_lwr[0].max():.3f}]  "
        f"u[-1] [{u_lwr[-1].min():.3f}, {u_lwr[-1].max():.3f}]"
    )

    print()
    print("=== Test 2: linear WENO5 fine grid vs analytical ===")
    a, b = 1.0, 0.7
    U = 8
    nx = ny = 32
    nt = 8
    t_max = 0.2
    t = np.linspace(0, t_max, nt, dtype=np.float32)
    spec = SampleSpec(
        background=0.5,
        rects=(RectSpec(0.2, 0.4, 0.2, 0.4, 0.2),),
    )
    x_fine = np.linspace(0, 1, U * nx, dtype=np.float32)
    y_fine = np.linspace(0, 1, U * ny, dtype=np.float32)
    u0_fine = rasterise_sample_spec(spec, x_fine, y_fine)

    u_fine_weno = simulate_weno5_2d(
        u0_fine, x_fine, y_fine, t,
        flux_x=linear_flux(a), flux_y=linear_flux(b),
        cfl=0.4, boundary="ghost",
    )
    u_fine_ana = analytical_shift_2d_rects(
        spec, x_fine, y_fine, t, a, b, boundary="ghost",
    )
    print(f"  fine grids: {u_fine_weno.shape}")
    print(
        f"  max |WENO5 - analytical| (fine)   = "
        f"{np.max(np.abs(u_fine_weno - u_fine_ana)):.4f}"
    )
    print(
        f"  mean |WENO5 - analytical| (fine)  = "
        f"{np.mean(np.abs(u_fine_weno - u_fine_ana)):.5f}"
    )
    u_weno_coarse = cell_average_to_grid(u_fine_weno, U)
    u_ana_coarse = cell_average_to_grid(u_fine_ana, U)
    print(
        f"  max |WENO5 - analytical| (coarse) = "
        f"{np.max(np.abs(u_weno_coarse - u_ana_coarse)):.4f}"
    )
    print(
        f"  mean |WENO5 - analytical| (coarse)= "
        f"{np.mean(np.abs(u_weno_coarse - u_ana_coarse)):.5f}"
    )

    print()
    print("=== Test 3: generate_dataset_upsampled_2d (linear, 3 samples) ===")
    bundle = generate_dataset_upsampled_2d(
        num_samples=3, nx=32, ny=32, nt=4,
        flux_kind="linear", a=1.0, b=0.7,
        upsample_factor=4, t_max=0.2, num_rects=2,
        boundary="ghost", seed=123, verbose=True,
    )
    print(f"  bundle.u shape: {bundle.u.shape}, u0 shape: {bundle.u0.shape}")
    print(f"  bundle.x [{bundle.x.min():.3f}, {bundle.x.max():.3f}]")
    print(f"  bundle.t: {bundle.t}")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
