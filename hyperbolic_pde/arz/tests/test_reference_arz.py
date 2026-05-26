"""Sanity tests for the Strang-split ARZ FV reference solver."""
from __future__ import annotations

import numpy as np

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz import riemann_arz as Rie


def test_well_balanced_equilibrium():
    """An IC at equilibrium (w0 = w_eq(rho0)) and constant rho must stay
    at that equilibrium under the Strang-split scheme: rho unchanged,
    w unchanged. Also tests both the hyperbolic step (constant rho, v) and
    the source (already at y_eq -> S = 0).
    """
    nx = 64
    rho0 = np.full(nx, 0.4, dtype=np.float64)
    w0 = P.w_eq(rho0)
    x, t, rho, w = Ref.solve_arz_reference(
        rho0, w0, x_min=0.0, x_max=1.0, t_max=0.5, nt_out=11,
        tau=0.1, cfl=0.4, boundary="periodic", refine=1,
    )
    drho = np.max(np.abs(rho - 0.4))
    dw = np.max(np.abs(w - w0[0]))
    assert drho < 1e-6, f"rho drifted off equilibrium: max |drho|={drho:.3e}"
    assert dw < 1e-6, f"w drifted off equilibrium: max |dw|={dw:.3e}"


def test_mass_conservation_periodic():
    """Total rho-mass is exactly conserved (no source on rho, periodic BCs)."""
    rng = np.random.default_rng(0)
    nx = 64
    rho0 = 0.5 + 0.3 * np.sin(2 * np.pi * np.linspace(0, 1, nx))
    rho0 = np.clip(rho0, 1e-3, 1.0 - 1e-3)
    w0 = P.w_eq(rho0) + 0.05  # off-equilibrium
    x, t, rho, w = Ref.solve_arz_reference(
        rho0, w0, x_min=0.0, x_max=1.0, t_max=0.3, nt_out=6,
        tau=0.2, cfl=0.4, boundary="periodic", refine=1,
    )
    mass0 = rho[0].sum()
    drift = np.max(np.abs(rho.sum(axis=1) - mass0))
    assert drift < 1e-4, f"rho-mass drifted: {drift:.3e}"


def test_source_relaxes_y_to_equilibrium():
    """A constant-rho IC strongly off-equilibrium should relax y -> y_eq
    on the timescale tau."""
    nx = 16
    rho0 = np.full(nx, 0.3, dtype=np.float64)
    w0 = np.full(nx, P.w_eq(0.3) + 0.5, dtype=np.float64)  # strongly off-eq.
    tau = 0.05
    x, t, rho, w = Ref.solve_arz_reference(
        rho0, w0, x_min=0.0, x_max=1.0, t_max=1.0, nt_out=11,
        tau=tau, cfl=0.4, boundary="periodic", refine=1,
    )
    # After several tau, w should be close to w_eq(0.3).
    err = np.max(np.abs(w[-1] - P.w_eq(0.3)))
    assert err < 1e-3, f"y did not relax to equilibrium: max |w - w_eq|={err:.3e}"


def test_homogeneous_riemann_matches_exact():
    """With very large tau (homogeneous limit), the FV solver should match
    the exact Riemann solution well at a finite-resolution tolerance."""
    nx_train = 128
    x_min, x_max = -1.0, 1.0
    x = np.linspace(x_min, x_max, nx_train)
    rho_L, w_L = 0.2, P.w_eq(0.2) + 0.10  # off-eq
    rho_R, w_R = 0.6, P.w_eq(0.6) - 0.05
    rho0 = np.where(x < 0.0, rho_L, rho_R).astype(np.float64)
    w0 = np.where(x < 0.0, w_L, w_R).astype(np.float64)

    t_max = 0.3
    x_out, t, rho_fv, w_fv = Ref.solve_arz_reference(
        rho0, w0, x_min=x_min, x_max=x_max, t_max=t_max, nt_out=4,
        tau=1e6, cfl=0.4, boundary="ghost", refine=4,
    )
    # Compare at the final time.
    rho_ex, w_ex, _ = Rie.solve_riemann_arz_xt(
        rho_L, w_L, rho_R, w_R, x_out.astype(np.float64), np.array([t_max]), x0=0.0,
    )
    # L1 error (not L_inf — shocks smear at finite resolution).
    err_rho_L1 = float(np.mean(np.abs(rho_fv[-1] - rho_ex[0])))
    err_w_L1 = float(np.mean(np.abs(w_fv[-1] - w_ex[0])))
    assert err_rho_L1 < 0.05, f"L1 rho error too large vs exact: {err_rho_L1:.3e}"
    assert err_w_L1 < 0.10, f"L1 w error too large vs exact: {err_w_L1:.3e}"


if __name__ == "__main__":
    test_well_balanced_equilibrium()
    test_mass_conservation_periodic()
    test_source_relaxes_y_to_equilibrium()
    test_homogeneous_riemann_matches_exact()
    print("All reference_arz tests passed.")
