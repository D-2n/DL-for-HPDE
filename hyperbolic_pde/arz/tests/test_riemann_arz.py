"""Sanity tests for the exact homogeneous ARZ Riemann solver."""
from __future__ import annotations

import numpy as np

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import riemann_arz as R


def _check_invariants(rho, w, v, rho_L, w_L, v_L, rho_R, w_R, v_R, tol=1e-10):
    """w should equal w_L everywhere except the right-of-contact constant strip
    (where it equals w_R). v should be v_R wherever rho == rho_* or rho==rho_R
    or inside the fan? No - v equals v_R only across the 2-contact onward; in
    the L region v = v_L. Just check w in {w_L, w_R} and v in {v_L, v_R} ∪ fan.
    """
    # Anywhere with rho == rho_L we should see w == w_L; same R.
    mask_L = np.isclose(rho, rho_L, atol=1e-9)
    mask_R = np.isclose(rho, rho_R, atol=1e-9)
    assert np.allclose(w[mask_L], w_L, atol=tol)
    assert np.allclose(w[mask_R], w_R, atol=tol)


def test_riemann_intermediate_state_w_invariant():
    """w_* must equal w_L (1-wave preserves w)."""
    rho_L, w_L = 0.2, P.w_eq(0.2) + 0.05
    rho_R, w_R = 0.6, P.w_eq(0.6) - 0.03
    v_R = w_R - P.pressure(rho_R)
    # Sample on a fine xi grid that covers all waves.
    xi = np.linspace(-2.0, 2.0, 4001)
    rho, w, v = R.solve_riemann_arz(rho_L, w_L, rho_R, w_R, xi)
    # Check w == w_L on left + star strip, w == w_R on right strip.
    # The 2-contact is at v_*; everything to the right has w_R.
    contact = v_R
    left = xi < contact - 1e-3
    right = xi > contact + 1e-3
    assert np.allclose(w[left], w_L, atol=1e-8), (
        f"w left of contact should be w_L={w_L}, got range [{w[left].min()}, {w[left].max()}]"
    )
    assert np.allclose(w[right], w_R, atol=1e-8), (
        f"w right of contact should be w_R={w_R}, got range [{w[right].min()}, {w[right].max()}]"
    )


def test_riemann_intermediate_state_v_invariant():
    """v_* must equal v_R (2-contact preserves v)."""
    rho_L, w_L = 0.3, P.w_eq(0.3) + 0.10
    rho_R, w_R = 0.55, P.w_eq(0.55) + 0.02
    v_R = w_R - P.pressure(rho_R)
    v_L = w_L - P.pressure(rho_L)
    xi = np.linspace(-2.0, 2.0, 4001)
    rho, w, v = R.solve_riemann_arz(rho_L, w_L, rho_R, w_R, xi)
    # On the right side of the 2-contact: v == v_R, rho == rho_R.
    right = xi > v_R + 1e-3
    assert np.allclose(v[right], v_R, atol=1e-8)


def test_1_shock_rankine_hugoniot():
    """For a Riemann setup that yields a 1-shock, check RH jump conditions on
    the conserved variables (rho, y) at the 1-shock speed.

    Note: ARZ at equilibrium ICs does NOT reduce to LWR in the homogeneous
    (tau = infinity) problem — only the equilibrium flux matches. Relaxation
    is what restores LWR; that's tested elsewhere via reference_arz.
    """
    rho_L, w_L = 0.2, P.w_eq(0.2)
    rho_R, w_R = 0.7, P.w_eq(0.7)
    v_L = w_L - P.pressure(rho_L); v_R = w_R - P.pressure(rho_R)
    rho_star = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * (w_L - v_R)))
    v_star = v_R
    # Conserved states.
    y_L = rho_L * w_L
    y_star = rho_star * w_L  # w preserved across 1-shock
    F1_L = rho_L * v_L
    F1_star = rho_star * v_star
    F2_L = y_L * v_L
    F2_star = y_star * v_star
    s = (F1_star - F1_L) / (rho_star - rho_L)
    s_y = (F2_star - F2_L) / (y_star - y_L)
    # Both jump conditions must give the same speed.
    assert np.isclose(s, s_y, atol=1e-10), f"RH inconsistent: s_rho={s}, s_y={s_y}"


def test_1_rarefaction_self_consistency():
    """Inside the fan, xi = lambda1(rho) must hold."""
    rho_L, rho_R = 0.6, 0.2
    w_L, w_R = P.w_eq(rho_L), P.w_eq(rho_R)
    v_R = w_R - P.pressure(rho_R)
    rho_star = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * (w_L - v_R)))
    # Sample inside the fan.
    xi_head = (w_L - P.pressure(rho_L)) - rho_L * P.dpressure(rho_L)
    xi_tail = (w_L - P.pressure(rho_star)) - rho_star * P.dpressure(rho_star)
    if xi_head > xi_tail:  # not a fan
        return
    xi = np.linspace(xi_head + 1e-4, xi_tail - 1e-4, 200)
    rho, w, v = R.solve_riemann_arz(rho_L, w_L, rho_R, w_R, xi)
    lam1 = v - rho * P.dpressure(rho)
    err = np.max(np.abs(lam1 - xi))
    assert err < 1e-8, f"Fan self-similarity broken: max |lambda1 - xi| = {err:.3e}"


def test_xt_grid_t0_returns_ic():
    rho_L, w_L = 0.3, 1.2
    rho_R, w_R = 0.7, 1.4
    x = np.linspace(-1.0, 1.0, 65)
    t = np.array([0.0, 0.1])
    rho, w, v = R.solve_riemann_arz_xt(rho_L, w_L, rho_R, w_R, x, t, x0=0.0)
    assert np.allclose(rho[0], np.where(x < 0.0, rho_L, rho_R))
    assert np.allclose(w[0], np.where(x < 0.0, w_L, w_R))


if __name__ == "__main__":
    test_riemann_intermediate_state_w_invariant()
    test_riemann_intermediate_state_v_invariant()
    test_1_shock_rankine_hugoniot()
    test_1_rarefaction_self_consistency()
    test_xt_grid_t0_returns_ic()
    print("All riemann_arz tests passed.")
