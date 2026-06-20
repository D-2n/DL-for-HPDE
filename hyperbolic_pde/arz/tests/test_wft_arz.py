"""Tests for ARZ wave-front tracking (machine-precision GT).

WFT must reproduce the exact self-similar Riemann solution:
  * EXACTLY (to machine eps) on pure shocks and pure contacts,
  * to O(rare_delta) inside rarefaction fans, converging linearly.
"""
from __future__ import annotations

import numpy as np
import pytest

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.riemann_arz import solve_riemann_arz_xt
from hyperbolic_pde.arz.wft_arz import solve_arz_wft_riemann_xt


@pytest.fixture(autouse=True)
def _pform():
    old = P.get_pressure_form()
    P.set_pressure_form("rho")
    yield
    P.set_pressure_form(old)


def _exact(rL, vL, rR, vR, x, t, x0):
    wL = float(vL + P.pressure(np.array(rL)))
    wR = float(vR + P.pressure(np.array(rR)))
    return solve_riemann_arz_xt(rL, wL, rR, wR, x, t, x0=x0)


def test_pure_contact_is_machine_exact():
    """v_L == v_R -> only the 2-contact; WFT must be exact."""
    rL, vL, rR, vR = 0.3, 0.5, 0.75, 0.5
    x = np.linspace(0.0, 1.0, 257); t = np.array([0.0, 0.15, 0.3])
    rE, wE, vE = _exact(rL, vL, rR, vR, x, t, 0.5)
    rW, wW, vW = solve_arz_wft_riemann_xt(rL, vL, rR, vR, x, t, x0=0.5, rare_delta=0.02)
    assert np.abs(rW - rE).max() < 1e-12
    assert np.abs(vW - vE).max() < 1e-12
    assert np.abs(wW - wE).max() < 1e-12


def test_shock_plus_contact_is_machine_exact():
    """1-shock + 2-contact (no rarefaction) -> exact to machine eps, except at
    most one cell may straddle a front position (a measure-zero, float-ambiguous
    point where the solution is genuinely discontinuous). So the L1 (mean) error
    is machine-precision, and at most ONE cell per field differs at all."""
    rL, vL, rR, vR = 0.30, 0.85, 0.70, 0.30
    x = np.linspace(0.0, 1.0, 257); t = np.array([0.0, 0.3])
    rE, wE, vE = _exact(rL, vL, rR, vR, x, t, 0.5)
    rW, wW, vW = solve_arz_wft_riemann_xt(rL, vL, rR, vR, x, t, x0=0.5, rare_delta=0.02)
    # At most one cell per output time may differ (a point sitting exactly on a
    # front); everywhere else WFT reproduces the exact solution bit-for-bit.
    assert int((np.abs(rW - rE) > 1e-12).sum()) <= t.size
    assert int((np.abs(vW - vE) > 1e-12).sum()) <= t.size
    # The non-front cells are exactly equal, so >= 99% of cells match to eps.
    frac_exact = float((np.abs(rW - rE) <= 1e-12).mean())
    assert frac_exact > 0.99


def test_rarefaction_converges_linearly_in_rare_delta():
    """Inside a 1-rarefaction the max error scales ~ rare_delta."""
    rL, vL, rR, vR = 0.75, 0.25, 0.25, 0.65   # rho decreases -> rarefaction
    x = np.linspace(0.0, 1.0, 513); t = np.array([0.3])
    rE, _, _ = _exact(rL, vL, rR, vR, x, t, 0.5)
    e_coarse = np.abs(
        solve_arz_wft_riemann_xt(rL, vL, rR, vR, x, t, x0=0.5, rare_delta=0.02)[0] - rE).max()
    e_fine = np.abs(
        solve_arz_wft_riemann_xt(rL, vL, rR, vR, x, t, x0=0.5, rare_delta=0.002)[0] - rE).max()
    # ~10x smaller rare_delta -> ~10x smaller error (allow generous slack).
    assert e_fine < 0.25 * e_coarse
    assert e_coarse < 0.05   # absolute sanity


def test_w_channel_exact_everywhere():
    """w is piecewise-constant (the 1-Riemann invariant) so WFT reproduces it
    exactly regardless of rarefactions."""
    rL, vL, rR, vR = 0.85, 0.10, 0.15, 0.90
    x = np.linspace(0.0, 1.0, 401); t = np.array([0.0, 0.2, 0.3])
    _, wE, _ = _exact(rL, vL, rR, vR, x, t, 0.5)
    _, wW, _ = solve_arz_wft_riemann_xt(rL, vL, rR, vR, x, t, x0=0.5, rare_delta=0.02)
    assert np.abs(wW - wE).max() < 1e-12


# --------------------------------------------------------------------------- #
# MULTI-SEGMENT verification: collisions, conservation, FV convergence.
# These are the checks single-Riemann tests can't make (no analytic GT exists
# for multi-segment data, so WFT IS the reference there -- it must therefore pass
# every internal-consistency law a correct solver satisfies).
# --------------------------------------------------------------------------- #
from hyperbolic_pde.arz import reference_arz as _Ref  # noqa: E402
from hyperbolic_pde.arz.wft_arz import (  # noqa: E402
    solve_arz_wft_xt, ArzFrontTracking)

# A multi-segment IC whose interior waves interact. (rho, v) per segment.
_MULTISEG_STATES = [(0.3, 0.8), (0.7, 0.3), (0.25, 0.7), (0.6, 0.2)]
_MULTISEG_BND = [0.3, 0.55, 0.78]


def _ic_cells(states, bnd, xm):
    si = np.searchsorted(bnd, xm, side="right")
    rho0 = np.array([states[int(k)][0] for k in si], dtype=float)
    v0 = np.array([states[int(k)][1] for k in si], dtype=float)
    return rho0, v0


def test_mass_conserved_via_flux_balance():
    """integral_window(rho dx) must change by exactly -(F1_right - F1_left)*dt
    (mass flux through the window edges); no spurious mass creation/loss. Tests
    WFT's conservation property directly, across the interior front collisions."""
    states, bnd = _MULTISEG_STATES, _MULTISEG_BND
    nx = 8000
    xm = np.linspace(0.0, 1.0, nx, endpoint=False) + 0.5 / nx
    dx = 1.0 / nx
    t = np.array([0.0, 0.05, 0.10, 0.15])   # short: no wave reaches the edges
    rW, wW, _ = solve_arz_wft_xt(states, bnd, xm, t, rare_delta=0.0005)
    vW = wW - P.pressure(rW)
    F1 = rW * vW
    mass = rW.sum(axis=1) * dx
    for k in range(1, t.size):
        dt = t[k] - t[0]
        flux_in = 0.5 * (F1[0, 0] + F1[k, 0])
        flux_out = 0.5 * (F1[0, -1] + F1[k, -1])
        predicted = mass[0] - (flux_out - flux_in) * dt
        assert abs(mass[k] - predicted) < 1e-9   # machine-precision conservation


def test_fv_schemes_converge_to_wft():
    """MUSCL and WENO5 error vs WFT must DECREASE monotonically as nx grows on a
    fixed multi-segment IC. If the finite-volume schemes converge TO WFT, WFT is
    validated as the correct vanishing-viscosity limit (the only way to check
    multi-segment data, which has no closed-form GT)."""
    states, bnd = _MULTISEG_STATES, _MULTISEG_BND
    tmax, nt = 0.3, 48
    t = np.linspace(0.0, tmax, nt)
    errs = []
    for nx in (128, 256, 512):
        xm = np.linspace(0.0, 1.0, nx, endpoint=False) + 0.5 / nx
        rW, wW, _ = solve_arz_wft_xt(states, bnd, xm, t, rare_delta=0.0005)
        rho0, v0 = _ic_cells(states, bnd, xm)
        w0 = v0 + P.pressure(rho0)
        rM, wM = _Ref.solve_arz_muscl(rho0, w0, 0.0, 1.0, tmax, nt,
                                      tau=1e9, cfl=0.4, boundary="ghost")
        errs.append(float(np.abs(rM[-1] - rW[-1]).mean()))
    # Strictly decreasing, and roughly halving (2nd order on a discontinuous
    # solution is ~1st-order in L1 -> error ~ O(dx)); require clear decrease.
    assert errs[1] < errs[0] and errs[2] < errs[1]
    assert errs[2] < 0.6 * errs[0]


def test_two_one_shocks_merge_into_one():
    """Two adjacent 1-shocks (w preserved, rho increasing L->M->R) where the left
    shock is faster must MERGE into a single 1-shock after they collide."""
    w_c = 1.2
    def st(rho):
        return (rho, float(w_c - P.pressure(np.array(rho))))
    states = [st(0.2), st(0.5), st(0.8)]   # rho increasing -> each pair a 1-shock
    solver = ArzFrontTracking([0.35, 0.6], states, rare_delta=0.005)
    # Both are 1-shocks at t=0.
    kinds0 = sorted(fr.kind for fr in solver.fronts)
    assert kinds0 == ["shock1", "shock1"]
    solver.simulate(0.8)
    # After collision: exactly one front, a single 1-shock between the outer
    # states (w preserved throughout, so no contact appears).
    assert len(solver.fronts) == 1
    fr = solver.fronts[0]
    assert fr.kind == "shock1"
    assert abs(fr.rho_l - 0.2) < 1e-9 and abs(fr.rho_r - 0.8) < 1e-9


def test_collision_front_count_stays_bounded():
    """Front count must not blow up under interactions: it should stay O(initial
    fan size) and terminate (no interaction storm). Guards against the runaway
    front-spawning failure mode of naive front tracking."""
    states, bnd = _MULTISEG_STATES, _MULTISEG_BND
    solver = ArzFrontTracking(bnd, states, rare_delta=0.02)
    n0 = len(solver.fronts)
    _segs, hist = solver.simulate(0.6)
    counts = [len(h[1]) for h in hist]
    # Boundedness: the count stays O(initial fan size) -- it MAY rise modestly
    # when a collision spawns a new rarefaction fan (legitimate, not a storm),
    # but must not run away. And the simulation must TERMINATE (finite events,
    # well under the max_events safety valve), proving no interaction storm.
    assert max(counts) <= 3 * n0 + 5
    assert len(hist) < 1000           # terminated in few events, no storm
    assert np.isfinite([fr.x for fr in solver.fronts]).all()
