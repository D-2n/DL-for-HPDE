"""Exact homogeneous ARZ Riemann solver (tau = infinity).

Solution structure (plan §3a):

    1-wave (GNL, on lambda1)   followed by   2-contact (LD, on lambda2 = v)

Riemann invariants:
    * w is constant across the 1-wave.
    * v is constant across the 2-contact.

So with left state (rho_L, w_L) and right state (rho_R, w_R), and writing
v_L = w_L - p(rho_L), v_R = w_R - p(rho_R):

    w_* = w_L            (preserved across 1-wave)
    v_* = v_R            (preserved across 2-contact)
    => p(rho_*) = w_L - v_R
    With p(rho) = rho + rho^2, solving rho^2 + rho - (w_L - v_R) = 0:
        rho_* = ( -1 + sqrt(1 + 4*(w_L - v_R)) ) / 2

Wave types (concave-ish GNL: lambda1 = v - rho - 2*rho^2 strictly decreasing in rho):
    * 1-shock      if  rho_L  <  rho_*    (lambda1_L > lambda1_*)
    * 1-rarefaction if rho_L  >  rho_*

The 2-contact has speed v_* (single LD characteristic).

Rarefaction fan parametrisation. Along the fan w = w_L is constant, and
xi = x/t = lambda1(rho) = v - rho*(1+2*rho) = (w_L - p(rho)) - rho - 2*rho^2
                       = w_L - 2*rho - 3*rho^2. Solve for rho:
    rho(xi) = ( -1 + sqrt(1 + 3*(w_L - xi)) ) / 3

1-shock speed via Rankine-Hugoniot (using y = rho*w, with w preserved):
    s = (rho_* * v_* - rho_L * v_L) / (rho_* - rho_L)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from hyperbolic_pde.arz import physics_arz as P


def _rho_star_from_w_minus_v(target: np.ndarray | float) -> np.ndarray | float:
    """Solve p(rho) = target for the active pressure closure.

    Returns the >= 0 root. `target` is clamped at 0 for numerical safety.

        p(rho) = rho            ->  rho_* = target
        p(rho) = rho + rho^2    ->  rho_* = (-1 + sqrt(1 + 4*target)) / 2
    """
    target = np.maximum(np.asarray(target, dtype=np.float64), 0.0)
    if P.get_pressure_form() == "rho":
        return target
    return 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * target))


def _rho_in_fan(xi: np.ndarray, w_L: float) -> np.ndarray:
    """Density profile inside the 1-rarefaction fan along x/t = xi.

    Across the fan, w is constant (= w_L) and xi = lambda1(rho) = v - rho*p'(rho).

        p(rho) = rho           : xi = w_L - 2*rho                =>  rho = (w_L - xi)/2
        p(rho) = rho + rho^2   : xi = w_L - 2*rho - 3*rho^2      =>  quadratic root below
    """
    if P.get_pressure_form() == "rho":
        return np.maximum(0.5 * (w_L - xi), 0.0)
    disc = np.maximum(1.0 + 3.0 * (w_L - xi), 0.0)
    return (-1.0 + np.sqrt(disc)) / 3.0


def solve_riemann_arz(
    rho_L: float, w_L: float,
    rho_R: float, w_R: float,
    xi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Self-similar exact solution sampled along xi = x/t.

    Parameters
    ----------
    rho_L, w_L, rho_R, w_R : floats
        Left/right primitive states.
    xi : array
        Similarity coordinates x/t at which to sample the solution.

    Returns
    -------
    rho, w, v : arrays, shapes match xi.
    """
    rho_L = float(rho_L)
    rho_R = float(rho_R)
    w_L = float(w_L)
    w_R = float(w_R)

    v_L = w_L - P.pressure(rho_L)
    v_R = w_R - P.pressure(rho_R)

    # Intermediate state: w_* = w_L, v_* = v_R, rho_* solves p(rho_*) = w_L - v_R.
    rho_star = float(_rho_star_from_w_minus_v(w_L - v_R))
    w_star = w_L
    v_star = v_R

    # 1-wave classification.
    lam1_L = v_L - rho_L * P.dpressure(rho_L)
    lam1_star = v_star - rho_star * P.dpressure(rho_star)

    xi = np.asarray(xi, dtype=np.float64)
    rho_out = np.empty_like(xi)
    w_out = np.empty_like(xi)
    v_out = np.empty_like(xi)

    is_shock = lam1_L > lam1_star + 1e-12  # strict Lax: lam1 decreases L -> *
    is_rare  = lam1_L < lam1_star - 1e-12
    # exact equality (degenerate, e.g. rho_L == rho_*): treat as zero-width fan.

    # 2-contact moves at v_star.
    contact_speed = v_star

    if is_shock:
        # 1-shock speed via RH on rho (w preserved across the wave).
        denom = (rho_star - rho_L)
        if abs(denom) < 1e-14:
            shock_speed = lam1_L
        else:
            shock_speed = (rho_star * v_star - rho_L * v_L) / denom

        left_mask = xi < shock_speed
        star_mask = (xi >= shock_speed) & (xi < contact_speed)
        right_mask = xi >= contact_speed

        rho_out[left_mask] = rho_L; w_out[left_mask] = w_L; v_out[left_mask] = v_L
        rho_out[star_mask] = rho_star; w_out[star_mask] = w_star; v_out[star_mask] = v_star
        rho_out[right_mask] = rho_R; w_out[right_mask] = w_R; v_out[right_mask] = v_R

    elif is_rare:
        # Rarefaction fan: lambda1 monotone (decreasing in rho), so xi sweeps from
        # lam1_L (head, faster) to lam1_star (tail, slower). With lam1_L > lam1_star
        # impossible here (rare => lam1_L < lam1_star). Head/tail:
        head = lam1_L           # smaller xi
        tail = lam1_star        # larger xi

        left_mask = xi < head
        fan_mask = (xi >= head) & (xi < tail)
        star_mask = (xi >= tail) & (xi < contact_speed)
        right_mask = xi >= contact_speed

        rho_out[left_mask] = rho_L; w_out[left_mask] = w_L; v_out[left_mask] = v_L

        rho_fan = _rho_in_fan(xi[fan_mask], w_L)
        rho_out[fan_mask] = rho_fan
        w_out[fan_mask] = w_L                              # w preserved in fan
        v_out[fan_mask] = w_L - P.pressure(rho_fan)

        rho_out[star_mask] = rho_star; w_out[star_mask] = w_star; v_out[star_mask] = v_star
        rho_out[right_mask] = rho_R; w_out[right_mask] = w_R; v_out[right_mask] = v_R

    else:
        # Degenerate: no 1-wave (rho_L == rho_*). Only the 2-contact remains.
        left_mask = xi < contact_speed
        right_mask = xi >= contact_speed
        rho_out[left_mask] = rho_L; w_out[left_mask] = w_L; v_out[left_mask] = v_L
        rho_out[right_mask] = rho_R; w_out[right_mask] = w_R; v_out[right_mask] = v_R

    return rho_out, w_out, v_out


def solve_riemann_arz_xt(
    rho_L: float, w_L: float,
    rho_R: float, w_R: float,
    x: np.ndarray, t: np.ndarray, x0: float = 0.0,
    t_eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the exact Riemann solution on an (x, t) grid.

    Returns rho, w, v of shape (nt, nx). At t == 0 the IC is returned (no
    division by zero); for t > 0 the self-similar profile xi = (x - x0)/t
    is sampled.
    """
    nx = x.size
    nt = t.size
    rho = np.empty((nt, nx), dtype=np.float64)
    w = np.empty((nt, nx), dtype=np.float64)
    v = np.empty((nt, nx), dtype=np.float64)
    for k, tk in enumerate(t):
        if tk <= t_eps:
            rho[k] = np.where(x < x0, rho_L, rho_R)
            w[k] = np.where(x < x0, w_L, w_R)
            v[k] = w[k] - P.pressure(rho[k])
        else:
            xi = (x - x0) / float(tk)
            rho[k], w[k], v[k] = solve_riemann_arz(rho_L, w_L, rho_R, w_R, xi)
    return rho, w, v
