from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


def flux(u: np.ndarray) -> np.ndarray:
    """Flux f(u) = u * (1 - u)."""
    return u * (1.0 - u)


def flux_prime(u: np.ndarray) -> np.ndarray:
    """Derivative f'(u) = 1 - 2u."""
    return 1.0 - 2.0 * u


def godunov_flux(u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
    """
    Godunov flux for f(u)=u(1-u) LWR traffic pde.

    """
    f_left = flux(u_left)
    f_right = flux(u_right)
    f_min = np.minimum(f_left, f_right)
    f_max = np.maximum(f_left, f_right)

    u_lo = np.minimum(u_left, u_right)
    u_hi = np.maximum(u_left, u_right)
    has_mid = (u_lo <= 0.5) & (0.5 <= u_hi)
    if np.any(has_mid):
        f_max = np.where(has_mid, np.maximum(f_max, 0.25), f_max)

    return np.where(u_left <= u_right, f_min, f_max)


def piecewise_constant_ic(
    x: np.ndarray,
    num_segments: int,
    u_min: float,
    u_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a piecewise-constant initial condition on grid x."""
    nx = x.size
    if num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    if num_segments == 1:
        val = rng.uniform(u_min, u_max)
        return np.full(nx, val, dtype=np.float32)

    cut_points = rng.choice(np.arange(1, nx), size=num_segments - 1, replace=False)
    cut_points.sort()
    cut_points = np.concatenate(([0], cut_points, [nx]))
    values = rng.uniform(u_min, u_max, size=num_segments)

    u0 = np.empty(nx, dtype=np.float32)
    for i in range(num_segments):
        start, end = cut_points[i], cut_points[i + 1]
        u0[start:end] = values[i]
    return u0


def piecewise_sine_ic(
    x: np.ndarray,
    num_segments: int,
    u_min: float,
    u_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Piecewise-sine IC: each segment gets a sine with random freq/phase."""
    nx = x.size
    num_segments = max(1, num_segments)
    cut_points = np.linspace(0, nx, num_segments + 1, dtype=int)
    u0 = np.empty(nx, dtype=np.float32)
    for i in range(num_segments):
        s, e = cut_points[i], cut_points[i + 1]
        if s >= e:
            continue
        freq = rng.uniform(0.5, 4.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        xs = x[s:e]
        raw = np.sin(freq * np.pi * (xs - xs[0]) / max(xs[-1] - xs[0], 1e-8) + phase)
        u0[s:e] = (u_min + u_max) / 2 + (u_max - u_min) / 2 * raw
    return u0.astype(np.float32)


def gaussian_mixture_ic(
    x: np.ndarray,
    num_segments: int,
    u_min: float,
    u_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sum of Gaussians IC: smooth bumps that steepen into shocks."""
    n_bumps = max(1, num_segments)
    centers = rng.uniform(x[0], x[-1], size=n_bumps)
    widths = rng.uniform(0.05, 0.3, size=n_bumps)
    heights = rng.uniform(0.3, 1.0, size=n_bumps)
    u0 = np.zeros_like(x, dtype=np.float64)
    for c, w, h in zip(centers, widths, heights):
        u0 += h * np.exp(-0.5 * ((x - c) / w) ** 2)
    # normalise to [u_min, u_max]
    u0_min, u0_max = u0.min(), u0.max()
    if u0_max - u0_min < 1e-8:
        u0[:] = (u_min + u_max) / 2
    else:
        u0 = u_min + (u_max - u_min) * (u0 - u0_min) / (u0_max - u0_min)
    return u0.astype(np.float32)


def riemann_ic(
    x: np.ndarray,
    num_segments: int,
    u_min: float,
    u_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Classic Riemann problem: single discontinuity at a random location."""
    x0 = rng.uniform(x[0] + 0.2 * (x[-1] - x[0]), x[0] + 0.8 * (x[-1] - x[0]))
    u_left = rng.uniform(u_min, u_max)
    u_right = rng.uniform(u_min, u_max)
    u0 = np.where(x <= x0, u_left, u_right).astype(np.float32)
    return u0


# Registry of IC generators (name -> function)
IC_REGISTRY: dict[str, callable] = {
    "piecewise_constant": piecewise_constant_ic,
    "piecewise_sine": piecewise_sine_ic,
    "gaussian_mixture": gaussian_mixture_ic,
    "riemann": riemann_ic,
}


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod slope limiter."""
    return np.where(
        a * b > 0,
        np.where(np.abs(a) < np.abs(b), a, b),
        0.0,
    )


def _apply_ghost_bc(u: np.ndarray, boundary: str, ng: int) -> np.ndarray:
    """Pad u with ng ghost cells on each side according to boundary type."""
    nx = u.size
    u_ext = np.empty(nx + 2 * ng, dtype=u.dtype)
    u_ext[ng:ng + nx] = u
    if boundary == "periodic":
        u_ext[:ng] = u[-ng:]
        u_ext[ng + nx:] = u[:ng]
    elif boundary in ("ghost", "fixed"):
        for i in range(ng):
            u_ext[i] = u[0]
            u_ext[ng + nx + i] = u[-1]
    else:
        raise ValueError(f"boundary must be 'periodic', 'ghost', or 'fixed', got '{boundary}'")
    return u_ext


def _godunov_step(u: np.ndarray, dx: float, dt: float, boundary: str) -> np.ndarray:
    """First-order Godunov update."""
    nx = u.size
    u_ext = _apply_ghost_bc(u, boundary, 1)
    fhat = godunov_flux(u_ext[:-1], u_ext[1:])  # nx+1 interfaces
    u_new = u - (dt / dx) * (fhat[1:] - fhat[:-1])
    if boundary == "fixed":
        u_new[0] = u[0]
        u_new[-1] = u[-1]
    return u_new


def _muscl_step(u: np.ndarray, dx: float, dt: float, boundary: str) -> np.ndarray:
    """Second-order MUSCL-Hancock update with minmod limiter."""
    nx = u.size
    u_ext = _apply_ghost_bc(u, boundary, 2)

    # slopes with minmod limiter
    du_fwd = u_ext[2:] - u_ext[1:-1]       # [nx+2]
    du_bwd = u_ext[1:-1] - u_ext[:-2]       # [nx+2]
    slope = _minmod(du_fwd, du_bwd)          # [nx+2]

    # reconstructed values at cell interfaces (cells indexed 1..nx in u_ext with offset 2)
    # left state at right interface of cell i:  u_i + 0.5 * slope_i
    # right state at left interface of cell i:  u_i - 0.5 * slope_i
    # We need interfaces i-1/2 for i=0..nx, so nx+1 interfaces
    # u_L at interface i+1/2 = u_ext[i+2] + 0.5*slope[i+1]  (cell i maps to u_ext index i+2, slope index i+1)
    # u_R at interface i+1/2 = u_ext[i+3] - 0.5*slope[i+2]

    # Half-step predictor (Hancock step)
    # slope is indexed [0..nx+1] corresponding to u_ext[1..nx+2]
    u_half = u_ext[1:-1] - 0.5 * (dt / dx) * (flux(u_ext[1:-1] + 0.5 * slope) - flux(u_ext[1:-1] - 0.5 * slope))
    slope_half = _minmod(u_half[1:] - u_half[:-1], u_half[:-1] - np.roll(u_half, 1)[:-1])

    # Interface states: nx+1 interfaces for nx cells
    # interface i+1/2: left from cell i, right from cell i+1
    # cells in u_half are indexed 0..nx+1 (corresponding to u_ext[1..nx+2])
    # Our physical cells are u_half[1..nx] (indices 1 to nx)
    u_L = u_half[1:-1] + 0.5 * slope[1:-1]   # [nx] left states at right interface of each cell
    u_R = u_half[1:-1] - 0.5 * slope[1:-1]    # [nx] right states at left interface of each cell

    # interface i+1/2 has left=u_L[i], right=u_R[i+1] for i=0..nx-1
    # We need nx+1 interfaces (0..nx), pad with ghost values
    fhat = godunov_flux(
        np.concatenate([u_L[:1], u_L]),        # nx+1
        np.concatenate([u_R, u_R[-1:]]),       # nx+1
    )

    u_new = u - (dt / dx) * (fhat[1:] - fhat[:-1])
    if boundary == "fixed":
        u_new[0] = u[0]
        u_new[-1] = u[-1]
    return u_new


_WENO_EPS = 1e-6


def _weno5_reconstruct_left(v: np.ndarray) -> np.ndarray:
    """Fifth-order WENO reconstruction of the value at the RIGHT interface
    of each cell from the LEFT-biased stencil.

    Input  v : [nx + 4] cell-centred values (2 ghosts each side).
    Output   : [nx + 1] reconstructed values at interfaces i+1/2
               for i = -1 .. nx-1  (i.e. all nx+1 physical interfaces).

    Uses Jiang-Shu WENO5 with classical smoothness indicators.
    """
    # We need 5-cell stencils centred so that interface i+1/2 uses cells
    # {i-2, i-1, i, i+1, i+2}. With 2 ghosts, the first physical interface
    # (-1/2) uses cells {-3..1} which needs 3 left ghosts. So caller must
    # pad with 3 ghosts.
    vm2 = v[0:-4]
    vm1 = v[1:-3]
    v0  = v[2:-2]
    vp1 = v[3:-1]
    vp2 = v[4:]

    # Candidate reconstructions (ENO stencils)
    p0 = (1.0/3.0)*vm2 - (7.0/6.0)*vm1 + (11.0/6.0)*v0
    p1 = -(1.0/6.0)*vm1 + (5.0/6.0)*v0 + (1.0/3.0)*vp1
    p2 = (1.0/3.0)*v0 + (5.0/6.0)*vp1 - (1.0/6.0)*vp2

    # Smoothness indicators
    b0 = (13.0/12.0)*(vm2 - 2.0*vm1 + v0)**2 + 0.25*(vm2 - 4.0*vm1 + 3.0*v0)**2
    b1 = (13.0/12.0)*(vm1 - 2.0*v0 + vp1)**2 + 0.25*(vm1 - vp1)**2
    b2 = (13.0/12.0)*(v0 - 2.0*vp1 + vp2)**2 + 0.25*(3.0*v0 - 4.0*vp1 + vp2)**2

    # Linear weights for upwind (left-biased) reconstruction
    d0, d1, d2 = 0.1, 0.6, 0.3
    a0 = d0 / (_WENO_EPS + b0)**2
    a1 = d1 / (_WENO_EPS + b1)**2
    a2 = d2 / (_WENO_EPS + b2)**2
    asum = a0 + a1 + a2
    w0 = a0 / asum
    w1 = a1 / asum
    w2 = a2 / asum

    return w0*p0 + w1*p1 + w2*p2


def _weno5_rhs(u: np.ndarray, dx: float, boundary: str) -> np.ndarray:
    """Spatial operator L(u) = -(F_{i+1/2} - F_{i-1/2})/dx using WENO5 with
    global Lax-Friedrichs flux splitting.
    """
    nx = u.size
    ng = 3
    u_ext = _apply_ghost_bc(u, boundary, ng)          # [nx + 6]

    # Global LF dissipation coefficient
    alpha = float(np.max(np.abs(flux_prime(u_ext))))
    alpha = max(alpha, 1e-8)

    f_ext = flux(u_ext)
    fp = 0.5 * (f_ext + alpha * u_ext)                # [nx+6] — +char direction
    fm = 0.5 * (f_ext - alpha * u_ext)                # [nx+6] — -char direction

    # Interface i+1/2 for i = -1 .. nx-1  → nx+1 interfaces.
    # Physical cell i lives at u_ext[i+3] (3 ghosts).
    # Left-biased stencil for interface i+1/2: cells {i-2..i+2} → u_ext[i+1..i+5].
    #   Over all interfaces need u_ext[0..nx+4] (length nx+5).
    # Right-biased stencil: cells {i-1..i+3} → u_ext[i+2..i+6].
    #   Over all interfaces need u_ext[1..nx+5] (length nx+5).
    fp_s = fp[0:-1]                   # [nx+5]  for left-biased on f+
    fm_s = fm[1:][::-1]               # [nx+5]  reversed → reuse left-biased kernel

    fhat_p = _weno5_reconstruct_left(fp_s)             # [nx+1]
    fhat_m = _weno5_reconstruct_left(fm_s)[::-1]       # [nx+1]

    fhat = fhat_p + fhat_m                             # numerical flux at interfaces

    return -(fhat[1:] - fhat[:-1]) / dx


def _weno5_step(u: np.ndarray, dx: float, dt: float, boundary: str) -> np.ndarray:
    """Fifth-order WENO in space + SSP-RK3 in time."""
    L = _weno5_rhs
    u1 = u + dt * L(u, dx, boundary)
    u2 = 0.75 * u + 0.25 * (u1 + dt * L(u1, dx, boundary))
    u_new = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * L(u2, dx, boundary))
    if boundary == "fixed":
        u_new[0] = u[0]
        u_new[-1] = u[-1]
    return u_new


# --------------------------------------------------------------------------- #
# DG(1): piecewise-linear DG with local Lax-Friedrichs flux + SSP-RK3
# --------------------------------------------------------------------------- #
# State: flat array of length 2*nx.
#   dofs[0::2] = cell means  u_i^0   (coefficient for phi_0 = 1)
#   dofs[1::2] = cell slopes u_i^1   (coefficient for phi_1 = sqrt(3)*(2xi-1))
#
# Basis on reference cell [0,1] — orthonormal w.r.t. L2([0,1]):
#   phi_0(xi) = 1
#   phi_1(xi) = sqrt(3) * (2*xi - 1)       dphi_1/dxi = 2*sqrt(3)
#
# Mass matrix = identity (orthonormal basis), so M^{-1} = I.
#
# Weak form (after integration by parts on the volume term):
#   d u_i^k / dt = (1/dx) * [ sum_q w_q f(u_h(xi_q)) dphi_k/dxi
#                              - phi_k(1)*fhat_{i+1/2} + phi_k(0)*fhat_{i-1/2} ]
#
# Stability: DG(1) + SSP-RK3 requires CFL <= 1/3.  We enforce this by
# halving the incoming dt when necessary before starting the RK3 stages.
#
# Slope limiter: minmod TVB limiter applied after each RK3 stage to prevent
# spurious oscillations near shocks.
#
# 2-point Gauss quadrature on [0,1] (exact for degree <= 3):
#   xi_q = [0.5 - 1/(2*sqrt(3)),  0.5 + 1/(2*sqrt(3))],  w_q = [0.5, 0.5]

_DG_SQRT3 = np.sqrt(3.0)
_DG_XI_Q  = np.array([0.5 - 1.0 / (2.0 * _DG_SQRT3), 0.5 + 1.0 / (2.0 * _DG_SQRT3)])
_DG_W_Q   = np.array([0.5, 0.5])
# Basis evaluated at quadrature nodes [2 bases x 2 nodes]
_DG_PHI_Q = np.array([
    np.ones(2),
    _DG_SQRT3 * (2.0 * _DG_XI_Q - 1.0),
])
# Basis evaluated at cell boundaries: xi=0 (left), xi=1 (right)
_DG_PHI_L = np.array([1.0, -_DG_SQRT3])   # phi_k(0)
_DG_PHI_R = np.array([1.0,  _DG_SQRT3])   # phi_k(1)
# Stability limit for DG(1) + SSP-RK3
_DG_CFL_MAX = 1.0 / 3.0


def _dg_project(u_cell: np.ndarray) -> np.ndarray:
    """L2-project cell-mean array [nx] onto DG(1) dofs [2*nx] (zero slopes)."""
    dofs = np.zeros(2 * u_cell.size, dtype=np.float64)
    dofs[0::2] = u_cell
    return dofs


def _dg_cell_means(dofs: np.ndarray) -> np.ndarray:
    """Extract cell means from DG dofs."""
    return dofs[0::2].copy()


def _dg_minmod_limiter(dofs: np.ndarray) -> np.ndarray:
    """Apply minmod slope limiter to DG(1) dofs in-place.

    Replaces u_i^1 with minmod(u_i^1, u_i^0 - u_{i-1}^0, u_{i+1}^0 - u_i^0)
    where the minmod comparison uses the physical slope = u_i^1 / sqrt(3).
    """
    dofs = dofs.copy()
    u0 = dofs[0::2]   # cell means [nx]
    u1 = dofs[1::2]   # slope dofs [nx] — physical slope = u1/sqrt(3)

    # Forward and backward differences of cell means
    du_fwd = np.empty_like(u0)
    du_bwd = np.empty_like(u0)
    du_fwd[:-1] = u0[1:] - u0[:-1]
    du_fwd[-1]  = 0.0
    du_bwd[1:]  = u0[1:] - u0[:-1]
    du_bwd[0]   = 0.0

    # Physical slope from slope dof: tilde_u = u1 / sqrt(3)
    tilde_u = u1 / _DG_SQRT3

    # minmod of three values
    def _minmod3(a, b, c):
        s = np.sign(a)
        same_sign = (s == np.sign(b)) & (s == np.sign(c))
        return np.where(same_sign, s * np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c))), 0.0)

    limited = _minmod3(tilde_u, du_fwd, du_bwd)
    dofs[1::2] = limited * _DG_SQRT3
    return dofs


def _dg_rhs(dofs: np.ndarray, dx: float, boundary: str) -> np.ndarray:
    """DG(1) spatial RHS: d(dofs)/dt per the weak form with LF numerical flux.

    Returns array of shape [2*nx] — same layout as dofs.
    """
    u0 = dofs[0::2]   # cell means  [nx]
    u1 = dofs[1::2]   # cell slopes [nx]

    # ---- Volume integral (only affects slope dof k=1) ----
    # (1/dx) * sum_q w_q * f(u_h(xi_q)) * dphi_1/dxi
    u_q = u0[:, None] + u1[:, None] * _DG_PHI_Q[1][None, :]  # [nx, 2]
    f_q = flux(u_q)                                            # [nx, 2]
    vol_1 = np.sum(_DG_W_Q * f_q, axis=1) * (2.0 * _DG_SQRT3) / dx  # [nx]

    # ---- Interface states ----
    u_R = u0 + u1 * _DG_PHI_R[1]   # u_h at right edge (xi=1) of each cell [nx]
    u_L = u0 + u1 * _DG_PHI_L[1]   # u_h at left  edge (xi=0) of each cell [nx]

    if boundary == "periodic":
        uR_ext = np.concatenate([u_R, u_R[:1]])    # left  state at interface i+1/2 [nx+1]
        uL_ext = np.concatenate([u_L[-1:], u_L])   # right state at interface i+1/2 [nx+1]
    else:
        uR_ext = np.concatenate([u_R, u_R[-1:]])
        uL_ext = np.concatenate([u_L[:1], u_L])

    # Local Lax-Friedrichs numerical flux at each of nx+1 interfaces
    alpha = np.maximum(np.abs(flux_prime(uR_ext)), np.abs(flux_prime(uL_ext)))
    fhat  = 0.5 * (flux(uR_ext) + flux(uL_ext)) - 0.5 * alpha * (uL_ext - uR_ext)

    fhat_r = fhat[1:]    # fhat at right interface of cell i  [nx]
    fhat_l = fhat[:-1]   # fhat at left  interface of cell i  [nx]

    # ---- Surface integral: -(phi_k(1)*fhat_r - phi_k(0)*fhat_l) / dx ----
    surf_0 = (_DG_PHI_R[0] * fhat_r - _DG_PHI_L[0] * fhat_l) / dx
    surf_1 = (_DG_PHI_R[1] * fhat_r - _DG_PHI_L[1] * fhat_l) / dx

    rhs = np.empty_like(dofs)
    rhs[0::2] = -surf_0          # vol_0 = 0 (dphi_0/dxi = 0)
    rhs[1::2] = vol_1 - surf_1
    return rhs


def _dg_step(u: np.ndarray, dx: float, dt: float, boundary: str) -> np.ndarray:
    """DG(1) step with SSP-RK3 and minmod slope limiter.

    Takes cell means [nx], returns cell means [nx].
    Internally subdivides dt to satisfy the DG CFL <= 1/3 stability limit.
    """
    u = np.asarray(u, dtype=np.float64)
    amax = float(np.nanmax(np.abs(flux_prime(u))))
    amax = max(amax, 1e-6)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        return u.astype(np.float32)
    dt_stable = _DG_CFL_MAX * dx / amax
    n_sub = max(1, int(np.ceil(dt / dt_stable)))
    dt_sub = dt / n_sub

    dofs = _dg_project(u)
    for _ in range(n_sub):
        d1 = _dg_minmod_limiter(dofs + dt_sub * _dg_rhs(dofs, dx, boundary))
        d2 = _dg_minmod_limiter(0.75 * dofs + 0.25 * (d1 + dt_sub * _dg_rhs(d1, dx, boundary)))
        dofs = _dg_minmod_limiter((1.0/3.0) * dofs + (2.0/3.0) * (d2 + dt_sub * _dg_rhs(d2, dx, boundary)))
        if not np.isfinite(dofs).all():
            dofs = _dg_project(u)
            break

    u_new = _dg_cell_means(dofs)
    if boundary == "fixed":
        u_new[0] = u[0]
        u_new[-1] = u[-1]
    return u_new


def solve_conservation_fvm(
    u0: np.ndarray,
    x_min: float,
    x_max: float,
    t_max: float,
    nt_out: int,
    cfl: float = 0.4,
    boundary: str = "periodic",
    method: str = "godunov",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve u_t + f(u)_x = 0 for f(u)=u(1-u).

    method: "godunov" (1st-order), "muscl" (2nd-order MUSCL-Hancock+minmod),
            "weno5" (5th-order WENO + SSP-RK3, LF flux splitting), or
            "dg" (DG(1) piecewise-linear + local LF flux + SSP-RK3).
    boundary: "periodic", "fixed", or "ghost".
    """
    if method not in ("godunov", "muscl", "weno5", "dg"):
        raise ValueError(f"method must be 'godunov', 'muscl', 'weno5', or 'dg', got '{method}'")

    step_fn = {"godunov": _godunov_step, "muscl": _muscl_step, "weno5": _weno5_step, "dg": _dg_step}[method]

    nx = u0.size
    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    dx = x[1] - x[0]
    t_out = np.linspace(0.0, t_max, nt_out, dtype=np.float32)

    u = u0.astype(np.float64).copy()
    u_hist = np.zeros((nt_out, nx), dtype=np.float32)
    u_hist[0] = u.astype(np.float32)

    t = 0.0
    k = 1
    while k < nt_out:
        amax = float(np.max(np.abs(flux_prime(u))))
        amax = max(1e-6, amax)
        dt = cfl * dx / amax
        dt = min(dt, t_max - t)
        u_prev = u.copy()
        t_prev = t
        u = step_fn(u, dx, dt, boundary)
        t += dt

        while k < nt_out and t >= t_out[k] - 1e-12:
            alpha = (t_out[k] - t_prev) / (t - t_prev) if t > t_prev else 1.0
            u_hist[k] = (u_prev + alpha * (u - u_prev)).astype(np.float32)
            k += 1

    return x, t_out, u_hist


def encode_ic(u0: np.ndarray, x: np.ndarray, num_points: int) -> np.ndarray:
    """Sample u0 at fixed points to create a compact encoding."""
    if num_points <= 0:
        return np.empty((0,), dtype=np.float32)
    xs = np.linspace(x.min(), x.max(), num_points, dtype=np.float32)
    idx = np.clip(np.searchsorted(x, xs), 0, x.size - 1)
    return u0[idx].astype(np.float32)


@dataclass
class DatasetBundle:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    u0: np.ndarray
    ic: np.ndarray


def _solve_one_sample(
    index: int,
    u0: np.ndarray,
    x_min: float,
    x_max: float,
    t_max: float,
    nt_out: int,
    cfl: float,
    boundary: str,
    method: str = "godunov",
) -> Tuple[int, np.ndarray]:
    if method == "lax_hopf":
        from hyperbolic_pde.data.lax_hopf import solve_lax_hopf
        _, _, u_hist = solve_lax_hopf(
            u0=u0,
            x_min=x_min,
            x_max=x_max,
            t_max=t_max,
            nt_out=nt_out,
            boundary=boundary,
        )
    else:
        _, _, u_hist = solve_conservation_fvm(
            u0=u0,
            x_min=x_min,
            x_max=x_max,
            t_max=t_max,
            nt_out=nt_out,
            cfl=cfl,
            boundary=boundary,
            method=method,
        )
    return index, u_hist


def generate_dataset(
    num_samples: int,
    nx: int,
    nt: int,
    x_min: float,
    x_max: float,
    t_max: float,
    cfl: float,
    num_segments: int | tuple[int, ...] | list[int],
    u_min: float,
    u_max: float,
    ic_points: int,
    boundary: str,
    seed: int = 42,
    num_workers: int | None = None,
    ic_types: list[str] | None = None,
    method: str = "godunov",
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    t = np.linspace(0.0, t_max, nt, dtype=np.float32)
    u = np.zeros((num_samples, nt, nx), dtype=np.float32)
    u0_all = np.zeros((num_samples, nx), dtype=np.float32)
    ic_all = np.zeros((num_samples, ic_points), dtype=np.float32)

    if isinstance(num_segments, (list, tuple, np.ndarray)):
        if len(num_segments) == 0:
            raise ValueError("num_segments list must be non-empty")
        seg_choices = [int(s) for s in num_segments]
        if any(s < 1 for s in seg_choices):
            raise ValueError("num_segments values must be >= 1")
    else:
        seg_choices = None
        num_segments = int(num_segments)
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1")

    # resolve IC generators
    if ic_types is None:
        ic_types = ["piecewise_constant"]
    ic_funcs = []
    for name in ic_types:
        if name not in IC_REGISTRY:
            raise ValueError(
                f"Unknown IC type '{name}'. Available: {list(IC_REGISTRY.keys())}"
            )
        ic_funcs.append(IC_REGISTRY[name])

    for i in range(num_samples):
        seg = int(rng.choice(seg_choices)) if seg_choices is not None else num_segments
        ic_fn = ic_funcs[rng.integers(len(ic_funcs))]
        u0 = ic_fn(x, seg, u_min, u_max, rng)
        u0_all[i] = u0
        ic_all[i] = encode_ic(u0, x, ic_points)

    use_lax_hopf = method == "lax_hopf"
    if use_lax_hopf:
        from hyperbolic_pde.data.lax_hopf import solve_lax_hopf

    log_every = 500
    print(f"Generating {num_samples} samples (nx={nx}, nt={nt}, method={method}) ...")
    if not num_workers or num_workers <= 1:
        for i in range(num_samples):
            if use_lax_hopf:
                _, _, u_hist = solve_lax_hopf(
                    u0=u0_all[i],
                    x_min=x_min,
                    x_max=x_max,
                    t_max=t_max,
                    nt_out=nt,
                    boundary=boundary,
                )
            else:
                _, _, u_hist = solve_conservation_fvm(
                    u0=u0_all[i],
                    x_min=x_min,
                    x_max=x_max,
                    t_max=t_max,
                    nt_out=nt,
                    cfl=cfl,
                    boundary=boundary,
                    method=method,
                )
            u[i] = u_hist
            if (i + 1) % log_every == 0 or (i + 1) == num_samples:
                print(f"  {i + 1}/{num_samples} samples done", flush=True)
        return DatasetBundle(x=x, t=t, u=u, u0=u0_all, ic=ic_all)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _solve_one_sample,
                i,
                u0_all[i],
                x_min,
                x_max,
                t_max,
                nt,
                cfl,
                boundary,
                method,
            )
            for i in range(num_samples)
        ]
        completed = 0
        for future in as_completed(futures):
            index, u_hist = future.result()
            u[index] = u_hist
            completed += 1
            if completed % log_every == 0 or completed == num_samples:
                print(f"  {completed}/{num_samples} samples done", flush=True)

    return DatasetBundle(x=x, t=t, u=u, u0=u0_all, ic=ic_all)


def save_dataset(bundle: DatasetBundle, path: Path) -> None:
    from hyperbolic_pde import resolve_data_path
    path = resolve_data_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=bundle.x,
        t=bundle.t,
        u=bundle.u,
        u0=bundle.u0,
        ic=bundle.ic,
    )


def load_dataset(path: Path) -> DatasetBundle:
    from hyperbolic_pde import resolve_data_path
    path = resolve_data_path(path)
    data = np.load(path)
    return DatasetBundle(
        x=data["x"],
        t=data["t"],
        u=data["u"],
        u0=data["u0"],
        ic=data["ic"],
    )
