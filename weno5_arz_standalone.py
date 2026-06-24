"""Standalone WENO5 + SSP-RK3 solver for the homogeneous/relaxation ARZ system.

Scheme: component-wise WENO5 (Jiang-Shu, classical smoothness indicators) with
global Lax-Friedrichs flux splitting on (rho, y), SSP-RK3 in time, and an
exact Strang-split relaxation step for finite tau.

"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Pressure closure  (single point of control)
# --------------------------------------------------------------------------- #
RHO_MIN = 1e-6                       # vacuum guard inside y = rho*w products
_P_FORM = "rho"                      # project default is ALWAYS "rho"; also "rho+rho2", ignore it tbh
_VALID_P_FORMS = ("rho", "rho+rho2")


def set_pressure_form(form: str) -> None:
    """Globally switch the pressure closure used by pressure/dpressure/y_eq."""
    global _P_FORM
    if form not in _VALID_P_FORMS:
        raise ValueError(f"Unknown pressure_form={form!r}; expected {_VALID_P_FORMS}.")
    _P_FORM = form


def get_pressure_form() -> str:
    return _P_FORM


def pressure(rho):
    """p(rho)."""
    if _P_FORM == "rho":
        return rho
    return rho + rho * rho


def dpressure(rho):
    """p'(rho), shape-matched to rho."""
    if _P_FORM == "rho":
        return np.ones_like(np.asarray(rho, dtype=float))
    return 1.0 + 2.0 * rho


def y_eq(rho):
    """y_eq(rho) = rho * w_eq(rho), with V(rho) = 1 - rho (Greenshields)."""
    if _P_FORM == "rho":
        return rho                       # = rho * 1
    return rho + rho * rho * rho


def spectral_radius(rho, w):
    """max(|lambda1|, |lambda2|): lambda1 = v - rho*p'(rho), lambda2 = v."""
    v = w - pressure(rho)
    lam1 = v - rho * dpressure(rho)
    lam2 = v
    return np.maximum(np.abs(lam1), np.abs(lam2))


# --------------------------------------------------------------------------- #
# WENO5 reconstruction kernel + ghost boundary
# --------------------------------------------------------------------------- #
_WENO_EPS = 1e-6


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


def _weno5_reconstruct_left(v: np.ndarray) -> np.ndarray:
    """Fifth-order WENO reconstruction of the value at the RIGHT interface
    of each cell from the LEFT-biased stencil.

    Input  v : [nx + 4] cell-centred values (2 ghosts each side).
    Output   : [nx + 1] reconstructed values at interfaces i+1/2.

    Jiang-Shu WENO5 with classical smoothness indicators.
    """
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


# --------------------------------------------------------------------------- #
# ARZ component-wise WENO5 + SSP-RK3
# --------------------------------------------------------------------------- #
def _weno5_arz_rhs(rho, y, dx, boundary):
    """Component-wise WENO5 + global Lax-Friedrichs flux on (rho, y)."""
    rho_safe = np.maximum(rho, RHO_MIN)
    w = y / rho_safe
    v = w - pressure(rho)
    lam1 = v - rho * dpressure(rho)
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
        fhat_p = _weno5_reconstruct_left(fp[0:-1])
        fhat_m = _weno5_reconstruct_left(fm[1:][::-1])[::-1]
        fhat = fhat_p + fhat_m
        return -(fhat[1:] - fhat[:-1]) / dx

    return _component_rhs(rho, F1), _component_rhs(y, F2)


def _weno5_arz_step(rho, y, dx, dt, boundary):
    """Single SSP-RK3 step on (rho, y)."""
    def L(rho_, y_):
        return _weno5_arz_rhs(rho_, y_, dx, boundary)

    Lr1, Ly1 = L(rho, y)
    rho1 = np.maximum(rho + dt * Lr1, RHO_MIN)
    y1   = y + dt * Ly1

    Lr2, Ly2 = L(rho1, y1)
    rho2 = np.maximum(0.75 * rho + 0.25 * (rho1 + dt * Lr2), RHO_MIN)
    y2   = 0.75 * y + 0.25 * (y1 + dt * Ly2)

    Lr3, Ly3 = L(rho2, y2)
    rho_n = np.maximum((1.0/3.0) * rho + (2.0/3.0) * (rho2 + dt * Lr3), RHO_MIN)
    y_n   = (1.0/3.0) * y + (2.0/3.0) * (y2 + dt * Ly3)
    return rho_n, y_n


def solve_arz_weno5(
    rho0, w0, x_min: float, x_max: float, t_max: float,
    nt_out: int, tau: float, cfl: float = 0.2, boundary: str = "ghost",
) -> Tuple[np.ndarray, np.ndarray]:
    """Strang-split WENO5 + SSP-RK3 on (rho, y) with exact relaxation.

    Parameters
    ----------
    rho0, w0 : (nx,) initial density and w = v + p(rho).
    x_min, x_max, t_max : domain extents.
    nt_out : number of output time snapshots (including t=0).
    tau : relaxation time; pass np.inf for the homogeneous (source-free) system.
    cfl : CFL number (WENO5 is stable to ~0.4; 0.2 is conservative).
    boundary : "ghost" (zero-gradient), "fixed", or "periodic".

    Returns (rho_hist, w_hist), each (nt_out, nx) float32.
    """
    rho0 = np.asarray(rho0, dtype=np.float64)
    w0   = np.asarray(w0,   dtype=np.float64)
    nx   = rho0.size
    dx   = (x_max - x_min) / nx
    t_out = np.linspace(0.0, t_max, nt_out, dtype=np.float64)

    rho = np.maximum(rho0.copy(), RHO_MIN)
    y   = rho * w0
    rho_hist = np.empty((nt_out, nx), dtype=np.float32)
    w_hist   = np.empty((nt_out, nx), dtype=np.float32)
    rho_hist[0] = rho.astype(np.float32)
    w_hist[0]   = (y / np.maximum(rho, RHO_MIN)).astype(np.float32)

    t = 0.0
    k = 1
    while k < nt_out:
        w_now   = y / np.maximum(rho, RHO_MIN)
        lam_max = max(float(spectral_radius(rho, w_now).max()), 1e-6)
        dt = min(cfl * dx / lam_max, t_out[k] - t)
        if dt <= 0:
            break
        # Strang: half-step source -> WENO hyperbolic -> half-step source.
        y_eq_ = y_eq(rho)
        y  = y_eq_ + (y - y_eq_) * np.exp(-0.5 * dt / tau)
        rho, y = _weno5_arz_step(rho, y, dx, dt, boundary)
        y_eq_ = y_eq(rho)
        y  = y_eq_ + (y - y_eq_) * np.exp(-0.5 * dt / tau)
        t += dt
        while k < nt_out and t >= t_out[k] - 1e-12:
            rho_hist[k] = rho.astype(np.float32)
            w_hist[k]   = (y / np.maximum(rho, RHO_MIN)).astype(np.float32)
            k += 1
    return rho_hist, w_hist


# --------------------------------------------------------------------------- #
# Demo: a single Riemann problem (homogeneous ARZ, tau = inf)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    set_pressure_form("rho+rho2")          # match the dataset closure

    nx, nt = 128, 128
    x_min, x_max, t_max = -1.0, 1.0, 1.0
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, nx)

    # Left/right primitive states (rho, v); w = v + p(rho).
    rhoL, vL = 0.1, 1.01
    rhoR, vR = 0.9, 0.05
    rho0 = np.where(x < 0.0, rhoL, rhoR)
    v0   = np.where(x < 0.0, vL,  vR)
    w0   = v0 + pressure(rho0)

    rho_h, w_h = solve_arz_weno5(
        rho0, w0, x_min, x_max, t_max, nt_out=nt, tau=np.inf, cfl=0.4,
    )
    print(f"pressure_form = {get_pressure_form()}")
    print(f"rho_hist shape = {rho_h.shape}, w_hist shape = {w_h.shape}")
    print(f"rho range: [{rho_h.min():.3f}, {rho_h.max():.3f}]")
    print(f"final-time rho[::16]: {np.round(rho_h[-1, ::16], 3)}")
