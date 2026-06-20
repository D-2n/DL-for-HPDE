"""Finite-volume reference solver for 1D ARZ with relaxation (plan §3b).

Strang splitting per step:
    U <- ODE_source(dt/2)
    U <- HLL_hyperbolic(dt)
    U <- ODE_source(dt/2)

Hyperbolic flux (selectable via flux_scheme):
  * 'hll' (default): HLL on conserved (rho, y) with wave-speed estimates from
    lambda1, lambda2. Fast, but averages the linearly-degenerate 2-contact away
    (over-smears it). This is the dataset ground-truth flux -- do NOT change the
    default, or regenerated data shifts.
  * 'godunov': exact-Riemann flux (see _godunov_flux) -- solves the true ARZ
    Riemann problem per interface and samples at xi=0. Sharp contact; the system
    analogue of the LWR exact-Godunov flux. For baselines/eval, not datagen.

Source step: at fixed rho, dy/dt = (y_eq(rho) - y) / tau is linear, exactly
integrable over dt:
    y_new = y_eq + (y_old - y_eq) * exp(-dt / tau)

CFL: dt = cfl * dx / max_i max(|lambda1_i|, |lambda2_i|).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.data.fvm import _apply_ghost_bc, _weno5_reconstruct_left


# --------------------------------------------------------------------------- #
# Boundary conditions (ghost padding)
# --------------------------------------------------------------------------- #
def _pad_ghost(arr: np.ndarray, ng: int, boundary: str) -> np.ndarray:
    nx = arr.size
    out = np.empty(nx + 2 * ng, dtype=arr.dtype)
    out[ng:ng + nx] = arr
    if boundary == "periodic":
        out[:ng] = arr[-ng:]
        out[ng + nx:] = arr[:ng]
    elif boundary in ("ghost", "fixed"):
        out[:ng] = arr[0]
        out[ng + nx:] = arr[-1]
    else:
        raise ValueError(f"boundary must be 'periodic'|'ghost'|'fixed', got {boundary!r}")
    return out


# --------------------------------------------------------------------------- #
# HLL numerical flux
# --------------------------------------------------------------------------- #
def _hll_flux(
    rho_L: np.ndarray, y_L: np.ndarray,
    rho_R: np.ndarray, y_R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """HLL flux for ARZ on conserved (rho, y).

    Wave-speed estimates: S_L = min(lambda1_L, lambda1_R),
                          S_R = max(lambda2_L, lambda2_R).
    """
    # Recover primitive states.
    _, w_L, v_L = P.to_primitive(rho_L, y_L)
    _, w_R, v_R = P.to_primitive(rho_R, y_R)
    lam1_L = v_L - rho_L * P.dpressure(rho_L)
    lam2_L = v_L
    lam1_R = v_R - rho_R * P.dpressure(rho_R)
    lam2_R = v_R

    S_L = np.minimum(lam1_L, lam1_R)
    S_R = np.maximum(lam2_L, lam2_R)

    F1_L, F2_L = P.flux(rho_L, y_L)
    F1_R, F2_R = P.flux(rho_R, y_R)

    # HLL formula, per component.
    denom = np.where(np.abs(S_R - S_L) < 1e-12, 1e-12, S_R - S_L)
    F1_hll = (S_R * F1_L - S_L * F1_R + S_L * S_R * (rho_R - rho_L)) / denom
    F2_hll = (S_R * F2_L - S_L * F2_R + S_L * S_R * (y_R - y_L)) / denom

    F1 = np.where(S_L >= 0.0, F1_L, np.where(S_R <= 0.0, F1_R, F1_hll))
    F2 = np.where(S_L >= 0.0, F2_L, np.where(S_R <= 0.0, F2_R, F2_hll))
    return F1, F2


# --------------------------------------------------------------------------- #
# HLLC numerical flux  (restores the linearly-degenerate 2-contact)
# --------------------------------------------------------------------------- #
def _hllc_flux(
    rho_L: np.ndarray, y_L: np.ndarray,
    rho_R: np.ndarray, y_R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """HLLC flux for ARZ on conserved (rho, y).

    HLL collapses the wave fan into ONE averaged intermediate state, so it smears
    the linearly-degenerate 2-contact (lambda2 = v, a discontinuity carrying a
    rho/w jump but *continuous* v). HLLC restores the contact as a sharp middle
    wave.

    **ARZ makes this nearly exact.** The two Riemann invariants are known in
    closed form (same structure the exact solver `riemann_arz` uses):

        * w is constant across the 1-wave  ->  w_* = w_L
        * v is constant across the 2-contact ->  v_* = v_R, and the contact moves
          at exactly S_* = v_R  (NOT an HLL average -- that was the bug).

    So the self-similar fan has just three regions sampled at xi=0:

        L  | 1-wave (slowest speed S_L) | star (w=w_L, v=v_R) | contact (S_*=v_R) | R

    The single intermediate (star) state is fully determined:
        v_*   = v_R,  w_* = w_L,  p(rho_*) implied -- but for an HLLC *flux* we get
        rho_* from the Rankine-Hugoniot jump across the 1-wave's HLL speed S_L:
            rho_* = rho_L * (S_L - v_L) / (S_L - S_*)          (mass RH, v_*=S_*)
        y_*   = rho_* * w_L                                    (w preserved L->*)
    and the Toro middle flux is  F_* = F_L + S_L (U_* - U_L).

    Sampling at the interface (xi=0):
        F = F_L     if S_L >= 0           (whole fan moves right)
        F = F_*     if S_L < 0 <= S_*     (star state straddles the interface)
        F = F_R     if S_* < 0            (contact has passed; right state at face)

    Degenerate denominators are floored and any non-finite result falls back to
    HLL, so the scheme never produces NaN (the failure mode of the old WENO5).
    """
    _, w_L, v_L = P.to_primitive(rho_L, y_L)
    _, w_R, v_R = P.to_primitive(rho_R, y_R)
    lam1_L = v_L - rho_L * P.dpressure(rho_L)
    lam1_R = v_R - rho_R * P.dpressure(rho_R)

    # 1-wave (slowest) speed estimate; contact speed is exact.
    S_L = np.minimum(lam1_L, lam1_R)
    S_star = v_R                       # exact contact speed for ARZ

    F1_L, F2_L = P.flux(rho_L, y_L)
    F1_R, F2_R = P.flux(rho_R, y_R)

    # Star state across the 1-wave (HLL speed S_L), enforcing v_* = S_*, w_* = w_L.
    denom = S_L - S_star
    denom = np.where(np.abs(denom) < 1e-9, np.where(denom >= 0, 1e-9, -1e-9), denom)
    rho_star = rho_L * (S_L - v_L) / denom
    rho_star = np.maximum(rho_star, P.RHO_MIN)
    y_star = rho_star * w_L

    F1_starL = F1_L + S_L * (rho_star - rho_L)
    F2_starL = F2_L + S_L * (y_star - y_L)

    # Sample the three-region fan at xi = 0.
    F1 = np.where(S_L >= 0.0, F1_L,
                  np.where(S_star >= 0.0, F1_starL, F1_R))
    F2 = np.where(S_L >= 0.0, F2_L,
                  np.where(S_star >= 0.0, F2_starL, F2_R))

    # Robustness fallback: any non-finite flux drops back to HLL there.
    bad = ~(np.isfinite(F1) & np.isfinite(F2))
    if bad.any():
        F1_hll, F2_hll = _hll_flux(rho_L, y_L, rho_R, y_R)
        F1 = np.where(bad, F1_hll, F1)
        F2 = np.where(bad, F2_hll, F2)
    return F1, F2


# --------------------------------------------------------------------------- #
# Exact (true Godunov) numerical flux
# --------------------------------------------------------------------------- #
def _godunov_flux(
    rho_L: np.ndarray, y_L: np.ndarray,
    rho_R: np.ndarray, y_R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Godunov flux for ARZ: solve the TRUE Riemann problem at each
    interface and sample the self-similar solution at xi = x/t = 0, then take
    its physical flux. This is the system analogue of the LWR exact-Godunov
    flux (fvm.godunov_flux) -- it resolves BOTH the GNL 1-wave and the
    linearly-degenerate 2-contact exactly, unlike HLL which averages the
    contact away into a single intermediate state.

    Reuses the exact ARZ Riemann solver `solve_riemann_arz` (the same one that
    generates the dataset ground truth). solve_riemann_arz is scalar-per-call,
    so we loop interfaces -- baselines are not on the training hot path.
    """
    from hyperbolic_pde.arz.riemann_arz import solve_riemann_arz

    n = rho_L.shape[0]
    # Recover primitive w on each side (interface states).
    _, w_L, _ = P.to_primitive(rho_L, y_L)
    _, w_R, _ = P.to_primitive(rho_R, y_R)

    rho_face = np.empty(n, dtype=np.float64)
    w_face = np.empty(n, dtype=np.float64)
    xi0 = np.array([0.0])  # sample the Riemann solution at the cell interface
    for i in range(n):
        r, wv, _ = solve_riemann_arz(
            float(rho_L[i]), float(w_L[i]),
            float(rho_R[i]), float(w_R[i]),
            xi0,
        )
        rho_face[i] = r[0]
        w_face[i] = wv[0]

    F1, F2 = P.flux_from_rw(rho_face, w_face)
    return F1, F2


def _hyperbolic_step(
    rho: np.ndarray, y: np.ndarray,
    dx: float, dt: float, boundary: str,
    flux_scheme: str = "hll",
) -> Tuple[np.ndarray, np.ndarray]:
    """First-order finite-volume hyperbolic update on (rho, y).

    flux_scheme: 'hll' (default, fast, contact-smearing) or 'godunov' (exact
    Riemann flux, sharp contact -- the LWR-equivalent Godunov).
    """
    rho_ext = _pad_ghost(rho, 1, boundary)
    y_ext = _pad_ghost(y, 1, boundary)
    if flux_scheme == "godunov":
        F1, F2 = _godunov_flux(rho_ext[:-1], y_ext[:-1], rho_ext[1:], y_ext[1:])
    elif flux_scheme == "hll":
        F1, F2 = _hll_flux(rho_ext[:-1], y_ext[:-1], rho_ext[1:], y_ext[1:])
    elif flux_scheme == "hllc":
        F1, F2 = _hllc_flux(rho_ext[:-1], y_ext[:-1], rho_ext[1:], y_ext[1:])
    else:
        raise ValueError(f"flux_scheme must be 'hll', 'hllc' or 'godunov', got {flux_scheme!r}")
    rho_new = rho - (dt / dx) * (F1[1:] - F1[:-1])
    y_new = y - (dt / dx) * (F2[1:] - F2[:-1])
    # Floor rho to RHO_MIN to keep w extraction safe.
    rho_new = np.maximum(rho_new, P.RHO_MIN)
    return rho_new, y_new


def _source_step(
    rho: np.ndarray, y: np.ndarray, dt: float, tau: float,
) -> np.ndarray:
    """Exact integration of dy/dt = (y_eq - y)/tau at fixed rho over dt."""
    y_e = P.y_eq(rho)
    return y_e + (y - y_e) * np.exp(-dt / tau)


# --------------------------------------------------------------------------- #
# WENO5 + SSP-RK3 solver
# --------------------------------------------------------------------------- #
def _weno5_arz_rhs(rho, y, dx, boundary):
    """Component-wise WENO5 + global LF flux on (rho, y)."""
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
    rho1 = np.maximum(rho + dt * Lr1, P.RHO_MIN)
    y1   = y + dt * Ly1

    Lr2, Ly2 = L(rho1, y1)
    rho2 = np.maximum(0.75 * rho + 0.25 * (rho1 + dt * Lr2), P.RHO_MIN)
    y2   = 0.75 * y + 0.25 * (y1 + dt * Ly2)

    Lr3, Ly3 = L(rho2, y2)
    rho_n = np.maximum((1.0/3.0) * rho + (2.0/3.0) * (rho2 + dt * Lr3), P.RHO_MIN)
    y_n   = (1.0/3.0) * y + (2.0/3.0) * (y2 + dt * Ly3)
    return rho_n, y_n


def solve_arz_weno5(
    rho0, w0, x_min: float, x_max: float, t_max: float,
    nt_out: int, tau: float, cfl: float = 0.2, boundary: str = "ghost",
) -> Tuple[np.ndarray, np.ndarray]:
    """Strang-split WENO5 + SSP-RK3 on (rho, y) with exact relaxation.

    Operates on the training grid directly (no refinement — that's the point
    of WENO5 vs the HLL reference which needs refine>1 for accuracy).
    Returns (rho_hist, w_hist) each of shape (nt_out, nx), float32.
    """
    rho0 = np.asarray(rho0, dtype=np.float64)
    w0   = np.asarray(w0,   dtype=np.float64)
    nx   = rho0.size
    dx   = (x_max - x_min) / nx
    t_out = np.linspace(0.0, t_max, nt_out, dtype=np.float64)

    rho = np.maximum(rho0.copy(), P.RHO_MIN)
    y   = rho * w0
    rho_hist = np.empty((nt_out, nx), dtype=np.float32)
    w_hist   = np.empty((nt_out, nx), dtype=np.float32)
    rho_hist[0] = rho.astype(np.float32)
    w_hist[0]   = (y / np.maximum(rho, P.RHO_MIN)).astype(np.float32)

    t = 0.0
    k = 1
    while k < nt_out:
        w_now   = y / np.maximum(rho, P.RHO_MIN)
        lam_max = max(float(P.spectral_radius(rho, w_now).max()), 1e-6)
        dt = min(cfl * dx / lam_max, t_out[k] - t)
        if dt <= 0:
            break
        # Strang: half-step source -> WENO hyperbolic -> half-step source.
        y_eq = P.y_eq(rho)
        y  = y_eq + (y - y_eq) * np.exp(-0.5 * dt / tau)
        rho, y = _weno5_arz_step(rho, y, dx, dt, boundary)
        y_eq = P.y_eq(rho)
        y  = y_eq + (y - y_eq) * np.exp(-0.5 * dt / tau)
        t += dt
        while k < nt_out and t >= t_out[k] - 1e-12:
            rho_hist[k] = rho.astype(np.float32)
            w_hist[k]   = (y / np.maximum(rho, P.RHO_MIN)).astype(np.float32)
            k += 1
    return rho_hist, w_hist


# --------------------------------------------------------------------------- #
# MUSCL-Hancock + HLLC  (2nd-order, contact-sharp, robust)
# --------------------------------------------------------------------------- #
def _minmod(a, b):
    """Scalar minmod limiter (vectorised): 0 if a,b differ in sign, else the
    smaller magnitude. The TVD limiter that kills MUSCL's spurious oscillations."""
    return np.where(a * b <= 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


def _muscl_reconstruct(q, boundary):
    """minmod-limited linear reconstruction of cell-centred field q.

    Returns (q_iL, q_iR): the left- and right-face extrapolated values of each
    interior cell i, i.e. q_iL = q_i - 0.5*slope, q_iR = q_i + 0.5*slope.
    Slope = minmod(q_i - q_{i-1}, q_{i+1} - q_i). One ghost layer each side.
    """
    qe = _pad_ghost(q, 1, boundary)          # length nx+2
    dminus = qe[1:-1] - qe[:-2]              # q_i - q_{i-1}
    dplus = qe[2:] - qe[1:-1]               # q_{i+1} - q_i
    slope = _minmod(dminus, dplus)
    q_iL = q - 0.5 * slope
    q_iR = q + 0.5 * slope
    return q_iL, q_iR


def _muscl_hancock_step(rho, y, dx, dt, boundary):
    """One MUSCL-Hancock predictor-corrector step on (rho, y) with HLLC flux.

    1. Reconstruct primitive (rho, v) with minmod slopes -> face values.
    2. Hancock predictor: evolve each cell's face values by dt/2 using the
       in-cell flux difference (boundary-extrapolated, no Riemann solve).
    3. HLLC at each interface between the predicted right-face of cell i and
       the predicted left-face of cell i+1.
    4. Conservative update.
    rho is floored to RHO_MIN throughout for vacuum safety.
    """
    rho = np.maximum(rho, P.RHO_MIN)
    _, w, v = P.to_primitive(rho, y)

    # 1. Reconstruct in primitive (rho, v) -- limited there so a sharp contact
    #    (rho jumps, v flat) does not bleed oscillations into v, and vice-versa.
    rhoL, rhoR = _muscl_reconstruct(rho, boundary)
    vL, vR = _muscl_reconstruct(v, boundary)
    rhoL = np.maximum(rhoL, P.RHO_MIN)
    rhoR = np.maximum(rhoR, P.RHO_MIN)

    # Convert face primitives to conserved (rho, y).
    wL = vL + P.pressure(rhoL)
    wR = vR + P.pressure(rhoR)
    yL = rhoL * wL
    yR = rhoR * wR

    # 2. Hancock half-step: U_face += (dt/2dx) * (F(U_iL) - F(U_iR)).
    F1L, F2L = P.flux_from_rw(rhoL, wL)
    F1R, F2R = P.flux_from_rw(rhoR, wR)
    half = 0.5 * dt / dx
    rhoL_p = np.maximum(rhoL + half * (F1L - F1R), P.RHO_MIN)
    rhoR_p = np.maximum(rhoR + half * (F1L - F1R), P.RHO_MIN)
    yL_p = yL + half * (F2L - F2R)
    yR_p = yR + half * (F2L - F2R)

    # 3. Interface HLLC between right-face of cell i and left-face of cell i+1.
    #    Pad the predicted face arrays with one ghost cell to form interfaces.
    rhoR_ext = _pad_ghost(rhoR_p, 1, boundary)
    yR_ext = _pad_ghost(yR_p, 1, boundary)
    rhoL_ext = _pad_ghost(rhoL_p, 1, boundary)
    yL_ext = _pad_ghost(yL_p, 1, boundary)
    # Interface k sits between cell k-1 (its right face) and cell k (its left
    # face). There are nx+1 interfaces; the left state is the right-face of the
    # left neighbour, the right state is the left-face of the right neighbour.
    rho_im = rhoR_ext[:-1]     # right face of left cell, length nx+1
    y_im = yR_ext[:-1]
    rho_ip = rhoL_ext[1:]      # left face of right cell, length nx+1
    y_ip = yL_ext[1:]
    F1, F2 = _hllc_flux(rho_im, y_im, rho_ip, y_ip)

    # 4. Conservative update.
    rho_new = np.maximum(rho - (dt / dx) * (F1[1:] - F1[:-1]), P.RHO_MIN)
    y_new = y - (dt / dx) * (F2[1:] - F2[:-1])
    return rho_new, y_new


def solve_arz_muscl(
    rho0, w0, x_min: float, x_max: float, t_max: float,
    nt_out: int, tau: float, cfl: float = 0.4, boundary: str = "ghost",
) -> Tuple[np.ndarray, np.ndarray]:
    """Strang-split MUSCL-Hancock + HLLC on (rho, y) with exact relaxation.

    Second-order, contact-resolving, and robust where the component-wise WENO5
    (global Lax-Friedrichs, no characteristic decomposition) blows up: the
    reconstruction is minmod-limited in primitive (rho, v) and rho is floored
    everywhere, so it cannot drive rho negative / produce NaN.

    Operates on the training grid directly (like solve_arz_weno5).
    Returns (rho_hist, w_hist) each (nt_out, nx), float32.
    """
    rho0 = np.asarray(rho0, dtype=np.float64)
    w0 = np.asarray(w0, dtype=np.float64)
    nx = rho0.size
    dx = (x_max - x_min) / nx
    t_out = np.linspace(0.0, t_max, nt_out, dtype=np.float64)

    rho = np.maximum(rho0.copy(), P.RHO_MIN)
    y = rho * w0
    rho_hist = np.empty((nt_out, nx), dtype=np.float32)
    w_hist = np.empty((nt_out, nx), dtype=np.float32)
    rho_hist[0] = rho.astype(np.float32)
    w_hist[0] = (y / np.maximum(rho, P.RHO_MIN)).astype(np.float32)

    t = 0.0
    k = 1
    while k < nt_out:
        w_now = y / np.maximum(rho, P.RHO_MIN)
        lam_max = max(float(P.spectral_radius(rho, w_now).max()), 1e-6)
        dt = min(cfl * dx / lam_max, t_out[k] - t)
        if dt <= 0:
            break
        # Strang: half-step source -> MUSCL-Hancock hyperbolic -> half-step source.
        y_eq = P.y_eq(rho)
        y = y_eq + (y - y_eq) * np.exp(-0.5 * dt / tau)
        rho, y = _muscl_hancock_step(rho, y, dx, dt, boundary)
        y_eq = P.y_eq(rho)
        y = y_eq + (y - y_eq) * np.exp(-0.5 * dt / tau)
        t += dt
        while k < nt_out and t >= t_out[k] - 1e-12:
            rho_hist[k] = rho.astype(np.float32)
            w_hist[k] = (y / np.maximum(rho, P.RHO_MIN)).astype(np.float32)
            k += 1
    return rho_hist, w_hist


# --------------------------------------------------------------------------- #
# HLL/Godunov driver
# --------------------------------------------------------------------------- #
def solve_arz_reference(
    rho0: np.ndarray, w0: np.ndarray,
    x_min: float, x_max: float, t_max: float,
    nt_out: int,
    tau: float,
    cfl: float = 0.4,
    boundary: str = "periodic",
    refine: int = 1,
    flux_scheme: str = "hll",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve 1D ARZ with relaxation.

    flux_scheme : 'hll' (default -- preserves the dataset ground-truth solver)
        or 'godunov' (exact-Riemann flux, sharp contact; for baselines/eval).

    Parameters
    ----------
    rho0, w0 : arrays of shape (nx,)
        Initial primitive state on the training grid.
    x_min, x_max, t_max : floats
    nt_out : int
        Number of output snapshots (incl. t=0).
    tau : float
        Relaxation time (use a large value, e.g. 1e6, for homogeneous ARZ).
    cfl : float
    boundary : 'periodic' | 'ghost' | 'fixed'
    refine : int
        Spatial refinement factor for the fine reference grid. The IC is
        upsampled (piecewise constant) to nx_fine = nx * refine, solved on
        that grid, and the snapshots are restricted back to nx by block-mean.

    Returns
    -------
    x, t, rho, w  — x shape (nx,), t shape (nt_out,), rho/w shape (nt_out, nx).
    """
    rho0 = np.asarray(rho0, dtype=np.float64)
    w0 = np.asarray(w0, dtype=np.float64)
    nx = rho0.size
    assert w0.size == nx

    # Fine grid (refined for ground-truth accuracy).
    nx_fine = nx * int(refine)
    if refine > 1:
        rho_f = np.repeat(rho0, refine)
        w_f = np.repeat(w0, refine)
    else:
        rho_f = rho0.copy()
        w_f = w0.copy()
    y_f = rho_f * w_f
    rho_f = np.maximum(rho_f, P.RHO_MIN)

    # Finite-volume cell-centred grid: nx cells of width dx = L/nx, centres at
    # x_min + (i+0.5)*dx. This MUST match the dataset's midpoint convention
    # (datagen_arz uses dx = (x_max-x_min)/nx, centres x_min+(i+0.5)*dx), or the
    # baseline lands half a cell off the ground truth and the dx used in the flux
    # divergence is wrong by a factor nx/(nx-1). [x_min, x_max] here is the FULL
    # domain extent (the caller passes the true edges, not the midpoint min/max).
    dx_fine = (x_max - x_min) / nx_fine
    x_fine = x_min + (np.arange(nx_fine) + 0.5) * dx_fine
    x_out = (x_min + (np.arange(nx) + 0.5) * ((x_max - x_min) / nx)).astype(np.float32)
    t_out = np.linspace(0.0, t_max, nt_out, dtype=np.float64)

    rho_hist = np.empty((nt_out, nx), dtype=np.float32)
    w_hist = np.empty((nt_out, nx), dtype=np.float32)

    def _restrict(arr_fine: np.ndarray) -> np.ndarray:
        if refine == 1:
            return arr_fine.astype(np.float32)
        # block-mean restriction
        return arr_fine.reshape(nx, refine).mean(axis=1).astype(np.float32)

    rho_hist[0] = _restrict(rho_f)
    w_hist[0] = _restrict(rho_f * 0 + w_f)  # y/rho stays w_f when refine==1

    t = 0.0
    k = 1
    while k < nt_out:
        # CFL on the fine grid.
        lam_max = float(P.spectral_radius(rho_f, w_f).max())
        lam_max = max(lam_max, 1e-6)
        dt = cfl * dx_fine / lam_max
        dt = min(dt, t_out[k] - t)
        if dt <= 0:
            break

        # Strang: 1/2 source -> hyperbolic -> 1/2 source.
        y_f = _source_step(rho_f, y_f, 0.5 * dt, tau)
        rho_f, y_f = _hyperbolic_step(rho_f, y_f, dx_fine, dt, boundary,
                                      flux_scheme=flux_scheme)
        y_f = _source_step(rho_f, y_f, 0.5 * dt, tau)
        # Refresh w_f from (rho_f, y_f) for CFL next iter.
        _, w_f, _ = P.to_primitive(rho_f, y_f)
        t += dt
        if not np.isfinite(rho_f).all() or not np.isfinite(y_f).all():
            raise RuntimeError(f"reference solver diverged at t={t:.4f}")

        # Record snapshot if we've reached the next output time.
        while k < nt_out and t >= t_out[k] - 1e-12:
            # No temporal interpolation needed here: dt is clamped to land
            # exactly on the next t_out[k].
            rho_hist[k] = _restrict(rho_f)
            w_hist[k] = _restrict(w_f)
            k += 1

    return x_out, t_out.astype(np.float32), rho_hist, w_hist
