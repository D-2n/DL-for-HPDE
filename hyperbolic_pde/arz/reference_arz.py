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
    else:
        raise ValueError(f"flux_scheme must be 'hll' or 'godunov', got {flux_scheme!r}")
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
# Driver
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
