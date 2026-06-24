"""Standalone first-order finite-volume solver for the 1D ARZ system (HLL/HLLC).

Self-contained (numpy only). Extracted from the DL-for-HPDE repo:
  - HLL / HLLC numerical fluxes + FV driver   <- hyperbolic_pde/arz/reference_arz.py
  - pressure / equilibrium closures           <- hyperbolic_pde/arz/physics_arz.py

ARZ in conserved form (rho, y = rho*w), with v = w - p(rho), w = y/rho:
    rho_t + (rho v)_x = 0
    y_t   + (y   v)_x = (y_eq(rho) - y) / tau
Pass tau = np.inf for the homogeneous (source-free) system.

Time integration: Strang splitting per step,
    1/2 relaxation source  ->  first-order FV hyperbolic step  ->  1/2 source,
with an EXACT integration of the (linear, fixed-rho) source ODE.

Flux schemes:
  "hll"  : HLL on conserved (rho, y). Fast, but averages the linearly-degenerate
           2-contact into one intermediate state -> smears contacts. This is the
           repo's dataset ground-truth flux.
  "hllc" : restores the 2-contact as a sharp middle wave. ARZ makes this nearly
           exact: w_* = w_L across the 1-wave, contact speed S_* = v_R exactly.

"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Pressure closure  (single point of control)
# --------------------------------------------------------------------------- #
RHO_MIN = 1e-6                       # vacuum guard inside y = rho*w products
_P_FORM = "rho"                      # project default is ALWAYS "rho"; also "rho+rho2"
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


def to_primitive(rho, y):
    """(rho, y) -> (rho, w, v) with a vacuum-safe division."""
    rho_safe = np.maximum(rho, RHO_MIN)
    w = y / rho_safe
    v = w - pressure(rho)
    return rho, w, v


def flux(rho, y):
    """Conserved flux (F1, F2) = (rho*v, y*v) from (rho, y)."""
    _, _, v = to_primitive(rho, y)
    return rho * v, y * v


def spectral_radius(rho, w):
    """max(|lambda1|, |lambda2|): lambda1 = v - rho*p'(rho), lambda2 = v."""
    v = w - pressure(rho)
    lam1 = v - rho * dpressure(rho)
    lam2 = v
    return np.maximum(np.abs(lam1), np.abs(lam2))


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
def _hll_flux(rho_L, y_L, rho_R, y_R) -> Tuple[np.ndarray, np.ndarray]:
    """HLL flux for ARZ on conserved (rho, y).

    Wave-speed estimates: S_L = min(lambda1_L, lambda1_R),
                          S_R = max(lambda2_L, lambda2_R).
    """
    _, w_L, v_L = to_primitive(rho_L, y_L)
    _, w_R, v_R = to_primitive(rho_R, y_R)
    lam1_L = v_L - rho_L * dpressure(rho_L)
    lam2_L = v_L
    lam1_R = v_R - rho_R * dpressure(rho_R)
    lam2_R = v_R

    S_L = np.minimum(lam1_L, lam1_R)
    S_R = np.maximum(lam2_L, lam2_R)

    F1_L, F2_L = flux(rho_L, y_L)
    F1_R, F2_R = flux(rho_R, y_R)

    denom = np.where(np.abs(S_R - S_L) < 1e-12, 1e-12, S_R - S_L)
    F1_hll = (S_R * F1_L - S_L * F1_R + S_L * S_R * (rho_R - rho_L)) / denom
    F2_hll = (S_R * F2_L - S_L * F2_R + S_L * S_R * (y_R - y_L)) / denom

    F1 = np.where(S_L >= 0.0, F1_L, np.where(S_R <= 0.0, F1_R, F1_hll))
    F2 = np.where(S_L >= 0.0, F2_L, np.where(S_R <= 0.0, F2_R, F2_hll))
    return F1, F2


# --------------------------------------------------------------------------- #
# HLLC numerical flux  (restores the linearly-degenerate 2-contact)
# --------------------------------------------------------------------------- #
def _hllc_flux(rho_L, y_L, rho_R, y_R) -> Tuple[np.ndarray, np.ndarray]:
    """HLLC flux for ARZ: sharp 2-contact via the exact ARZ invariants
    (w_* = w_L across the 1-wave, contact speed S_* = v_R). Non-finite results
    fall back to HLL pointwise, so the scheme never produces NaN."""
    _, w_L, v_L = to_primitive(rho_L, y_L)
    _, w_R, v_R = to_primitive(rho_R, y_R)
    lam1_L = v_L - rho_L * dpressure(rho_L)
    lam1_R = v_R - rho_R * dpressure(rho_R)

    S_L = np.minimum(lam1_L, lam1_R)
    S_star = v_R                       # exact contact speed for ARZ

    F1_L, F2_L = flux(rho_L, y_L)
    F1_R, F2_R = flux(rho_R, y_R)

    # Star state across the 1-wave (HLL speed S_L), enforcing v_* = S_*, w_* = w_L.
    denom = S_L - S_star
    denom = np.where(np.abs(denom) < 1e-9, np.where(denom >= 0, 1e-9, -1e-9), denom)
    rho_star = np.maximum(rho_L * (S_L - v_L) / denom, RHO_MIN)
    y_star = rho_star * w_L

    F1_starL = F1_L + S_L * (rho_star - rho_L)
    F2_starL = F2_L + S_L * (y_star - y_L)

    F1 = np.where(S_L >= 0.0, F1_L, np.where(S_star >= 0.0, F1_starL, F1_R))
    F2 = np.where(S_L >= 0.0, F2_L, np.where(S_star >= 0.0, F2_starL, F2_R))

    bad = ~(np.isfinite(F1) & np.isfinite(F2))
    if bad.any():
        F1_hll, F2_hll = _hll_flux(rho_L, y_L, rho_R, y_R)
        F1 = np.where(bad, F1_hll, F1)
        F2 = np.where(bad, F2_hll, F2)
    return F1, F2


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def _hyperbolic_step(rho, y, dx, dt, boundary, flux_scheme):
    """First-order finite-volume hyperbolic update on (rho, y)."""
    rho_ext = _pad_ghost(rho, 1, boundary)
    y_ext = _pad_ghost(y, 1, boundary)
    if flux_scheme == "hll":
        F1, F2 = _hll_flux(rho_ext[:-1], y_ext[:-1], rho_ext[1:], y_ext[1:])
    elif flux_scheme == "hllc":
        F1, F2 = _hllc_flux(rho_ext[:-1], y_ext[:-1], rho_ext[1:], y_ext[1:])
    else:
        raise ValueError(f"flux_scheme must be 'hll' or 'hllc', got {flux_scheme!r}")
    rho_new = np.maximum(rho - (dt / dx) * (F1[1:] - F1[:-1]), RHO_MIN)
    y_new = y - (dt / dx) * (F2[1:] - F2[:-1])
    return rho_new, y_new


def _source_step(rho, y, dt, tau):
    """Exact integration of dy/dt = (y_eq - y)/tau at fixed rho over dt."""
    y_e = y_eq(rho)
    return y_e + (y - y_e) * np.exp(-dt / tau)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def solve_arz_fv(
    rho0, w0, x_min: float, x_max: float, t_max: float,
    nt_out: int, tau: float, cfl: float = 0.4, boundary: str = "ghost",
    flux_scheme: str = "hll",
) -> Tuple[np.ndarray, np.ndarray]:
    """Strang-split first-order FV (HLL/HLLC) on (rho, y) with exact relaxation.

    Parameters
    ----------
    rho0, w0 : (nx,) initial density and w = v + p(rho).
    x_min, x_max, t_max : domain extents (pass TRUE edges, not cell midpoints).
    nt_out : number of output snapshots (including t=0).
    tau : relaxation time; pass np.inf for the homogeneous (source-free) system.
    cfl : CFL number (first-order FV is stable to ~0.9; 0.4 is conservative).
    boundary : "ghost" (zero-gradient), "fixed", or "periodic".
    flux_scheme : "hll" (default) or "hllc" (sharp contact).

    Returns (rho_hist, w_hist), each (nt_out, nx) float32.
    """
    rho0 = np.asarray(rho0, dtype=np.float64)
    w0 = np.asarray(w0, dtype=np.float64)
    nx = rho0.size
    dx = (x_max - x_min) / nx
    t_out = np.linspace(0.0, t_max, nt_out, dtype=np.float64)

    rho = np.maximum(rho0.copy(), RHO_MIN)
    y = rho * w0
    _, w, _ = to_primitive(rho, y)

    rho_hist = np.empty((nt_out, nx), dtype=np.float32)
    w_hist = np.empty((nt_out, nx), dtype=np.float32)
    rho_hist[0] = rho.astype(np.float32)
    w_hist[0] = w.astype(np.float32)

    t = 0.0
    k = 1
    while k < nt_out:
        lam_max = max(float(spectral_radius(rho, w).max()), 1e-6)
        dt = min(cfl * dx / lam_max, t_out[k] - t)
        if dt <= 0:
            break
        # Strang: 1/2 source -> hyperbolic -> 1/2 source.
        y = _source_step(rho, y, 0.5 * dt, tau)
        rho, y = _hyperbolic_step(rho, y, dx, dt, boundary, flux_scheme)
        y = _source_step(rho, y, 0.5 * dt, tau)
        _, w, _ = to_primitive(rho, y)
        t += dt
        if not (np.isfinite(rho).all() and np.isfinite(y).all()):
            raise RuntimeError(f"solver diverged at t={t:.4f}")
        while k < nt_out and t >= t_out[k] - 1e-12:
            rho_hist[k] = rho.astype(np.float32)
            w_hist[k] = w.astype(np.float32)
            k += 1
    return rho_hist, w_hist


# --------------------------------------------------------------------------- #
# Demo: a single Riemann problem (homogeneous ARZ, tau = inf)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    set_pressure_form("rho")               # project default

    nx, nt = 128, 128
    x_min, x_max, t_max = -1.0, 1.0, 1.0
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, nx)

    rhoL, vL = 0.1, 1.01
    rhoR, vR = 0.9, 0.05
    rho0 = np.where(x < 0.0, rhoL, rhoR)
    v0 = np.where(x < 0.0, vL, vR)
    w0 = v0 + pressure(rho0)

    for scheme in ("hll", "hllc"):
        rho_h, w_h = solve_arz_fv(
            rho0, w0, x_min, x_max, t_max, nt_out=nt, tau=np.inf,
            cfl=0.4, flux_scheme=scheme,
        )
        print(f"[{scheme}] p={get_pressure_form()}  rho_hist{rho_h.shape}  "
              f"final rho[::16]={np.round(rho_h[-1, ::16], 3)}")
