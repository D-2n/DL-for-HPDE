"""ARZ physics, vectorised over numpy and torch.

All quantities use the frozen pressure p(rho) = rho + rho^2 (see plan §0).
State is carried as (rho, w); the conservative pair (rho, y = rho*w) is
formed only inside flux / source evaluations, never inside w itself
(vacuum safety: rho can hit 0 without polluting w).
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

try:
    import torch
    _Array = Union[np.ndarray, "torch.Tensor"]
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _Array = np.ndarray  # type: ignore
    _HAS_TORCH = False


RHO_MIN = 1e-6  # vacuum guard inside y = rho*w products


# --------------------------------------------------------------------------- #
# Backend-agnostic helpers
# --------------------------------------------------------------------------- #
def _abs(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.abs()
    return np.abs(x)


def _maximum(a, b):
    if _HAS_TORCH and (isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)):
        return torch.maximum(
            a if isinstance(a, torch.Tensor) else torch.as_tensor(a),
            b if isinstance(b, torch.Tensor) else torch.as_tensor(b),
        )
    return np.maximum(a, b)


# --------------------------------------------------------------------------- #
# Closures
# --------------------------------------------------------------------------- #
# Pressure form toggle. The whole ARZ codebase reads p / p' / w_eq / y_eq from
# this module, so flipping _P_FORM here is the single point of control.
#
#   "rho+rho2" : p(rho) = rho + rho^2          (original AR-style closure)
#   "rho"      : p(rho) = rho                  (linear -- simpler Riemann problem)
#
# Use `set_pressure_form(...)` from scripts at startup; never read _P_FORM
# directly outside this module.
_P_FORM = "rho+rho2"
_VALID_P_FORMS = ("rho", "rho+rho2")


def set_pressure_form(form: str) -> None:
    """Globally switch the pressure closure used by pressure/dpressure/w_eq/y_eq."""
    global _P_FORM
    if form not in _VALID_P_FORMS:
        raise ValueError(f"Unknown pressure_form={form!r}; expected one of {_VALID_P_FORMS}.")
    _P_FORM = form


def get_pressure_form() -> str:
    return _P_FORM


def _ones_like(rho):
    """Backend-agnostic ones-like (broadcast-safe scalar substitute)."""
    if _HAS_TORCH and isinstance(rho, torch.Tensor):
        return torch.ones_like(rho)
    if isinstance(rho, np.ndarray):
        return np.ones_like(rho)
    return 1.0


def _zeros_like(rho):
    if _HAS_TORCH and isinstance(rho, torch.Tensor):
        return torch.zeros_like(rho)
    if isinstance(rho, np.ndarray):
        return np.zeros_like(rho)
    return 0.0


def pressure(rho):
    """p(rho)  -- closure chosen by `set_pressure_form`."""
    if _P_FORM == "rho":
        return rho
    return rho + rho * rho


def dpressure(rho):
    """p'(rho)  -- closure chosen by `set_pressure_form`.

    Returns a tensor/array matching `rho` even when the derivative is constant,
    so downstream broadcasting (e.g. lambda_ij = v - rho * p'(rho)) stays
    shape-consistent.
    """
    if _P_FORM == "rho":
        return _ones_like(rho)
    return 1.0 + 2.0 * rho


def Veq(rho):
    """V(rho) = 1 - rho  (Greenshields equilibrium velocity)."""
    return 1.0 - rho


def w_eq(rho):
    """w_eq(rho) = V(rho) + p(rho)."""
    if _P_FORM == "rho":
        # 1 - rho + rho = 1; broadcast-safe.
        return _ones_like(rho)
    return 1.0 + rho * rho


def y_eq(rho):
    """y_eq(rho) = rho * w_eq(rho)."""
    if _P_FORM == "rho":
        return rho                # = rho * 1
    return rho + rho * rho * rho


def velocity(rho, w):
    """Primitive velocity v = w - p(rho)."""
    return w - pressure(rho)


# --------------------------------------------------------------------------- #
# Conserved <-> primitive
# --------------------------------------------------------------------------- #
def to_conserved(rho, w):
    """(rho, w) -> (rho, y = rho*w)."""
    return rho, rho * w


def to_primitive(rho, y):
    """(rho, y) -> (rho, w, v).

    Uses RHO_MIN vacuum guard inside the division so y/rho stays finite.
    Outside this routine, never divide by rho.
    """
    rho_safe = _maximum(rho, RHO_MIN)
    w = y / rho_safe
    v = w - pressure(rho)
    return rho, w, v


# --------------------------------------------------------------------------- #
# Eigenvalues & flux
# --------------------------------------------------------------------------- #
def eigenvalues(rho, w):
    """Return (lambda1, lambda2) from (rho, w).

    lambda1 = v - rho * p'(rho) = v - rho - 2*rho^2  (GNL)
    lambda2 = v                                       (LD contact)
    """
    v = velocity(rho, w)
    lam2 = v
    lam1 = v - rho * dpressure(rho)
    return lam1, lam2


def flux(rho, y):
    """Conserved flux (F1, F2) from (rho, y).

    F1 = rho * v
    F2 = y   * v
    """
    # v from (rho, y) via to_primitive — vacuum-safe.
    _, _, v = to_primitive(rho, y)
    F1 = rho * v
    F2 = y * v
    return F1, F2


def flux_from_rw(rho, w):
    """Conserved flux from primitive (rho, w)."""
    v = velocity(rho, w)
    F1 = rho * v
    F2 = (rho * w) * v
    return F1, F2


def relaxation_source(rho, y, tau):
    """Source term S = (0, (y_eq - y)/tau).

    Returns (S_rho, S_y) — S_rho is identically zero.
    """
    S_y = (y_eq(rho) - y) / tau
    # build a zero of the right shape/backend
    if _HAS_TORCH and isinstance(rho, torch.Tensor):
        S_rho = torch.zeros_like(rho)
    else:
        S_rho = np.zeros_like(rho)
    return S_rho, S_y


# --------------------------------------------------------------------------- #
# Interface helpers (used by gates, HLL flux, etc.)
# --------------------------------------------------------------------------- #
def interface_state(rho_L, w_L, rho_R, w_R, mode: str = "arith"):
    """Average interface state in (rho, v), then recompute (w, lambda1, lambda2).

    Returns dict-like tuple: (rho_ij, v_ij, w_ij, lam1_ij, lam2_ij).

    mode='arith' (default): plain arithmetic average of (rho, v) at the interface.
    mode='roe' is stubbed for later — falls back to arith for now with a note.
    """
    if mode not in {"arith", "roe"}:
        raise ValueError(f"interface_state mode must be 'arith' or 'roe'; got {mode!r}")
    if mode == "roe":
        # Stub: a proper ARZ Roe average lives in the w variable.
        # For now defer to arithmetic; sharpen later.
        pass

    v_L = velocity(rho_L, w_L)
    v_R = velocity(rho_R, w_R)
    rho_ij = 0.5 * (rho_L + rho_R)
    v_ij = 0.5 * (v_L + v_R)
    w_ij = v_ij + pressure(rho_ij)
    lam1_ij = v_ij - rho_ij * dpressure(rho_ij)
    lam2_ij = v_ij
    return rho_ij, v_ij, w_ij, lam1_ij, lam2_ij


def wave_type(dv, dw, eps: float = 1e-8):
    """theta = |dw| / (|dw| + |dv| + eps).

    Heuristic indicator of which Riemann wave the jump is on:
      theta ~ 1  -> 2-contact (LD; v continuous, w jumps)
      theta ~ 0  -> 1-wave    (GNL; w continuous, v jumps)
    """
    adv = _abs(dv)
    adw = _abs(dw)
    return adw / (adw + adv + eps)


# --------------------------------------------------------------------------- #
# Convenience: max wave speed (spectral radius) for CFL
# --------------------------------------------------------------------------- #
def spectral_radius(rho, w):
    """max(|lambda1|, |lambda2|) — used for CFL / wave-speed estimates."""
    lam1, lam2 = eigenvalues(rho, w)
    return _maximum(_abs(lam1), _abs(lam2))
