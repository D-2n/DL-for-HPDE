"""Standalone component WENO5-Z + clip-PP solver for the ARZ traffic model.

This single file reproduces the ``weno5_clip`` scheme used in the paper's ARZ
scheme comparison -- i.e. exactly
``reference_arz.solve_arz_weno5(variant="component", pp_limiter="clip")`` --
with NO dependencies beyond NumPy, so it can be dropped into any project.

Scheme
------
* Spatial: component-wise WENO5 with **WENO-Z** nonlinear weights and global
  Lax-Friedrichs flux splitting on the conserved variables (rho, y = rho*w).
* Time: SSP-RK3.
* Positivity ("clip" PP limiter): after every RK stage the state is projected
  back into the physical box rho in [RHO_MIN, rho_cap], v in [v_min, v_max]
  ONLY when a stage leaves it (otherwise the high-order update is untouched).
* Source (relaxation): exact Strang splitting -- a half relaxation step, the
  hyperbolic SSP-RK3 step, then a second half relaxation step. Pass tau=inf
  (default) for the homogeneous system.

The ARZ system in conserved form is
    d_t rho + d_x (rho v)     = 0
    d_t y   + d_x (y   v)     = (y_eq(rho) - y) / tau ,     y = rho * w,
with primitive velocity v = w - p(rho) and the Greenshields equilibrium
w_eq(rho) = (1 - rho) + p(rho). Two pressure closures are supported:
    p(rho) = rho            ("rho",      project default)
    p(rho) = rho + rho^2    ("rho+rho2")

API
---
    rho_hist, w_hist = solve_arz_weno5_clip(
        rho0, w0, x_min, x_max, t_max, nt_out, tau=np.inf, cfl=0.2,
        boundary="ghost",
    )
Both histories have shape (nt_out, nx), float32. ``solve_arz_weno5_clip``
raises ``WenoInstabilityError`` if the run goes non-finite / leaves the box
(the same policy the dataset generator uses to skip an IC).

CLI demo
--------
    python weno5_clip_arz_standalone.py                 # Riemann demo
    python weno5_clip_arz_standalone.py --nx 256 --cfl 0.2 --pressure-form rho
"""
from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
RHO_MIN = 1e-6            # vacuum guard inside y = rho*w divisions
RHO_CAP_DEFAULT = 1.0 - 1e-6
V_MIN_DEFAULT = 0.0
V_MAX_DEFAULT = 1.0
_WENO_EPS = 1e-6

_P_FORM = "rho"
_VALID_P_FORMS = ("rho", "rho+rho2")


class WenoInstabilityError(RuntimeError):
    """Raised when WENO5 produces non-finite states or leaves the physical box."""


# --------------------------------------------------------------------------- #
# Pressure closure (single point of control)
# --------------------------------------------------------------------------- #
def set_pressure_form(form: str) -> None:
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
        return rho                       # rho * 1
    return rho + rho * rho * rho


def velocity(rho, w):
    """Primitive velocity v = w - p(rho)."""
    return w - pressure(rho)


def eigenvalues(rho, w):
    """(lambda1, lambda2) = (v - rho*p'(rho), v)."""
    v = velocity(rho, w)
    return v - rho * dpressure(rho), v


def spectral_radius(rho, w):
    lam1, lam2 = eigenvalues(rho, w)
    return np.maximum(np.abs(lam1), np.abs(lam2))


# --------------------------------------------------------------------------- #
# Ghost boundary + WENO5-Z reconstruction
# --------------------------------------------------------------------------- #
def _apply_ghost_bc(u: np.ndarray, boundary: str, ng: int) -> np.ndarray:
    """Pad u with ng ghost cells per side."""
    nx = u.size
    out = np.empty(nx + 2 * ng, dtype=u.dtype)
    out[ng:ng + nx] = u
    if boundary == "periodic":
        out[:ng] = u[-ng:]
        out[ng + nx:] = u[:ng]
    elif boundary in ("ghost", "fixed"):
        out[:ng] = u[0]
        out[ng + nx:] = u[-1]
    else:
        raise ValueError(
            f"boundary must be 'periodic'|'ghost'|'fixed', got {boundary!r}"
        )
    return out


def _weno5_reconstruct_left(v: np.ndarray) -> np.ndarray:
    """Fifth-order WENO reconstruction at each cell's RIGHT interface from the
    left-biased stencil, using **WENO-Z** weights (robust on rough stencils).

    Input  v : [n] cell-centred values (already ghost-padded by the caller).
    Output   : [n - 4] reconstructed interface values.
    """
    vm2 = v[0:-4]
    vm1 = v[1:-3]
    v0 = v[2:-2]
    vp1 = v[3:-1]
    vp2 = v[4:]

    p0 = (1.0 / 3.0) * vm2 - (7.0 / 6.0) * vm1 + (11.0 / 6.0) * v0
    p1 = -(1.0 / 6.0) * vm1 + (5.0 / 6.0) * v0 + (1.0 / 3.0) * vp1
    p2 = (1.0 / 3.0) * v0 + (5.0 / 6.0) * vp1 - (1.0 / 6.0) * vp2

    b0 = (13.0 / 12.0) * (vm2 - 2.0 * vm1 + v0) ** 2 + 0.25 * (vm2 - 4.0 * vm1 + 3.0 * v0) ** 2
    b1 = (13.0 / 12.0) * (vm1 - 2.0 * v0 + vp1) ** 2 + 0.25 * (vm1 - vp1) ** 2
    b2 = (13.0 / 12.0) * (v0 - 2.0 * vp1 + vp2) ** 2 + 0.25 * (3.0 * v0 - 4.0 * vp1 + vp2) ** 2

    tau5 = (b0 + b1 + b2) + _WENO_EPS
    d0, d1, d2 = 0.1, 0.6, 0.3
    a0 = d0 * (1.0 + (tau5 / (_WENO_EPS + b0)) ** 2)
    a1 = d1 * (1.0 + (tau5 / (_WENO_EPS + b1)) ** 2)
    a2 = d2 * (1.0 + (tau5 / (_WENO_EPS + b2)) ** 2)
    asum = a0 + a1 + a2
    lin = d0 + d1 + d2
    w0 = np.where(asum > 1e-300, a0 / asum, d0 / lin)
    w1 = np.where(asum > 1e-300, a1 / asum, d1 / lin)
    w2 = np.where(asum > 1e-300, a2 / asum, d2 / lin)
    return w0 * p0 + w1 * p1 + w2 * p2


# --------------------------------------------------------------------------- #
# Component WENO5 fluxes + clip positivity limiter
# --------------------------------------------------------------------------- #
def _weno5_component_fluxes(
    rho: np.ndarray, y: np.ndarray, boundary: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """WENO5-Z interface fluxes on (rho, y) with global LF splitting."""
    rho_safe = np.maximum(rho, RHO_MIN)
    w = y / rho_safe
    v = w - pressure(rho)
    lam1 = v - rho * dpressure(rho)
    lam2 = v
    alpha = max(float(np.max(np.maximum(np.abs(lam1), np.abs(lam2)))), 1e-8)

    def _iface_flux(u, f):
        ng = 3
        u_ext = _apply_ghost_bc(u, boundary, ng)
        f_ext = _apply_ghost_bc(f, boundary, ng)
        fp = 0.5 * (f_ext + alpha * u_ext)
        fm = 0.5 * (f_ext - alpha * u_ext)
        fhat_p = _weno5_reconstruct_left(fp[0:-1])
        fhat_m = _weno5_reconstruct_left(fm[1:][::-1])[::-1]
        return fhat_p + fhat_m

    F1 = rho_safe * v
    F2 = y * v
    return _iface_flux(rho, F1), _iface_flux(y, F2)


def _state_needs_bound(
    rho: np.ndarray, y: np.ndarray,
    *, rho_cap: float, v_min: float, v_max: float, tol: float = 1e-12,
) -> bool:
    if not (np.isfinite(rho).all() and np.isfinite(y).all()):
        return True
    rho_safe = np.maximum(rho, RHO_MIN)
    v = velocity(rho_safe, y / rho_safe)
    return (
        float(rho.min()) < RHO_MIN - tol
        or float(rho.max()) > rho_cap + tol
        or float(v.min()) < v_min - tol
        or float(v.max()) > v_max + tol
    )


def _bound_arz_state(
    rho: np.ndarray, y: np.ndarray,
    *, rho_cap: float, v_min: float, v_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project (rho, y) to the physical box (clip PP limiter)."""
    if not _state_needs_bound(rho, y, rho_cap=rho_cap, v_min=v_min, v_max=v_max):
        return rho, y
    rho_in = np.asarray(rho, dtype=np.float64)
    y_in = np.asarray(y, dtype=np.float64)
    rho_safe_in = np.maximum(rho_in, RHO_MIN)
    v_in = velocity(rho_safe_in, y_in / rho_safe_in)
    rho_out = np.clip(rho_in, RHO_MIN, rho_cap)
    rho_safe_out = np.maximum(rho_out, RHO_MIN)
    v_out = np.clip(v_in, v_min, v_max)
    w_out = v_out + pressure(rho_safe_out)
    return rho_out, rho_safe_out * w_out


def _flux_divergence_update(rho, y, F1, F2, dx, dt):
    lam = dt / dx
    return rho - lam * (F1[1:] - F1[:-1]), y - lam * (F2[1:] - F2[:-1])


def _hyperbolic_step(rho, y, dx, dt, boundary, *, rho_cap, v_min, v_max):
    """One WENO5 hyperbolic substep with clip bounding on the update."""
    F1, F2 = _weno5_component_fluxes(rho, y, boundary)
    rho_new, y_new = _flux_divergence_update(rho, y, F1, F2, dx, dt)
    if _state_needs_bound(rho_new, y_new, rho_cap=rho_cap, v_min=v_min, v_max=v_max):
        rho_new, y_new = _bound_arz_state(rho_new, y_new, rho_cap=rho_cap, v_min=v_min, v_max=v_max)
    return rho_new, y_new


def _ssp_rk3_step(rho, y, dx, dt, boundary, *, rho_cap, v_min, v_max):
    """SSP-RK3 hyperbolic step; clip PP limiter applied after every stage."""
    bk = dict(rho_cap=rho_cap, v_min=v_min, v_max=v_max)

    def _advance(rho_, y_):
        return _hyperbolic_step(rho_, y_, dx, dt, boundary, **bk)

    def _finish(rho_, y_):
        return _bound_arz_state(rho_, y_, **bk)

    rho1, y1 = _finish(*_advance(rho, y))
    h_r, h_y = _advance(rho1, y1)
    rho2, y2 = _finish(0.75 * rho + 0.25 * h_r, 0.75 * y + 0.25 * h_y)
    h_r, h_y = _advance(rho2, y2)
    return _finish((1.0 / 3.0) * rho + (2.0 / 3.0) * h_r,
                   (1.0 / 3.0) * y + (2.0 / 3.0) * h_y)


# --------------------------------------------------------------------------- #
# Physics validation
# --------------------------------------------------------------------------- #
def validate_arz_physics(
    rho_hist, w_hist,
    *, rho_min_allowed: float = 0.0, rho_max_allowed: float = 1.0 + 1e-2,
    v_min: float = V_MIN_DEFAULT, v_max: float = V_MAX_DEFAULT, tol: float = 1e-3,
) -> None:
    """Raise WenoInstabilityError if a trajectory leaves the physical ARZ box."""
    v = velocity(rho_hist, w_hist)
    rho_max = float(np.nanmax(rho_hist))
    rho_min = float(np.nanmin(rho_hist))
    if rho_max > rho_max_allowed + tol or rho_min < rho_min_allowed - tol:
        raise WenoInstabilityError(
            f"rho out of range: [{rho_min:.4f}, {rho_max:.4f}] "
            f"(allowed [{rho_min_allowed}, {rho_max_allowed}])"
        )
    if float(np.nanmax(v)) > v_max + tol or float(np.nanmin(v)) < v_min - tol:
        raise WenoInstabilityError(
            f"v out of range: [{float(np.nanmin(v)):.4f}, {float(np.nanmax(v)):.4f}] "
            f"(expected [{v_min}, {v_max}])"
        )


# --------------------------------------------------------------------------- #
# Main solver
# --------------------------------------------------------------------------- #
def solve_arz_weno5_clip(
    rho0, w0, x_min: float, x_max: float, t_max: float,
    nt_out: int, tau: float = np.inf, cfl: float = 0.2, boundary: str = "ghost",
    *,
    rho_cap: float = RHO_CAP_DEFAULT,
    v_min: float = V_MIN_DEFAULT,
    v_max: float = V_MAX_DEFAULT,
    abort_rho_max: float = 1.05,
    max_substeps: int = 100_000,
    validate_physics: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Component WENO5-Z + clip-PP + SSP-RK3 with Strang-split exact relaxation.

    Parameters
    ----------
    rho0, w0 : (nx,) initial density and w = v + p(rho).
    x_min, x_max, t_max : domain extents (cells are [x_min, x_max]/nx wide).
    nt_out : number of output snapshots (including t=0).
    tau : relaxation time; np.inf for the homogeneous (source-free) system.
    cfl : CFL number (WENO5 is stable to ~0.4; 0.2 is conservative / dataset default).
    boundary : "ghost" (zero-gradient), "fixed", or "periodic".

    Returns (rho_hist, w_hist), each (nt_out, nx) float32.
    Raises WenoInstabilityError on non-finite states or box violations.
    """
    rho0 = np.asarray(rho0, dtype=np.float64)
    w0 = np.asarray(w0, dtype=np.float64)
    nx = rho0.size
    dx = (x_max - x_min) / nx
    t_out = np.linspace(0.0, t_max, nt_out, dtype=np.float64)
    tau_eff = tau if np.isfinite(tau) else 1e12

    rho = np.maximum(rho0.copy(), RHO_MIN)
    y = rho * w0
    rho, y = _bound_arz_state(rho, y, rho_cap=rho_cap, v_min=v_min, v_max=v_max)

    rho_hist = np.empty((nt_out, nx), dtype=np.float32)
    w_hist = np.empty((nt_out, nx), dtype=np.float32)
    rho_hist[0] = rho.astype(np.float32)
    w_hist[0] = (y / np.maximum(rho, RHO_MIN)).astype(np.float32)

    def _check(stage: str) -> None:
        if not (np.isfinite(rho).all() and np.isfinite(y).all()):
            raise WenoInstabilityError(f"non-finite (rho, y) after {stage}")
        if abort_rho_max is not None and float(rho.max()) > abort_rho_max:
            raise WenoInstabilityError(
                f"rho overflow after {stage}: max rho={float(rho.max()):.3f} > {abort_rho_max}"
            )

    t = 0.0
    k = 1
    n_sub = 0
    while k < nt_out:
        n_sub += 1
        if n_sub > max_substeps:
            raise WenoInstabilityError(
                f"exceeded max_substeps={max_substeps} at t={t:.4f} snapshot {k}/{nt_out}"
            )
        w_now = y / np.maximum(rho, RHO_MIN)
        lam_max = max(float(spectral_radius(rho, w_now).max()), 1e-6)
        if not np.isfinite(lam_max):
            raise WenoInstabilityError("non-finite wave speed")
        dt = min(cfl * dx / lam_max, t_out[k] - t)
        if not np.isfinite(dt) or dt <= 0:
            raise WenoInstabilityError(f"invalid dt={dt} at t={t:.4f} snapshot {k}/{nt_out}")
        # Strang: half-step source -> WENO hyperbolic -> half-step source.
        y_eq_ = y_eq(rho)
        y = y_eq_ + (y - y_eq_) * np.exp(-0.5 * dt / tau_eff)
        rho, y = _ssp_rk3_step(rho, y, dx, dt, boundary, rho_cap=rho_cap, v_min=v_min, v_max=v_max)
        _check("WENO step")
        y_eq_ = y_eq(rho)
        y = y_eq_ + (y - y_eq_) * np.exp(-0.5 * dt / tau_eff)
        rho, y = _bound_arz_state(rho, y, rho_cap=rho_cap, v_min=v_min, v_max=v_max)
        _check("relaxation")
        t += dt
        while k < nt_out and t >= t_out[k] - 1e-12:
            rho_hist[k] = rho.astype(np.float32)
            w_hist[k] = (y / np.maximum(rho, RHO_MIN)).astype(np.float32)
            k += 1
    if validate_physics:
        validate_arz_physics(rho_hist, w_hist, v_min=v_min, v_max=v_max)
    return rho_hist, w_hist


# --------------------------------------------------------------------------- #
# CLI demo: a single Riemann problem (homogeneous ARZ, tau = inf)
# --------------------------------------------------------------------------- #
def _demo(args: argparse.Namespace) -> None:
    set_pressure_form(args.pressure_form)
    nx, nt = args.nx, args.nt
    x_min, x_max, t_max = args.x_min, args.x_max, args.t_max
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, nx)

    # Left/right primitive states (rho, v); w = v + p(rho).
    rho0 = np.where(x < 0.5 * (x_min + x_max), args.rho_l, args.rho_r)
    v0 = np.where(x < 0.5 * (x_min + x_max), args.v_l, args.v_r)
    w0 = v0 + pressure(rho0)

    rho_h, w_h = solve_arz_weno5_clip(
        rho0, w0, x_min, x_max, t_max, nt_out=nt, tau=args.tau, cfl=args.cfl,
        boundary=args.boundary,
    )
    v_h = w_h - pressure(rho_h)
    print(f"pressure_form = {get_pressure_form()}  tau = {args.tau}  cfl = {args.cfl}")
    print(f"rho_hist shape = {rho_h.shape}  w_hist shape = {w_h.shape}")
    print(f"rho range: [{rho_h.min():.4f}, {rho_h.max():.4f}]  "
          f"v range: [{v_h.min():.4f}, {v_h.max():.4f}]")
    print(f"final-time rho[::{max(1, nx // 8)}]: "
          f"{np.round(rho_h[-1, ::max(1, nx // 8)], 3)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Component WENO5-Z + clip-PP ARZ solver (weno5_clip). "
                    "Runs a Riemann demo; import solve_arz_weno5_clip to use on your own ICs."
    )
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--x-min", type=float, default=-1.0)
    ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--t-max", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=float("inf"))
    ap.add_argument("--cfl", type=float, default=0.2)
    ap.add_argument("--boundary", type=str, default="ghost",
                    choices=["ghost", "fixed", "periodic"])
    ap.add_argument("--pressure-form", type=str, default="rho",
                    choices=list(_VALID_P_FORMS))
    ap.add_argument("--rho-l", type=float, default=0.3)
    ap.add_argument("--rho-r", type=float, default=0.7)
    ap.add_argument("--v-l", type=float, default=0.6)
    ap.add_argument("--v-r", type=float, default=0.2)
    _demo(ap.parse_args())


if __name__ == "__main__":
    main()
