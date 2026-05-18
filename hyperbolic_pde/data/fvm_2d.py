"""2D LWR data generator: WENO5 + Strang dimensional splitting + SSP-RK3.

Solves the isotropic scalar conservation law
    u_t + f(u)_x + g(u)_y = 0,    f(u) = g(u) = u(1-u)
on a Cartesian grid.

Each Strang step is:
    L_x^{dt/2} -> L_y^{dt} -> L_x^{dt/2}
where each L_d sweep is one SSP-RK3 update of the 1D WENO5 + Lax-Friedrichs
operator from ``hyperbolic_pde.data.fvm``.

Public API:
    simulate_lwr_2d(u0, x, y, t_out, ...)            single sample
    piecewise_rectangles_ic_2d(x, y, ...)            IC generator
    generate_dataset_2d(...)                         dataset driver
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from hyperbolic_pde.data.fvm import (
    _apply_ghost_bc,
    _weno5_reconstruct_left,
    flux,
    flux_prime,
)


# --------------------------------------------------------------------------- #
# 1D WENO5 RHS and RK3 step (reused for each axis sweep)
# --------------------------------------------------------------------------- #
def _weno5_rhs_1d(u: np.ndarray, dx: float, boundary: str) -> np.ndarray:
    """L(u) = -(F_{i+1/2} - F_{i-1/2})/dx using WENO5 + global LF splitting.

    Mirrors ``fvm._weno5_rhs`` but kept local so the 2D module is self-contained
    against future signature drift.
    """
    nx = u.size
    ng = 3
    u_ext = _apply_ghost_bc(u, boundary, ng)
    alpha = float(np.max(np.abs(flux_prime(u_ext))))
    alpha = max(alpha, 1e-8)

    f_ext = flux(u_ext)
    fp = 0.5 * (f_ext + alpha * u_ext)
    fm = 0.5 * (f_ext - alpha * u_ext)

    fp_s = fp[0:-1]
    fm_s = fm[1:][::-1]

    fhat_p = _weno5_reconstruct_left(fp_s)
    fhat_m = _weno5_reconstruct_left(fm_s)[::-1]

    fhat = fhat_p + fhat_m
    return -(fhat[1:] - fhat[:-1]) / dx


def _rk3_step_1d(u: np.ndarray, dx: float, dt: float, boundary: str) -> np.ndarray:
    """One SSP-RK3 step of the WENO5 1D operator."""
    L = _weno5_rhs_1d
    u1 = u + dt * L(u, dx, boundary)
    u2 = 0.75 * u + 0.25 * (u1 + dt * L(u1, dx, boundary))
    u_new = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * L(u2, dx, boundary))
    return u_new


def _sweep_x(u: np.ndarray, dx: float, dt: float, boundary: str) -> np.ndarray:
    """Apply 1D RK3 sweep along x for each y-row. ``u`` has shape [ny, nx]."""
    ny, _ = u.shape
    out = np.empty_like(u)
    for j in range(ny):
        out[j] = _rk3_step_1d(u[j], dx, dt, boundary)
    return out


def _sweep_y(u: np.ndarray, dy: float, dt: float, boundary: str) -> np.ndarray:
    """Apply 1D RK3 sweep along y for each x-column."""
    _, nx = u.shape
    out = np.empty_like(u)
    for i in range(nx):
        out[:, i] = _rk3_step_1d(u[:, i], dy, dt, boundary)
    return out


def _strang_step(
    u: np.ndarray, dx: float, dy: float, dt: float, boundary: str,
) -> np.ndarray:
    """One Strang-split step: L_x^{dt/2} L_y^{dt} L_x^{dt/2}."""
    u = _sweep_x(u, dx, 0.5 * dt, boundary)
    u = _sweep_y(u, dy, dt, boundary)
    u = _sweep_x(u, dx, 0.5 * dt, boundary)
    return u


# --------------------------------------------------------------------------- #
# Initial condition generator
# --------------------------------------------------------------------------- #
def piecewise_rectangles_ic_2d(
    x: np.ndarray,
    y: np.ndarray,
    num_rects: int,
    u_min: float,
    u_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random axis-aligned piecewise-constant rectangles on the (y, x) grid.

    Returns ``u0`` of shape ``[ny, nx]``. The background is one constant value
    drawn from ``[u_min, u_max]``; ``num_rects`` overlay rectangles are then
    written on top in random order (later rectangles paint over earlier ones).
    """
    if num_rects < 1:
        raise ValueError("num_rects must be >= 1")
    nx = x.size
    ny = y.size

    u0 = np.full((ny, nx), rng.uniform(u_min, u_max), dtype=np.float32)
    for _ in range(num_rects):
        i0 = int(rng.integers(0, nx))
        i1 = int(rng.integers(0, nx))
        j0 = int(rng.integers(0, ny))
        j1 = int(rng.integers(0, ny))
        if i0 > i1:
            i0, i1 = i1, i0
        if j0 > j1:
            j0, j1 = j1, j0
        u0[j0:j1 + 1, i0:i1 + 1] = rng.uniform(u_min, u_max)
    return u0


# --------------------------------------------------------------------------- #
# Single-sample simulation driver
# --------------------------------------------------------------------------- #
def simulate_lwr_2d(
    u0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t_out: np.ndarray,
    cfl: float = 0.4,
    boundary: str = "periodic",
) -> np.ndarray:
    """Simulate 2D LWR with WENO5 + Strang splitting + RK3.

    Parameters
    ----------
    u0 : [ny, nx] initial state.
    x  : [nx] uniform grid.
    y  : [ny] uniform grid.
    t_out : [nt] increasing output times, t_out[0] = 0.
    cfl : Courant number for adaptive sub-stepping (2D bound used).
    boundary : "periodic", "ghost", or "fixed" (forwarded to 1D sweep).

    Returns
    -------
    u_hist : [nt, ny, nx] float32 history at the requested output times.
    """
    if u0.ndim != 2:
        raise ValueError(f"u0 must be 2D [ny, nx], got shape {u0.shape}")
    ny, nx = u0.shape
    if y.size != ny or x.size != nx:
        raise ValueError(
            f"grid mismatch: u0 is [{ny}, {nx}] but y={y.size}, x={x.size}"
        )
    if t_out[0] != 0.0:
        raise ValueError("t_out[0] must be 0.0")

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    t_max = float(t_out[-1])
    nt = t_out.size

    u = u0.astype(np.float64).copy()
    u_hist = np.zeros((nt, ny, nx), dtype=np.float32)
    u_hist[0] = u.astype(np.float32)

    t = 0.0
    k = 1
    while k < nt:
        amax = float(np.max(np.abs(flux_prime(u))))
        amax = max(1e-6, amax)
        # 2D CFL: dt * (amax/dx + amax/dy) <= cfl  (matches the LF / RK3 bound).
        dt = cfl / (amax / dx + amax / dy)
        dt = min(dt, t_max - t)

        u_prev = u.copy()
        t_prev = t
        u = _strang_step(u, dx, dy, dt, boundary)
        if not np.isfinite(u).all():
            raise RuntimeError(
                f"WENO5/Strang solver diverged at t={t:.4f} (non-finite values)"
            )
        t += dt

        while k < nt and t >= t_out[k] - 1e-12:
            alpha = (t_out[k] - t_prev) / (t - t_prev) if t > t_prev else 1.0
            u_hist[k] = (u_prev + alpha * (u - u_prev)).astype(np.float32)
            k += 1

    return u_hist


# --------------------------------------------------------------------------- #
# Dataset driver
# --------------------------------------------------------------------------- #
@dataclass
class Dataset2DBundle:
    x: np.ndarray         # [nx]
    y: np.ndarray         # [ny]
    t: np.ndarray         # [nt]
    u: np.ndarray         # [N, nt, ny, nx]
    u0: np.ndarray        # [N, ny, nx]


def generate_dataset_2d(
    num_samples: int,
    nx: int,
    ny: int,
    nt: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    t_max: float,
    cfl: float,
    num_rects: int | tuple[int, int],
    u_min: float,
    u_max: float,
    boundary: str = "periodic",
    seed: int = 42,
) -> Dataset2DBundle:
    """Generate a 2D LWR dataset with piecewise-rectangle ICs.

    ``num_rects`` can be a single int or a (lo, hi) range sampled per sample.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    y = np.linspace(y_min, y_max, ny, dtype=np.float32)
    t = np.linspace(0.0, t_max, nt, dtype=np.float32)

    if isinstance(num_rects, int):
        rect_range = (num_rects, num_rects)
    else:
        rect_range = (int(num_rects[0]), int(num_rects[1]))

    u_all = np.zeros((num_samples, nt, ny, nx), dtype=np.float32)
    u0_all = np.zeros((num_samples, ny, nx), dtype=np.float32)

    for i in range(num_samples):
        r = int(rng.integers(rect_range[0], rect_range[1] + 1))
        u0 = piecewise_rectangles_ic_2d(x, y, r, u_min, u_max, rng)
        u_all[i] = simulate_lwr_2d(u0, x, y, t, cfl=cfl, boundary=boundary)
        u0_all[i] = u0

    return Dataset2DBundle(x=x, y=y, t=t, u=u_all, u0=u0_all)


def save_dataset_2d(bundle: Dataset2DBundle, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=bundle.x,
        y=bundle.y,
        t=bundle.t,
        u=bundle.u,
        u0=bundle.u0,
    )


def load_dataset_2d(path: Path) -> Dataset2DBundle:
    path = Path(path)
    with np.load(path) as data:
        return Dataset2DBundle(
            x=data["x"], y=data["y"], t=data["t"],
            u=data["u"], u0=data["u0"],
        )
