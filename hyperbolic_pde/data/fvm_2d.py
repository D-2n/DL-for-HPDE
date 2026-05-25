"""2D scalar conservation law data generator: WENO5 + Strang split + SSP-RK3.

Solves
    u_t + f(u)_x + g(u)_y = 0
on a Cartesian grid, with the flux pair (f, g) supplied at call time. The
existing LWR case f(u) = g(u) = u(1-u) is preserved via the
``simulate_lwr_2d`` wrapper. The new ``simulate_weno5_2d`` takes arbitrary
flux pairs, and ``generate_dataset_upsampled_2d`` implements the up-sampled
WENO5 + cell-average down-sampling pipeline used as ground truth across the
2D v0 (linear) and v1 (nonlinear) HypNO models.

Public API:
    simulate_lwr_2d(u0, x, y, t_out, ...)            single LWR sample (legacy)
    simulate_weno5_2d(u0, x, y, t_out, flux_x, flux_y, ...)   flux-parametric
    piecewise_rectangles_ic_2d(x, y, ...)            IC generator
    generate_dataset_2d(...)                         legacy LWR dataset driver
    generate_dataset_upsampled_2d(...)               fine -> cell-average pipeline
    analytical_shift_2d_rects(...)                   exact linear-advection GT
    upwind_flux_x / upwind_flux_y                    standalone helpers for lifting
    LWR_FLUX, linear_flux(speed)                     FluxPair builders
"""
from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

from hyperbolic_pde.data.fvm import (
    flux,
    flux_prime,
)

_WENO_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Flux abstraction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FluxPair:
    """Scalar flux f(u) and its derivative f'(u), as a single bundle."""
    fn: Callable[[np.ndarray], np.ndarray]
    fprime: Callable[[np.ndarray], np.ndarray]


LWR_FLUX = FluxPair(flux, flux_prime)
"""LWR scalar flux: f(u) = u(1 - u), f'(u) = 1 - 2u."""


def linear_flux(speed: float) -> FluxPair:
    """Linear constant-coefficient flux F(u) = speed * u (constant derivative).

    Used for v0 2D linear advection: x-flux uses speed=a, y-flux uses speed=b.
    """
    s = float(speed)

    def _fn(u: np.ndarray) -> np.ndarray:
        return s * u

    def _fprime(u: np.ndarray) -> np.ndarray:
        return np.full_like(u, s)

    return FluxPair(fn=_fn, fprime=_fprime)


# --------------------------------------------------------------------------- #
# Standalone upwind helpers (lifting layer's numerical-flux features)
# --------------------------------------------------------------------------- #
def upwind_flux_x(u: np.ndarray, a: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (F_left, F_right) face fluxes for the x-direction upwind scheme.

    For constant a:
        F^x_{i-1/2, j} = a * u_{i-1, j}  if a > 0  else  a * u_{i, j}
        F^x_{i+1/2, j} = a * u_{i, j}    if a > 0  else  a * u_{i+1, j}

    Boundary: zero-order (replicate) extrapolation, matching the model's
    ghost-BC convention. Shapes both [..., ny, nx].

    Used only by the model's lifting layer to compute IC numerical-flux node
    features. The data generator does not call this.
    """
    if a >= 0:
        u_left_neighbor = np.concatenate([u[..., :, :1], u[..., :, :-1]], axis=-1)
        return a * u_left_neighbor, a * u
    u_right_neighbor = np.concatenate([u[..., :, 1:], u[..., :, -1:]], axis=-1)
    return a * u, a * u_right_neighbor


def upwind_flux_y(u: np.ndarray, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (F_below, F_above) face fluxes for the y-direction upwind scheme.

    Convention: u has shape [..., ny, nx] with y increasing along axis -2.
    Boundary: replicate.
    """
    if b >= 0:
        u_below_neighbor = np.concatenate([u[..., :1, :], u[..., :-1, :]], axis=-2)
        return b * u_below_neighbor, b * u
    u_above_neighbor = np.concatenate([u[..., 1:, :], u[..., -1:, :]], axis=-2)
    return b * u, b * u_above_neighbor


# --------------------------------------------------------------------------- #
# Vectorised WENO5 RHS and RK3 step (all rows/columns processed at once)
# --------------------------------------------------------------------------- #
def _apply_ghost_bc_nd(u: np.ndarray, boundary: str, ng: int) -> np.ndarray:
    """Pad the last axis of ``u`` with ``ng`` ghost cells."""
    pad = [(0, 0)] * (u.ndim - 1) + [(ng, ng)]
    if boundary == "periodic":
        return np.pad(u, pad, mode="wrap")
    elif boundary in ("ghost", "fixed"):
        return np.pad(u, pad, mode="edge")
    else:
        raise ValueError(f"Unknown boundary: {boundary!r}")


def _weno5_reconstruct_left_nd(v: np.ndarray) -> np.ndarray:
    """WENO5 left reconstruction along the last axis. ``v: [..., n+4] -> [..., n+1]``."""
    vm2 = v[..., 0:-4]
    vm1 = v[..., 1:-3]
    v0  = v[..., 2:-2]
    vp1 = v[..., 3:-1]
    vp2 = v[..., 4:]

    p0 = ( 1.0 / 3.0) * vm2 - (7.0 / 6.0) * vm1 + (11.0 / 6.0) * v0
    p1 = (-1.0 / 6.0) * vm1 + (5.0 / 6.0) * v0  + ( 1.0 / 3.0) * vp1
    p2 = ( 1.0 / 3.0) * v0  + (5.0 / 6.0) * vp1 - ( 1.0 / 6.0) * vp2

    b0 = (13.0 / 12.0) * (vm2 - 2.0 * vm1 + v0 ) ** 2 + 0.25 * (vm2 - 4.0 * vm1 + 3.0 * v0 ) ** 2
    b1 = (13.0 / 12.0) * (vm1 - 2.0 * v0  + vp1) ** 2 + 0.25 * (vm1 - vp1) ** 2
    b2 = (13.0 / 12.0) * (v0  - 2.0 * vp1 + vp2) ** 2 + 0.25 * (3.0 * v0 - 4.0 * vp1 + vp2) ** 2

    a0 = 0.1 / (_WENO_EPS + b0) ** 2
    a1 = 0.6 / (_WENO_EPS + b1) ** 2
    a2 = 0.3 / (_WENO_EPS + b2) ** 2
    asum = a0 + a1 + a2
    return (a0 * p0 + a1 * p1 + a2 * p2) / asum


def _weno5_rhs_nd(
    u: np.ndarray, dx: float, boundary: str,
    flux_pair: FluxPair = LWR_FLUX,
) -> np.ndarray:
    """WENO5 + global LF spatial operator along the last axis. ``u: [..., nx]``."""
    ng = 3
    u_ext = _apply_ghost_bc_nd(u, boundary, ng)
    alpha = float(np.max(np.abs(flux_pair.fprime(u_ext))))
    alpha = max(alpha, 1e-8)

    f_ext = flux_pair.fn(u_ext)
    fp = 0.5 * (f_ext + alpha * u_ext)
    fm = 0.5 * (f_ext - alpha * u_ext)

    fp_s = fp[..., 0:-1]
    fm_s = fm[..., 1:][..., ::-1]

    fhat_p = _weno5_reconstruct_left_nd(fp_s)
    fhat_m = _weno5_reconstruct_left_nd(fm_s)[..., ::-1]
    fhat = fhat_p + fhat_m
    return -(fhat[..., 1:] - fhat[..., :-1]) / dx


def _rk3_step_nd(
    u: np.ndarray, dx: float, dt: float, boundary: str,
    flux_pair: FluxPair = LWR_FLUX,
) -> np.ndarray:
    """SSP-RK3 step of the WENO5 operator along the last axis."""
    def L(v: np.ndarray) -> np.ndarray:
        return _weno5_rhs_nd(v, dx, boundary, flux_pair)

    u1 = u + dt * L(u)
    u2 = 0.75 * u + 0.25 * (u1 + dt * L(u1))
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * L(u2))


def _sweep_x(
    u: np.ndarray, dx: float, dt: float, boundary: str,
    flux_pair: FluxPair = LWR_FLUX,
) -> np.ndarray:
    """RK3 sweep along x for all y-rows simultaneously. ``u: [ny, nx]``."""
    return _rk3_step_nd(u, dx, dt, boundary, flux_pair)


def _sweep_y(
    u: np.ndarray, dy: float, dt: float, boundary: str,
    flux_pair: FluxPair = LWR_FLUX,
) -> np.ndarray:
    """RK3 sweep along y for all x-columns simultaneously. ``u: [ny, nx]``."""
    uT = np.ascontiguousarray(u.T)   # [nx, ny] — contiguous for cache efficiency
    return _rk3_step_nd(uT, dy, dt, boundary, flux_pair).T


def _strang_step(
    u: np.ndarray, dx: float, dy: float, dt: float, boundary: str,
    flux_x: FluxPair = LWR_FLUX, flux_y: FluxPair = LWR_FLUX,
) -> np.ndarray:
    """One Strang-split step: L_x^{dt/2} L_y^{dt} L_x^{dt/2}.

    Anisotropic flux is supported by passing different `flux_x` and `flux_y`.
    For the isotropic LWR case both default to `LWR_FLUX`.
    """
    u = _sweep_x(u, dx, 0.5 * dt, boundary, flux_x)
    u = _sweep_y(u, dy, dt, boundary, flux_y)
    u = _sweep_x(u, dx, 0.5 * dt, boundary, flux_x)
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
    """Simulate 2D LWR with WENO5 + Strang splitting + RK3 (legacy wrapper)."""
    return simulate_weno5_2d(
        u0, x, y, t_out,
        flux_x=LWR_FLUX, flux_y=LWR_FLUX,
        cfl=cfl, boundary=boundary,
    )


def simulate_weno5_2d(
    u0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t_out: np.ndarray,
    *,
    flux_x: FluxPair,
    flux_y: FluxPair,
    cfl: float = 0.4,
    boundary: str = "periodic",
) -> np.ndarray:
    """Simulate 2D scalar conservation law with WENO5 + Strang split + SSP-RK3.

    Parameters
    ----------
    u0 : [ny, nx] initial state.
    x  : [nx] uniform grid.
    y  : [ny] uniform grid.
    t_out : [nt] increasing output times, t_out[0] = 0.
    flux_x, flux_y : FluxPair, x- and y-direction fluxes (can differ).
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
        a_x = float(np.max(np.abs(flux_x.fprime(u))))
        a_y = float(np.max(np.abs(flux_y.fprime(u))))
        a_x = max(1e-6, a_x)
        a_y = max(1e-6, a_y)
        # 2D CFL: dt * (a_x/dx + a_y/dy) <= cfl  (matches the LF / RK3 bound).
        dt = cfl / (a_x / dx + a_y / dy)
        dt = min(dt, t_max - t)

        u_prev = u.copy()
        t_prev = t
        u = _strang_step(u, dx, dy, dt, boundary, flux_x=flux_x, flux_y=flux_y)
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


# --------------------------------------------------------------------------- #
# Up-sampled WENO5 + cell-average ground-truth pipeline (v0 linear / v1 LWR)
# --------------------------------------------------------------------------- #
def cell_average_to_grid(u_fine: np.ndarray, factor: int) -> np.ndarray:
    """Cell-average a fine spatial grid down by ``factor`` along the last two axes.

    Accepts shape (..., U*ny, U*nx); returns (..., ny, nx) where U = factor.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return u_fine.copy()
    *batch, fy, fx = u_fine.shape
    if fy % factor != 0 or fx % factor != 0:
        raise ValueError(
            f"fine grid {fy}x{fx} not divisible by factor {factor}"
        )
    ny, nx = fy // factor, fx // factor
    reshaped = u_fine.reshape(*batch, ny, factor, nx, factor)
    return reshaped.mean(axis=(-3, -1))


@dataclass(frozen=True)
class RectSpec:
    """Single piecewise-constant rectangle in *physical* (x, y) coords.

    `x_lo, x_hi` are the rectangle's x-extent (half-open `[x_lo, x_hi)` in spirit
    but enforced by cell-index rasterisation), and similarly in y. Used for the
    parametric rectangle IC so we can re-rasterise at any grid resolution
    without rounding to a single fixed grid.
    """
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    value: float


@dataclass(frozen=True)
class SampleSpec:
    """Parametric per-sample IC description: background + ordered rectangles."""
    background: float
    rects: tuple[RectSpec, ...]


def _sample_rectangle_ic_spec(
    x_min: float, x_max: float, y_min: float, y_max: float,
    num_rects: int, u_min: float, u_max: float,
    rng: np.random.Generator,
) -> SampleSpec:
    """Sample a SampleSpec analogous to ``piecewise_rectangles_ic_2d`` but
    expressed in physical coordinates (resolution-independent)."""
    bg = float(rng.uniform(u_min, u_max))
    rects = []
    for _ in range(num_rects):
        a, b = float(rng.uniform(x_min, x_max)), float(rng.uniform(x_min, x_max))
        c, d = float(rng.uniform(y_min, y_max)), float(rng.uniform(y_min, y_max))
        xl, xh = (a, b) if a <= b else (b, a)
        yl, yh = (c, d) if c <= d else (d, c)
        v = float(rng.uniform(u_min, u_max))
        rects.append(RectSpec(xl, xh, yl, yh, v))
    return SampleSpec(background=bg, rects=tuple(rects))


def rasterise_sample_spec(
    spec: SampleSpec, x: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    """Rasterise a SampleSpec onto a grid via cell-centre point sampling.

    Returns ``u0`` of shape ``[ny, nx]``. Later rectangles overwrite earlier
    ones (same as ``piecewise_rectangles_ic_2d``). Cell-centre sampling means
    rectangle edges land at the nearest cell boundary in either direction.
    """
    ny, nx = y.size, x.size
    u0 = np.full((ny, nx), spec.background, dtype=np.float32)
    for r in spec.rects:
        i_lo = int(np.searchsorted(x, r.x_lo, side="left"))
        i_hi = int(np.searchsorted(x, r.x_hi, side="right"))
        j_lo = int(np.searchsorted(y, r.y_lo, side="left"))
        j_hi = int(np.searchsorted(y, r.y_hi, side="right"))
        if i_lo >= i_hi or j_lo >= j_hi:
            continue
        u0[j_lo:j_hi, i_lo:i_hi] = np.float32(r.value)
    return u0


def analytical_shift_2d_rects(
    spec: SampleSpec,
    x: np.ndarray,
    y: np.ndarray,
    t_out: np.ndarray,
    a: float, b: float,
    boundary: str = "ghost",
) -> np.ndarray:
    """Exact analytical solution for linear advection u_t + a u_x + b u_y = 0.

    Method of characteristics: u(x_i, y_j, t_k) = u_0(x_i - a*t_k, y_j - b*t_k)
    with ghost-BC clamping on the back-shifted point. The IC is the SampleSpec
    (sharp rectangle edges preserved infinitely under the true entropy
    solution), so no time-cumulative diffusion. Returns [nt, ny, nx].

    Only ``boundary="ghost"`` is supported here — periodic would just be a
    modulo, but v0 is locked to ghost.
    """
    if boundary != "ghost":
        raise NotImplementedError(
            "analytical_shift_2d_rects currently supports boundary='ghost' only"
        )
    nt = t_out.size
    out = np.zeros((nt, y.size, x.size), dtype=np.float32)
    for k, tk in enumerate(t_out):
        # Back-shift query points along characteristics, then clamp.
        x_back = np.clip(x - a * float(tk), x[0], x[-1])
        y_back = np.clip(y - b * float(tk), y[0], y[-1])
        # Rasterise the IC at the back-shifted points: evaluate SampleSpec
        # value at each query (x_back[i], y_back[j]).
        frame = np.full((y.size, x.size), spec.background, dtype=np.float32)
        for r in spec.rects:
            mask_x = (x_back >= r.x_lo) & (x_back < r.x_hi)
            mask_y = (y_back >= r.y_lo) & (y_back < r.y_hi)
            mask = mask_y[:, None] & mask_x[None, :]
            frame = np.where(mask, np.float32(r.value), frame)
        out[k] = frame
    return out


def _simulate_sample_worker(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample worker for multiprocessing. Reconstructs FluxPairs to stay picklable."""
    spec, x_fine, y_fine, t, flux_kind, a, b, cfl, boundary, U = args
    if flux_kind == "linear":
        fx = linear_flux(float(a))
        fy = linear_flux(float(b))
    else:
        fx = LWR_FLUX
        fy = LWR_FLUX
    u0_fine = rasterise_sample_spec(spec, x_fine, y_fine)
    u_fine = simulate_weno5_2d(
        u0_fine, x_fine, y_fine, t,
        flux_x=fx, flux_y=fy,
        cfl=cfl, boundary=boundary,
    )
    u = cell_average_to_grid(u_fine, U).astype(np.float32)
    u0 = cell_average_to_grid(u0_fine, U).astype(np.float32)
    return u, u0


def generate_dataset_upsampled_2d(
    num_samples: int,
    nx: int,
    ny: int,
    nt: int,
    *,
    flux_kind: str,         # "linear" or "lwr"
    a: float | None = None,
    b: float | None = None,
    upsample_factor: int = 8,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    t_max: float = 0.5,
    cfl: float = 0.4,
    num_rects: int | tuple[int, int] = (1, 3),
    u_min: float = 0.1,
    u_max: float = 0.9,
    boundary: str = "ghost",
    seed: int = 42,
    num_workers: int = 1,
    verbose: bool = True,
) -> Dataset2DBundle:
    """Generate a 2D dataset by up-sampled WENO5 + cell-averaging.

    For each sample:
      1. Sample a parametric IC (background + rectangles in physical coords).
      2. Rasterise the IC at the **fine** grid (``upsample_factor`` x the
         target grid in each spatial direction).
      3. Run WENO5 + LF + SSP-RK3 on the fine grid for the chosen flux
         (linear `a*u, b*u` for v0 or LWR `u(1-u)` isotropic for v1).
      4. Cell-average each output snapshot down to the target grid.
      5. Cell-average the fine IC down to the target u0.

    Set ``num_workers > 1`` to parallelise across samples (``multiprocessing``).
    """
    if flux_kind not in ("linear", "lwr"):
        raise ValueError(f"flux_kind must be 'linear' or 'lwr', got {flux_kind!r}")
    if flux_kind == "linear" and (a is None or b is None):
        raise ValueError("flux_kind='linear' requires both a and b")
    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1")

    U = int(upsample_factor)
    rng = np.random.default_rng(seed)

    x_coarse = np.linspace(x_min, x_max, nx, dtype=np.float32)
    y_coarse = np.linspace(y_min, y_max, ny, dtype=np.float32)
    t = np.linspace(0.0, t_max, nt, dtype=np.float32)

    x_fine = np.linspace(x_min, x_max, U * nx, dtype=np.float32)
    y_fine = np.linspace(y_min, y_max, U * ny, dtype=np.float32)

    rect_range = (num_rects, num_rects) if isinstance(num_rects, int) else (int(num_rects[0]), int(num_rects[1]))

    # Pre-generate all ICs from a single RNG so results are deterministic regardless of num_workers.
    specs = []
    for _ in range(num_samples):
        r = int(rng.integers(rect_range[0], rect_range[1] + 1))
        specs.append(_sample_rectangle_ic_spec(x_min, x_max, y_min, y_max, r, u_min, u_max, rng))

    if verbose:
        print(
            f"[gen_2d_upsampled] flux={flux_kind} target={nx}x{ny}x{nt} "
            f"U={U} -> fine={U*nx}x{U*ny}  workers={num_workers}",
            flush=True,
        )

    task_args = [
        (spec, x_fine, y_fine, t, flux_kind, a, b, cfl, boundary, U)
        for spec in specs
    ]

    u_all = np.zeros((num_samples, nt, ny, nx), dtype=np.float32)
    u0_all = np.zeros((num_samples, ny, nx), dtype=np.float32)
    log_every = max(1, num_samples // 20)

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            for i, (u, u0) in enumerate(pool.imap(_simulate_sample_worker, task_args)):
                u_all[i] = u
                u0_all[i] = u0
                if verbose and ((i + 1) % log_every == 0 or (i + 1) == num_samples):
                    print(f"  {i + 1}/{num_samples} samples done", flush=True)
    else:
        for i, args in enumerate(task_args):
            u_all[i], u0_all[i] = _simulate_sample_worker(args)
            if verbose and ((i + 1) % log_every == 0 or (i + 1) == num_samples):
                print(f"  {i + 1}/{num_samples} samples done", flush=True)

    return Dataset2DBundle(x=x_coarse, y=y_coarse, t=t, u=u_all, u0=u0_all)


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
