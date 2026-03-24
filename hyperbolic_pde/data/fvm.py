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

    method: "godunov" (first-order) or "muscl" (second-order MUSCL-Hancock + minmod).
    boundary: "periodic", "fixed", or "ghost".
    """
    if method not in ("godunov", "muscl"):
        raise ValueError(f"method must be 'godunov' or 'muscl', got '{method}'")

    step_fn = _godunov_step if method == "godunov" else _muscl_step

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
        u = step_fn(u, dx, dt, boundary)

        t += dt
        while k < nt_out and t >= t_out[k] - 1e-12:
            u_hist[k] = u.astype(np.float32)
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

    if not num_workers or num_workers <= 1:
        for i in range(num_samples):
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
        for future in as_completed(futures):
            index, u_hist = future.result()
            u[index] = u_hist

    return DatasetBundle(x=x, t=t, u=u, u0=u0_all, ic=ic_all)


def save_dataset(bundle: DatasetBundle, path: Path) -> None:
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
    data = np.load(path)
    return DatasetBundle(
        x=data["x"],
        t=data["t"],
        u=data["u"],
        u0=data["u0"],
        ic=data["ic"],
    )
