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


def solve_conservation_fvm(
    u0: np.ndarray,
    x_min: float,
    x_max: float,
    t_max: float,
    nt_out: int,
    cfl: float = 0.4,
    boundary: str = "periodic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve u_t + f(u)_x = 0 with Godunov FVM for f(u)=u(1-u).

    boundary: "periodic", "fixed", or "ghost".
    - "fixed": keeps boundary values constant (u[0], u[-1]).
    - "ghost": transmissive ghost cells (copies edge values).
    """
    nx = u0.size
    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    dx = x[1] - x[0]
    t_out = np.linspace(0.0, t_max, nt_out, dtype=np.float32)

    u = u0.astype(np.float32).copy()
    u_hist = np.zeros((nt_out, nx), dtype=np.float32)
    u_hist[0] = u

    t = 0.0
    k = 1
    while k < nt_out:
        amax = float(np.max(np.abs(flux_prime(u))))
        amax = max(1e-6, amax)
        dt = cfl * dx / amax
        dt = min(dt, t_max - t)
        if boundary == "periodic":
            u_right = np.roll(u, -1)
            fhat = godunov_flux(u, u_right)
            u = u - (dt / dx) * (fhat - np.roll(fhat, 1))
        elif boundary == "ghost":
            u_ext = np.empty(nx + 2, dtype=np.float32)
            u_ext[1:-1] = u
            u_ext[0] = u[0]
            u_ext[-1] = u[-1]
            fhat = godunov_flux(u_ext[:-1], u_ext[1:])
            u = u - (dt / dx) * (fhat[1:] - fhat[:-1])
        elif boundary == "fixed":
            u_left = u[:-1]
            u_right = u[1:]
            fhat = godunov_flux(u_left, u_right)
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] - (dt / dx) * (fhat[1:] - fhat[:-1])
            u_new[0] = u[0]
            u_new[-1] = u[-1]
            u = u_new
        else:
            raise ValueError("boundary must be 'periodic', 'ghost', or 'fixed'!!!")

        t += dt
        while k < nt_out and t >= t_out[k] - 1e-12:
            u_hist[k] = u
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
) -> Tuple[int, np.ndarray]:
    _, _, u_hist = solve_conservation_fvm(
        u0=u0,
        x_min=x_min,
        x_max=x_max,
        t_max=t_max,
        nt_out=nt_out,
        cfl=cfl,
        boundary=boundary,
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
    num_segments: int,
    u_min: float,
    u_max: float,
    ic_points: int,
    boundary: str,
    seed: int = 42,
    num_workers: int | None = None,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    t = np.linspace(0.0, t_max, nt, dtype=np.float32)
    u = np.zeros((num_samples, nt, nx), dtype=np.float32)
    u0_all = np.zeros((num_samples, nx), dtype=np.float32)
    ic_all = np.zeros((num_samples, ic_points), dtype=np.float32)

    for i in range(num_samples):
        u0 = piecewise_constant_ic(x, num_segments, u_min, u_max, rng)
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
