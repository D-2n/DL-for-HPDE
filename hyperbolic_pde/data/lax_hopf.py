from __future__ import annotations

from typing import Tuple

import numpy as np


# Lax-Oleinik / Lax-Hopf exact entropy solver for LWR with quadratic flux
# f(u) = u(1-u). The solution is obtained pointwise from the variational
# representation of the antiderivative potential N(x,t), no time stepping.
#
# Setup. Let N_x = u, N_t = -f(u); then N_t + H(N_x) = 0 with H(p) = f(p),
# which is concave. The Lax-Oleinik formula for concave H is
#
#     N(x,t) = sup_y { N_0(y) + t * M((x-y)/t) },   M(v) = inf_p [pv - H(p)].
#
# For H(p) = p - p^2 one finds M(v) = -(1 - v)^2 / 4, so
#
#     N(x,t) = sup_y { N_0(y) - (x - y - t)^2 / (4t) }.                  (*)
#
# By the envelope theorem the entropy solution at (x,t) is
#
#     u(x,t) = N_x(x,t) = (y*(x,t) + t - x) / (2t),                      (**)
#
# where y* is the maximiser in (*). At smooth points this reduces to
# u(x,t) = u_0(y*) (characteristics). At a kink of u_0 where y* lands on a
# cell edge, (**) correctly produces the rarefaction value.


def _cell_edges(x: np.ndarray) -> np.ndarray:
    """Cell edges for a uniform cell-centred grid: nx+1 values bracketing x."""
    dx = x[1] - x[0]
    return np.concatenate([[x[0] - 0.5 * dx], x + 0.5 * dx]).astype(np.float64)


def _cumulative_N0(u0: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Values of the antiderivative N_0 at cell edges, with N_0(edges[0]) = 0."""
    dx = edges[1] - edges[0]
    return np.concatenate([[0.0], np.cumsum(u0 * dx)]).astype(np.float64)


def _solve_one_time_slice(
    u0: np.ndarray,
    edges: np.ndarray,
    N0_edges: np.ndarray,
    t: float,
    x_out: np.ndarray,
) -> np.ndarray:
    """Evaluate (*) and (**) for all x_out at a single time t > 0.

    Returns u(x_out, t) of shape (nx_out,). Vectorised over (x_out, cells).
    Candidate maximisers per cell: y_k* = x - t(1 - 2 u_k), clamped to the
    cell. Two extra "tail" columns handle constant extrapolation beyond the
    physical domain (boundary='ghost').
    """
    inv_4t = 1.0 / (4.0 * t)
    inv_2t = 1.0 / (2.0 * t)

    # Per-cell candidates.
    u0b = u0[None, :]                                              # (1, nx)
    el = edges[None, :-1]                                          # (1, nx)  left edges
    er = edges[None, 1:]                                           # (1, nx)  right edges
    x_col = x_out[:, None]                                         # (nx_out, 1)

    y_interior = x_col - t * (1.0 - 2.0 * u0b)                     # (nx_out, nx)
    y_clamp_cells = np.clip(y_interior, el, er)
    N0_cells = N0_edges[None, :-1] + u0b * (y_clamp_cells - el)
    phi_cells = N0_cells - (x_col - y_clamp_cells - t) ** 2 * inv_4t

    # Left tail: y in (-inf, edges[0]], slope u_L = u0[0].
    uL = float(u0[0])
    y_interior_L = x_out - t * (1.0 - 2.0 * uL)
    y_clamp_L = np.minimum(y_interior_L, edges[0])
    N0_L = N0_edges[0] + uL * (y_clamp_L - edges[0])
    phi_L = N0_L - (x_out - y_clamp_L - t) ** 2 * inv_4t

    # Right tail: y in [edges[-1], +inf), slope u_R = u0[-1].
    uR = float(u0[-1])
    y_interior_R = x_out - t * (1.0 - 2.0 * uR)
    y_clamp_R = np.maximum(y_interior_R, edges[-1])
    N0_R = N0_edges[-1] + uR * (y_clamp_R - edges[-1])
    phi_R = N0_R - (x_out - y_clamp_R - t) ** 2 * inv_4t

    # Stack all segments: (nx_out, nx + 2).
    phi_all = np.concatenate(
        [phi_cells, phi_L[:, None], phi_R[:, None]], axis=1,
    )
    y_all = np.concatenate(
        [y_clamp_cells, y_clamp_L[:, None], y_clamp_R[:, None]], axis=1,
    )
    k_star = np.argmax(phi_all, axis=1)
    y_star = np.take_along_axis(y_all, k_star[:, None], axis=1)[:, 0]

    # Envelope-theorem value (**).
    u_out = (y_star + t - x_out) * inv_2t

    # Safety clamp to physical range; max principle guarantees values lie in
    # [min u0, max u0], but clip absorbs tiny fp drift.
    return np.clip(u_out, 0.0, 1.0).astype(np.float64)


def solve_lax_hopf(
    u0: np.ndarray,
    x_min: float,
    x_max: float,
    t_max: float,
    nt_out: int,
    boundary: str = "ghost",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact entropy solution of u_t + f(u)_x = 0 with f(u) = u(1-u).

    Mirrors the signature of solve_conservation_fvm in data/fvm.py so the two
    are drop-in interchangeable. Only boundary='ghost' is supported: the IC is
    extended by constants u0[0] and u0[-1] beyond the physical domain.

    Returns (x, t_out, u_hist) with u_hist of shape (nt_out, nx).
    """
    if boundary != "ghost":
        raise NotImplementedError(
            f"solve_lax_hopf currently supports only boundary='ghost'; got '{boundary}'. "
            "Periodic BC would require a sum-over-shifts form of the Lax-Oleinik formula."
        )

    nx = u0.size
    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    t_out = np.linspace(0.0, t_max, nt_out, dtype=np.float32)

    u0_d = u0.astype(np.float64)
    x_d = x.astype(np.float64)
    edges = _cell_edges(x_d)
    N0_edges = _cumulative_N0(u0_d, edges)

    u_hist = np.zeros((nt_out, nx), dtype=np.float32)
    u_hist[0] = u0.astype(np.float32)
    for k in range(1, nt_out):
        t = float(t_out[k])
        if t <= 0.0:
            u_hist[k] = u0.astype(np.float32)
            continue
        u_hist[k] = _solve_one_time_slice(u0_d, edges, N0_edges, t, x_d).astype(np.float32)

    return x, t_out, u_hist
