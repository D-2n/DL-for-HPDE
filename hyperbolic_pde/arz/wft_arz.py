"""Wave-front tracking (WFT) for the homogeneous ARZ system  (machine-precision GT).

Adapts the scalar-LWR front tracker (notebooks/WFT-algorithm.ipynb) to the ARZ
2x2 system. The solution is carried as a finite set of straight-line fronts with
NO spatial grid, so there is zero numerical diffusion: the only approximation is
the fan resolution `rare_delta` used to discretise 1-rarefactions into a sequence
of small contact-free fronts. As `rare_delta -> 0` the result converges to the
exact entropy solution, and for piecewise-constant data with no rarefactions it
is exact to machine precision.

Why this is the right ground truth for ARZ
------------------------------------------
ARZ has a clean Riemann structure (see `riemann_arz`): a genuinely-nonlinear
1-wave (shock or rarefaction in lambda1) followed by a linearly-degenerate
2-contact (lambda2 = v). The two Riemann invariants are exact:

    * w is constant across the 1-wave        ->  w_* = w_L
    * v is constant across the 2-contact     ->  v_* = v_R   (contact speed = v_R)

so every elementary wave has an EXACT speed:
    * 1-shock:        Rankine-Hugoniot speed on rho with w preserved,
                          s = (rho_* v_* - rho_L v_L) / (rho_* - rho_L)
    * 1-rarefaction:  a fan of fronts, each carrying a sliver of the fan, moving
                          at the local characteristic speed lambda1(rho)
    * 2-contact:      s = v_* = v_R

Front collisions are resolved by re-solving the exact ARZ Riemann problem between
the two outer states (reusing `solve_riemann_arz`'s classification logic). This
is the system analogue of the scalar front tracker's `riemann_fronts`.

State convention. A front separates a LEFT state (rho_l, w_l) from a RIGHT state
(rho_r, w_r). Primitive v is derived as v = w - p(rho). All states are stored in
(rho, w) -- the division-free, vacuum-safe pair used everywhere in this package.

Pressure form follows the module global in `physics_arz` (set via
`set_pressure_form`); WFT reads p / p' / the fan profile from there.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.riemann_arz import _rho_star_from_w_minus_v, _rho_in_fan


# --------------------------------------------------------------------------- #
# Front
# --------------------------------------------------------------------------- #
@dataclass
class ArzFront:
    """A single discontinuity separating (rho_l, w_l) | (rho_r, w_r)."""
    x: float        # position
    rho_l: float
    w_l: float
    rho_r: float
    w_r: float
    s: float        # propagation speed
    kind: str = ""  # 'shock1' | 'rare1' | 'contact2'  (bookkeeping only)

    def vl(self) -> float:
        return self.w_l - float(P.pressure(np.array(self.rho_l)))

    def vr(self) -> float:
        return self.w_r - float(P.pressure(np.array(self.rho_r)))


# --------------------------------------------------------------------------- #
# Elementary-wave helpers
# --------------------------------------------------------------------------- #
def _lambda1(rho: float, w: float) -> float:
    v = w - float(P.pressure(np.array(rho)))
    return v - rho * float(P.dpressure(np.array(rho)))


def _shock1_speed(rho_L: float, w_L: float, rho_star: float, v_star: float) -> float:
    """RH speed of the 1-shock (w preserved across it, so w_* = w_L)."""
    v_L = w_L - float(P.pressure(np.array(rho_L)))
    denom = rho_star - rho_L
    if abs(denom) < 1e-14:
        return _lambda1(rho_L, w_L)
    return (rho_star * v_star - rho_L * v_L) / denom


class ArzFrontTracking:
    """Front tracking for homogeneous ARZ (tau = inf).

    Parameters
    ----------
    boundaries : list[float]
        Interface locations [x1, ..., xm], strictly increasing.
    states : list[tuple[float, float]]
        Piecewise-constant primitive states [(rho_0, v_0), ..., (rho_m, v_m)] with
        u(x) = state_k on [x_k, x_{k+1}); len(states) = len(boundaries) + 1.
        States are given in (rho, v); stored internally as (rho, w).
    rare_delta : float
        1-rarefaction fan resolution (in rho). Smaller -> finer / more exact.
    tol : float
        Numerical tolerance for jump/collision detection.
    """

    def __init__(self, boundaries, states, rare_delta: float = 0.01, tol: float = 1e-12):
        if len(states) != len(boundaries) + 1:
            raise ValueError("Need len(states) = len(boundaries) + 1.")
        self.rare_delta = float(rare_delta)
        self.tol = float(tol)

        # Convert (rho, v) -> (rho, w); keep the outer states for profile rebuild.
        rw = [self._to_rw(r, v) for (r, v) in states]
        self.left_state = rw[0]
        self.right_state = rw[-1]

        self.fronts: List[ArzFront] = []
        for x0, (rl, wl), (rr, wr) in zip(boundaries, rw[:-1], rw[1:]):
            self.fronts.extend(self.riemann_fronts(rl, wl, rr, wr, x0))
        self.fronts = self._sort_fronts(self.fronts)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_rw(rho: float, v: float) -> Tuple[float, float]:
        return float(rho), float(v + float(P.pressure(np.array(float(rho)))))

    def _sort_fronts(self, fronts: List[ArzFront]) -> List[ArzFront]:
        # Sort by position; tie-break by speed so co-located fans stay ordered.
        return sorted(fronts, key=lambda fr: (fr.x, fr.s))

    # ------------------------------------------------------------------ #
    # Exact Riemann solver in front-tracking form (the heart of WFT).
    # ------------------------------------------------------------------ #
    def riemann_fronts(self, rho_L, w_L, rho_R, w_R, x0) -> List[ArzFront]:
        """Decompose the Riemann problem (rho_L,w_L)|(rho_R,w_R) into fronts.

        Structure:  L --[1-wave]-- (rho_*, w_*=w_L) --[2-contact]-- R
        with v_* = v_R, p(rho_*) = w_L - v_R.
        """
        rho_L = float(rho_L); w_L = float(w_L)
        rho_R = float(rho_R); w_R = float(w_R)
        v_L = w_L - float(P.pressure(np.array(rho_L)))
        v_R = w_R - float(P.pressure(np.array(rho_R)))

        # Intermediate (star) state.
        rho_star = float(_rho_star_from_w_minus_v(w_L - v_R))
        w_star = w_L
        v_star = v_R

        fronts: List[ArzFront] = []

        # --- 1-wave: shock vs rarefaction by Lax on lambda1 -------------- #
        lam1_L = _lambda1(rho_L, w_L)
        lam1_star = _lambda1(rho_star, w_star)
        if abs(rho_star - rho_L) > self.tol:
            if lam1_L > lam1_star + 1e-12:
                # 1-shock: single front L | star.
                s = _shock1_speed(rho_L, w_L, rho_star, v_star)
                fronts.append(ArzFront(x0, rho_L, w_L, rho_star, w_star, s, "shock1"))
            else:
                # 1-rarefaction: fan of small fronts L -> star in rho. w = w_L
                # throughout; each sliver front moves at the local lambda1 of its
                # left edge (entropy-correct ordering of the fan).
                n = max(1, int(math.ceil(abs(rho_L - rho_star) / self.rare_delta)))
                rhos = np.linspace(rho_L, rho_star, n + 1)
                for a, b in zip(rhos[:-1], rhos[1:]):
                    if abs(b - a) <= self.tol:
                        continue
                    # Speed of a fan front: average characteristic speed of the
                    # sliver (its RH speed equals the mean lambda1 on [a,b] for a
                    # vanishing jump; using the left-edge lambda1 keeps the fan
                    # monotone and is the standard Dafermos front-tracking choice).
                    s = 0.5 * (_lambda1(float(a), w_L) + _lambda1(float(b), w_L))
                    fronts.append(ArzFront(x0, float(a), w_L, float(b), w_L, s, "rare1"))

        # --- 2-contact: star | R, speed v_* = v_R ----------------------- #
        if abs(rho_star - rho_R) > self.tol or abs(w_star - w_R) > self.tol:
            fronts.append(ArzFront(x0, rho_star, w_star, rho_R, w_R, v_star, "contact2"))

        return fronts

    # ------------------------------------------------------------------ #
    def _advance(self, dt: float) -> None:
        for fr in self.fronts:
            fr.x += fr.s * dt

    def _next_event_time(self) -> float:
        """Earliest collision time among adjacent fronts (+inf if none)."""
        if len(self.fronts) < 2:
            return math.inf
        dt_min = math.inf
        for a, b in zip(self.fronts[:-1], self.fronts[1:]):
            gap = b.x - a.x
            ds = a.s - b.s          # a catches up to b only if a faster than b
            if ds > self.tol and gap > -self.tol:
                dt = gap / ds
                if dt < dt_min:
                    dt_min = dt
        return dt_min

    def _resolve_collisions_at_current_time(self) -> None:
        """Replace each cluster of co-located fronts by the Riemann fan between
        the cluster's outer states."""
        if not self.fronts:
            return
        new_fronts: List[ArzFront] = []
        i, n = 0, len(self.fronts)
        while i < n:
            j = i
            while j + 1 < n and abs(self.fronts[j + 1].x - self.fronts[i].x) <= self.tol:
                j += 1
            if j == i:
                new_fronts.append(self.fronts[i])
            else:
                x0 = self.fronts[i].x
                L = self.fronts[i]
                R = self.fronts[j]
                new_fronts.extend(
                    self.riemann_fronts(L.rho_l, L.w_l, R.rho_r, R.w_r, x0))
            i = j + 1
        self.fronts = self._sort_fronts(new_fronts)

    # ------------------------------------------------------------------ #
    def simulate(self, T: float, keep_history: bool = True,
                 keep_segments: bool = True, max_events: int = 50000):
        """Evolve to time T. Returns (segments, history).

        segments : list of (t0, x0, t1, x1) straight-line front trajectories
                   (for the space-time front diagram). Empty if keep_segments=False.
        history  : list of (t, deepcopy(fronts)) after each interaction event.
                   Empty (only the final state retained in self.fronts) if
                   keep_history=False -- the datagen path does NOT need the per-event
                   history, and deep-copying every front at every event is itself an
                   O(n*events) cost for busy ICs. Use keep_history=False there.

        max_events bounds pathological interaction storms; if hit, the partially
        evolved state is returned with a warning (rather than silently spinning).
        """
        T = float(T)
        t = 0.0
        segments = [] if keep_segments else None
        history = [(t, deepcopy(self.fronts))] if keep_history else []
        ev = 0

        while t < T and self.fronts and ev < max_events:
            ev += 1
            dt = self._next_event_time()

            if not math.isfinite(dt) or t + dt > T:
                dt = T - t
                if dt <= self.tol:
                    break
                if keep_segments:
                    for fr in self.fronts:
                        segments.append((t, fr.x, t + dt, fr.x + fr.s * dt))
                self._advance(dt)
                t = T
                if keep_history:
                    history.append((t, deepcopy(self.fronts)))
                break

            if dt <= self.tol:
                self._resolve_collisions_at_current_time()
                if keep_history:
                    history.append((t, deepcopy(self.fronts)))
                continue

            if keep_segments:
                for fr in self.fronts:
                    segments.append((t, fr.x, t + dt, fr.x + fr.s * dt))
            self._advance(dt)
            t += dt
            self._resolve_collisions_at_current_time()
            if keep_history:
                history.append((t, deepcopy(self.fronts)))

        if ev >= max_events:
            import warnings
            warnings.warn(
                f"WFT simulate() hit max_events={max_events} at t={t:.4f}<{T} "
                f"with {len(self.fronts)} fronts -- interaction storm (rare_delta "
                f"too fine for this IC?). Returning partially-evolved state.",
                RuntimeWarning,
            )
        return segments or [], history

    # ------------------------------------------------------------------ #
    def profile_at(self, T: float, x: np.ndarray):
        """Sample (rho, w, v) at time T on points x WITHOUT mutating self.

        Works on a fresh copy: advance fronts to T (no interactions are added
        between events here -- collisions are handled inside simulate), then read
        off the piecewise-constant state. For correctness across interactions,
        call `simulate(T)` first so `self.fronts` is the post-interaction set at
        T, then this samples that set.
        """
        x = np.asarray(x, dtype=np.float64)
        rho = np.empty_like(x)
        w = np.empty_like(x)

        fronts = self._sort_fronts(self.fronts)
        # Left-of-everything state.
        if fronts:
            state_rho, state_w = fronts[0].rho_l, fronts[0].w_l
        else:
            state_rho, state_w = self.left_state

        # Build breakpoints (front positions) and the state to the RIGHT of each.
        xs = [fr.x for fr in fronts]
        right_rho = [fr.rho_r for fr in fronts]
        right_w = [fr.w_r for fr in fronts]

        idx = np.searchsorted(xs, x, side="right") if xs else np.zeros_like(x, dtype=int)
        for k in range(x.size):
            j = int(idx[k])
            if j == 0:
                rho[k], w[k] = state_rho, state_w
            else:
                rho[k], w[k] = right_rho[j - 1], right_w[j - 1]
        v = w - P.pressure(rho)
        return rho, w, v


# --------------------------------------------------------------------------- #
# Convenience driver: solve a Riemann IC on an (x, t) grid (GT for datagen/eval).
# --------------------------------------------------------------------------- #
def solve_arz_wft_xt(
    states, boundaries,
    x: np.ndarray, t: np.ndarray,
    rare_delta: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate WFT on an (x, t) grid for piecewise-constant IC.

    Parameters
    ----------
    states : list[(rho, v)]   piecewise-constant primitive states.
    boundaries : list[float]  interface locations (len = len(states) - 1).
    x : (nx,) sample points.   t : (nt,) output times (assumed increasing, t[0]>=0).
    rare_delta : fan resolution.

    Returns rho, w, v of shape (nt, nx).
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    nt, nx = t.size, x.size
    rho = np.empty((nt, nx)); w = np.empty((nt, nx)); v = np.empty((nt, nx))

    # Single monotonic pass: build ONE solver and step it forward through the
    # (assumed increasing) output times, sampling at each. The previous version
    # rebuilt the solver and re-simulated from t=0 for EVERY output time -- with
    # nt output snapshots that is nt full simulations, which is what made busy
    # multi-segment datagen explode. simulate() advances self.fronts in place and
    # profile_at() is read-only, so stepping incrementally is exact.
    solver = ArzFrontTracking(boundaries, states, rare_delta=rare_delta)
    t_prev = 0.0
    for k, tk in enumerate(t):
        tk = float(tk)
        if tk > t_prev + solver.tol:
            # advance from t_prev to tk (relative time; simulate runs 0..(tk-t_prev)
            # on the current front set), no history/segments needed for sampling.
            solver.simulate(tk - t_prev, keep_history=False, keep_segments=False)
            t_prev = tk
        r, ww, vv = solver.profile_at(tk, x)
        rho[k], w[k], v[k] = r, ww, vv
    return rho, w, v


def solve_arz_wft_riemann_xt(
    rho_L: float, v_L: float, rho_R: float, v_R: float,
    x: np.ndarray, t: np.ndarray, x0: float = 0.0,
    rare_delta: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-interface Riemann convenience wrapper (mirrors
    riemann_arz.solve_riemann_arz_xt's signature but in (rho, v))."""
    return solve_arz_wft_xt([(rho_L, v_L), (rho_R, v_R)], [x0], x, t,
                            rare_delta=rare_delta)
