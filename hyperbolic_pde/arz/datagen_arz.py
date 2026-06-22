"""Stratified ARZ dataset generation (plan §4).

Mirrors hyperbolic_pde.data.fvm.generate_dataset, but samples TWO fields
(rho, v) per IC via the existing scalar samplers (called twice), converts
to (rho, w), and solves with the appropriate ground-truth solver:

    * Riemann ICs           -> exact homogeneous solver (riemann_arz)
    * piecewise constant /  -> Strang-split FV reference (reference_arz)
      piecewise sine            with the configured tau.

Important: the homogeneous solver corresponds to tau = infinity. For a
fixed-tau dataset, even Riemann ICs should be propagated with the relaxation
source — so we use reference_arz uniformly when tau is finite, and fall back
to the exact Riemann solver only when tau is None / inf. (The plan's wording
in §4 implicitly assumes a near-homogeneous setup; this implementation keeps
both paths available via the `use_exact_riemann` flag.)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import multiprocessing as mp

import numpy as np

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz.reference_arz import solve_arz_weno5
from hyperbolic_pde.arz.wft_arz import solve_arz_wft_xt
from hyperbolic_pde.arz import riemann_arz as Rie
from hyperbolic_pde.data.fvm import (
    _stratified_pair,
    piecewise_constant_stratified_ic,
    piecewise_sine_ic,
    riemann_stratified_ic,
)


# Default value ranges: rho in [0.1, 0.9] (vacuum-free), v in [0, 1].
_RHO_MIN_DEFAULT = 0.1
_RHO_MAX_DEFAULT = 0.9
_V_MIN_DEFAULT = 0.0
_V_MAX_DEFAULT = 1.0


IC_FUNCS = {
    "riemann_stratified": riemann_stratified_ic,
    "piecewise_constant_stratified": piecewise_constant_stratified_ic,
    "piecewise_sine": piecewise_sine_ic,
}


@dataclass
class ArzDatasetBundle:
    x: np.ndarray              # (nx,)
    t: np.ndarray              # (nt,)
    rho: np.ndarray            # (N, nt, nx)
    w: np.ndarray              # (N, nt, nx)
    v: np.ndarray              # (N, nt, nx)
    rho0: np.ndarray           # (N, nx)
    w0: np.ndarray             # (N, nx)
    v0: np.ndarray             # (N, nx)
    num_segments: np.ndarray   # (N,) int
    ic_type: np.ndarray        # (N,) str
    tau: float                 # scalar metadata
    p_form: str = "rho+rho2"   # pressure closure used to build the GT


def _sample_riemann_colocated(
    x: np.ndarray, rng: np.random.Generator,
    rho_min: float, rho_max: float, v_min: float, v_max: float,
    n_jump_bins: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Riemann IC with rho and v jumps sharing a single interface x0.

    If n_jump_bins > 0, jump magnitudes for both rho and v are drawn from a
    uniformly chosen bin of [0, span], giving equal coverage of weak and
    strong jumps. Without binning, _stratified_pair draws magnitude from
    U(0.03, 0.95)*span, which under-samples the small-jump tail (e.g. a
    rho jump of 0.05 on span 0.8 falls in the bottom 4% of that distribution).
    """
    x0 = rng.uniform(x[0] + 0.2 * (x[-1] - x[0]), x[0] + 0.8 * (x[-1] - x[0]))
    if n_jump_bins > 0:
        rho_left, rho_right = _stratified_pair_binned(rho_min, rho_max, rng, n_jump_bins)
        v_left,   v_right   = _stratified_pair_binned(v_min,   v_max,   rng, n_jump_bins)
    else:
        rho_left, rho_right = _stratified_pair(rho_min, rho_max, rng)
        v_left,   v_right   = _stratified_pair(v_min,   v_max,   rng)
    rho0 = np.where(x <= x0, rho_left, rho_right).astype(np.float64)
    v0   = np.where(x <= x0, v_left,   v_right  ).astype(np.float64)
    return rho0, v0


def _stratified_pair_binned(
    u_min: float, u_max: float, rng: np.random.Generator, n_bins: int,
) -> Tuple[float, float]:
    """Like _stratified_pair but with jump magnitude drawn from a uniformly
    chosen bin of [0, span], so weak and strong jumps are equally represented.

    The bin is chosen uniformly at random, then the magnitude is drawn
    uniformly within that bin. This prevents the U(0.03, 0.95)*span draw
    from concentrating mass on moderate jumps and starving the weak-jump tail.
    A minimum jump floor of 0.01*span is kept so degenerate zero-jump samples
    don't appear (they would be constant ICs, not Riemann problems).
    """
    span = u_max - u_min
    min_jump = max(0.01 * span, 1e-4)
    edges = np.linspace(min_jump, span, n_bins + 1)
    b = int(rng.integers(0, n_bins))
    du = float(rng.uniform(edges[b], edges[b + 1]))
    u_bar = rng.uniform(u_min, u_max)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    lo = u_bar - 0.5 * du
    hi = u_bar + 0.5 * du
    if lo < u_min:
        lo, hi = u_min, u_min + du
    if hi > u_max:
        lo, hi = u_max - du, u_max
    lo = max(u_min, lo)
    hi = min(u_max, hi)
    if sign > 0:
        return float(lo), float(hi)
    return float(hi), float(lo)


def _sample_riemann_vacuum_free(
    x: np.ndarray, rng: np.random.Generator,
    rho_min: float, rho_max: float, v_min: float, v_max: float,
) -> Tuple[float, float, float, float, float]:
    """Sample a homogeneous-ARZ Riemann problem that is VACUUM-FREE BY
    CONSTRUCTION and whose every density state lies in [rho_min, rho_max].

    Rationale (2026-06-10): drawing the four endpoints (rho_L,v_L,rho_R,v_R) at
    random produces ~10% vacuum intermediate states (w_L - v_R < 0 => rho_* clamped
    to 0) and ~10% over-jam (rho_* > 1) under [0.1,0.9]^2 sampling. Both are
    unphysical / unrepresentable by the (sigmoid-rho) decoder. Instead we sample
    the PHYSICAL degrees of freedom directly and derive the rest from the Riemann
    invariants, so rho_L, rho_*, rho_R are all in range and rho_* is never 0:

      independent draws:  rho_L, rho_star, rho_R  in [rho_min, rho_max];  v_* in [v_min,v_max]
      contact (v preserved):     v_R = v_*
      1-wave  (w preserved):     v_L = v_* + p(rho_*) - p(rho_L)
                                 (since w_L = w_*  =>  v_L + p(rho_L) = v_* + p(rho_*))

    v_L is left UNBOUNDED (it is whatever w-preservation dictates; ARZ velocity
    may be negative/large and the model's v output is linear/unbounded).

    Returns (rho_L, v_L, rho_R, v_R, x0).
    """
    rho_L    = float(rng.uniform(rho_min, rho_max))
    rho_star = float(rng.uniform(rho_min, rho_max))
    rho_R    = float(rng.uniform(rho_min, rho_max))
    v_star   = float(rng.uniform(v_min, v_max))
    v_R = v_star
    # p(rho) honoring the active closure; pressure() handles rho vs rho+rho^2.
    v_L = v_star + float(P.pressure(np.array(rho_star)) - P.pressure(np.array(rho_L)))
    x0 = float(rng.uniform(x[0] + 0.2 * (x[-1] - x[0]),
                           x[0] + 0.8 * (x[-1] - x[0])))
    return rho_L, v_L, rho_R, v_R, x0


def _sample_riemann_stratified_cell(
    x: np.ndarray, rng: np.random.Generator,
    rho_min: float, rho_max: float, v_min: float, v_max: float,
    one_wave_is_shock: bool,
    one_wave_jump_range: Tuple[float, float],
    contact_jump_range: Tuple[float, float],
    max_tries: int = 64,
) -> Tuple[float, float, float, float, float]:
    """Sample a vacuum-free Riemann problem inside a STRATIFICATION CELL.

    Unlike `_sample_riemann_vacuum_free` (pure uniform i.i.d. endpoints), this
    constrains the WAVE STRUCTURE so a grid of cells covers the
    (1-wave type) x (1-wave strength) x (contact strength) space uniformly.

    Wave structure (see riemann_arz):
      * 1-wave connects (rho_L, v_L) -> (rho_*, v_*); it is a SHOCK if
        rho_L < rho_* and a RAREFACTION if rho_L > rho_*. w is preserved
        across it, so v_L = v_* + p(rho_*) - p(rho_L).
      * 2-contact connects (rho_*, v_*) -> (rho_R, v_R); v is preserved
        (v_R = v_*), rho jumps rho_* -> rho_R.

    We sample the three densities so that:
      * |rho_* - rho_L| lies in `one_wave_jump_range`, with the sign fixed by
        `one_wave_is_shock` (rho_L < rho_* for a shock, rho_L > rho_* for a
        rarefaction),
      * |rho_R - rho_*| lies in `contact_jump_range`, sign drawn at random.
    rho_* is drawn first (uniform in range), then rho_L and rho_R are placed at
    a jump magnitude inside the cell range; we reject-and-retry the rare draw
    that would push an endpoint outside [rho_min, rho_max], finally clamping.

    All three velocities are kept in [v_min, v_max] (intended subset of [0, 1]):
      v_R = v_* and v_L = v_* + p(rho_*) - p(rho_L). For fixed rho_L, rho_* the
      admissible v_* is [v_min, v_max] intersected with [-dp, 1_v - dp] where
      dp = p(rho_*) - p(rho_L); we draw v_* from that interval directly (so v_L
      lands in range by construction) and treat an empty interval as a failed
      try. This keeps the w_L = w_* invariant EXACT (no post-hoc clipping of v)
      while bounding v -- unlike the legacy sampler which left v_L unbounded.

    Returns (rho_L, v_L, rho_R, v_R, x0), same contract as the vacuum-free
    sampler. Vacuum-free is automatic: rho_* in [rho_min, rho_max] > 0.
    """
    j_lo, j_hi = one_wave_jump_range
    c_lo, c_hi = contact_jump_range
    rho_L = rho_star = rho_R = None
    v_star = v_L = None
    for _ in range(max_tries):
        rho_star = float(rng.uniform(rho_min, rho_max))
        jmag = float(rng.uniform(j_lo, j_hi))
        # shock: rho_L < rho_*  => rho_L = rho_* - jmag ; rarefaction: rho_L > rho_*
        rho_L_try = rho_star - jmag if one_wave_is_shock else rho_star + jmag
        cmag = float(rng.uniform(c_lo, c_hi))
        csign = 1.0 if rng.random() < 0.5 else -1.0
        rho_R_try = rho_star + csign * cmag
        if not (rho_min <= rho_L_try <= rho_max and rho_min <= rho_R_try <= rho_max):
            continue
        # Admissible v_* so that BOTH v_* and v_L = v_* + dp stay in [v_min,v_max].
        dp = float(P.pressure(np.array(rho_star)) - P.pressure(np.array(rho_L_try)))
        lo = max(v_min, v_min - dp)
        hi = min(v_max, v_max - dp)
        if lo > hi:
            continue  # no v_* keeps both velocities in range for this rho draw
        rho_L, rho_star_ok, rho_R = rho_L_try, rho_star, rho_R_try
        v_star = float(rng.uniform(lo, hi))
        v_L = v_star + dp
        break
    if v_L is None:
        # Fallback (every try failed): clamp rho into range, place v_* mid-range
        # and clip v_L. Rare; only for pathological cell/range combinations.
        rho_L = float(np.clip(rho_L_try, rho_min, rho_max))
        rho_R = float(np.clip(rho_R_try, rho_min, rho_max))
        dp = float(P.pressure(np.array(rho_star)) - P.pressure(np.array(rho_L)))
        v_star = float(np.clip(0.5 * (v_min + v_max), v_min, v_max))
        v_L = float(np.clip(v_star + dp, v_min, v_max))

    v_R = v_star
    x0 = float(rng.uniform(x[0] + 0.2 * (x[-1] - x[0]),
                           x[0] + 0.8 * (x[-1] - x[0])))
    return rho_L, v_L, rho_R, v_R, x0


def _sample_rho_v_pair(
    ic_name: str, x: np.ndarray, num_segments: int, rng: np.random.Generator,
    rho_min: float, rho_max: float, v_min: float, v_max: float,
    n_jump_bins: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample rho and v fields with the chosen IC family.

    For riemann_stratified, the two channels share a single jump location
    (`_sample_riemann_colocated`) so the resulting Riemann problem actually
    has a discontinuity in both rho AND v at the same interface.

    For other families (piecewise_constant_stratified, piecewise_sine), the
    two fields are still drawn independently — they aren't single-jump
    problems, so co-location isn't a meaningful constraint.

    n_jump_bins: if > 0, passed to _sample_riemann_colocated to stratify
    jump magnitudes across bins (ensures weak jumps are equally represented).
    """
    if ic_name == "riemann_stratified":
        return _sample_riemann_colocated(x, rng, rho_min, rho_max, v_min, v_max,
                                         n_jump_bins=n_jump_bins)
    if ic_name not in IC_FUNCS:
        raise ValueError(f"Unknown IC family {ic_name!r}; available: {list(IC_FUNCS)}")
    ic_fn = IC_FUNCS[ic_name]
    rho0 = ic_fn(x, num_segments, rho_min, rho_max, rng)
    v0 = ic_fn(x, num_segments, v_min, v_max, rng)
    return rho0.astype(np.float64), v0.astype(np.float64)


def _segments_from_cells(
    rho0: np.ndarray, w0: np.ndarray,
    x_min: float, x_max: float, jump_tol: float = 1e-9,
) -> Tuple[list, list]:
    """Recover piecewise-constant (states, boundaries) from cell arrays.

    For a piecewise-constant IC sampled at cell midpoints (datagen convention:
    nx cells, dx = (x_max-x_min)/nx, centres x_min + (i+0.5)*dx), detect the
    interfaces (adjacent cells whose state differs) and return:
      states     = [(rho, v), ...]   one per constant segment (primitive (rho,v))
      boundaries = [...]              interface x-locations between segments

    A boundary between cell i and i+1 is placed at the shared cell edge
    x_min + (i+1)*dx. WFT then evolves the exact piecewise-constant data.
    """
    nx = rho0.size
    dx = (x_max - x_min) / nx
    v0 = w0 - P.pressure(rho0)
    states = [(float(rho0[0]), float(v0[0]))]
    boundaries: list = []
    for i in range(nx - 1):
        if abs(rho0[i + 1] - rho0[i]) > jump_tol or abs(w0[i + 1] - w0[i]) > jump_tol:
            boundaries.append(float(x_min + (i + 1) * dx))
            states.append((float(rho0[i + 1]), float(v0[i + 1])))
    return states, boundaries


def _solve_one(
    ic_name: str,
    rho0: np.ndarray, w0: np.ndarray,
    x_min: float, x_max: float, t_max: float, nt: int,
    tau: float, cfl: float, boundary: str, refine: int,
    use_exact_riemann: bool,
    fv_solver: str = "hll",
    wft_rare_delta: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return rho_hist, w_hist of shape (nt, nx).

    fv_solver controls which scheme is used for non-exact-Riemann samples:
      "hll"   -- Strang-split HLL reference solver (default, legacy behaviour)
      "weno5" -- component-wise WENO5 + SSP-RK3 + Strang-split exact relaxation
      "wft"   -- wave-front tracking, MACHINE-PRECISION GT for piecewise-constant
                 ICs (riemann_stratified, piecewise_constant_stratified). It is
                 homogeneous (tau=inf) and exact on shocks/contacts; the ONLY
                 error is the rarefaction-fan resolution. piecewise_sine is NOT
                 piecewise-constant -> it falls back to the HLL FV reference (WFT
                 cannot represent a smooth profile as finitely many fronts).

    When use_exact_riemann=True and ic_name=="riemann_stratified", the exact
    analytic solver is used regardless of fv_solver.
    """
    nx = rho0.size

    if use_exact_riemann and ic_name == "riemann_stratified":
        # Detect the single jump location: pick the cell with the largest
        # adjacent jump in rho0 (or v0). riemann_stratified_ic builds a 2-segment
        # field, so this is well-defined.
        dx = (x_max - x_min) / nx
        x_mid = np.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, nx, dtype=np.float64)
        d = np.abs(np.diff(rho0)) + np.abs(np.diff(w0))
        j = int(np.argmax(d))
        x0 = 0.5 * (x_mid[j] + x_mid[j + 1])
        rho_L = float(rho0[j]); w_L = float(w0[j])
        rho_R = float(rho0[j + 1]); w_R = float(w0[j + 1])
        t = np.linspace(0.0, t_max, nt, dtype=np.float64)
        rho_hist, w_hist, _ = Rie.solve_riemann_arz_xt(
            rho_L, w_L, rho_R, w_R, x_mid, t, x0=x0,
        )
        return rho_hist.astype(np.float32), w_hist.astype(np.float32)

    if fv_solver == "wft":
        # Wave-front tracking: machine-precision GT for piecewise-constant ICs.
        # WFT is homogeneous (tau=inf); only valid when there is no relaxation
        # source. piecewise_sine is smooth -> not representable as fronts -> HLL.
        if ic_name == "piecewise_sine":
            _, _, rho_hist, w_hist = Ref.solve_arz_reference(
                rho0, w0, x_min, x_max, t_max, nt_out=nt,
                tau=tau, cfl=cfl, boundary=boundary, refine=refine,
            )
            return rho_hist, w_hist
        if np.isfinite(tau):
            raise ValueError(
                "fv_solver='wft' is homogeneous (tau=inf) only; got finite "
                f"tau={tau}. Use 'hll'/'weno5' for relaxation datasets."
            )
        dx = (x_max - x_min) / nx
        x_mid = np.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, nx, dtype=np.float64)
        t = np.linspace(0.0, t_max, nt, dtype=np.float64)
        states, boundaries = _segments_from_cells(rho0, w0, x_min, x_max)
        # Fan resolution. Default (wft_rare_delta<=0) scales with the GRID:
        # sub-cell fans are invisible on the nx-cell output and only inflate the
        # front count (each interface spawns ~|drho|/rare_delta fronts). A busy
        # multi-segment IC at rare_delta<<dx makes simulate() O(n^2) explode (the
        # 2026-06-20 datagen hang), so the auto value is tied to a tenth of a cell
        # -- finer than the output grid but still cheap. Pass wft_rare_delta>0 to
        # override with an explicit absolute fan resolution (smaller = sharper
        # rarefactions, more fronts, slower on busy ICs).
        if wft_rare_delta and wft_rare_delta > 0.0:
            rare_delta = float(wft_rare_delta)
        else:
            rare_delta = max(0.1 * dx, 1e-4)
        rho_hist, w_hist, _ = solve_arz_wft_xt(
            states, boundaries, x_mid, t, rare_delta=rare_delta,
        )
        return rho_hist.astype(np.float32), w_hist.astype(np.float32)

    if fv_solver == "weno5":
        # WENO5 + SSP-RK3 + Strang-split exact relaxation.
        # tau=inf -> baseline_tau clamped to a large finite value so exp(-dt/tau)~1.
        baseline_tau = tau if np.isfinite(tau) else 1e12
        rho_hist, w_hist = solve_arz_weno5(
            rho0, w0, x_min, x_max, t_max, nt_out=nt,
            tau=baseline_tau, cfl=cfl, boundary=boundary,
        )
        return rho_hist, w_hist

    # Default: Strang-split HLL FV reference solver.
    _, _, rho_hist, w_hist = Ref.solve_arz_reference(
        rho0, w0, x_min, x_max, t_max, nt_out=nt,
        tau=tau, cfl=cfl, boundary=boundary, refine=refine,
    )
    return rho_hist, w_hist


def _build_stratified_riemann_plan(
    x: np.ndarray, rng: np.random.Generator, num_samples: int,
    rho_min: float, rho_max: float, v_min: float, v_max: float,
    n_strength_bins: int,
) -> list:
    """Build a list of `num_samples` vacuum-free Riemann states that uniformly
    covers the (1-wave type) x (1-wave strength) x (contact strength) grid.

    Grid: 2 wave types (1-shock, 1-rarefaction) x n_strength_bins (1-wave) x
    n_strength_bins (contact) = 2 * n_strength_bins**2 cells. Samples are
    distributed as evenly as possible across cells (remainder spread over the
    first cells), then the full list is shuffled so adjacent samples aren't
    correlated. Jump magnitudes span (0, rho_max - rho_min] split into equal
    bins; sign/wave-type is fixed per cell.

    Returns a list of (rho_L, v_L, rho_R, v_R, x0) tuples.
    """
    jump_span = rho_max - rho_min
    edges = np.linspace(0.0, jump_span, n_strength_bins + 1)
    # Avoid a degenerate 0-width 1-wave (rho_L == rho_*): floor the first bin.
    edges[0] = min(0.02 * jump_span, 0.5 * edges[1])
    bins = [(float(edges[b]), float(edges[b + 1])) for b in range(n_strength_bins)]

    cells = []  # (is_shock, one_wave_range, contact_range)
    for is_shock in (True, False):
        for jb in bins:
            for cb in bins:
                cells.append((is_shock, jb, cb))

    n_cells = len(cells)
    base = num_samples // n_cells
    rem = num_samples % n_cells
    plan = []
    for ci, (is_shock, jb, cb) in enumerate(cells):
        count = base + (1 if ci < rem else 0)
        for _ in range(count):
            plan.append(_sample_riemann_stratified_cell(
                x, rng, rho_min, rho_max, v_min, v_max,
                one_wave_is_shock=is_shock,
                one_wave_jump_range=jb, contact_jump_range=cb,
            ))
    rng.shuffle(plan)
    return plan


def generate_arz_riemann_exact_dataset(
    num_samples: int,
    nx: int, nt: int,
    x_min: float, x_max: float, t_max: float,
    seed: int = 0,
    rho_min: float = _RHO_MIN_DEFAULT,
    rho_max: float = _RHO_MAX_DEFAULT,
    v_min: float = _V_MIN_DEFAULT,
    v_max: float = _V_MAX_DEFAULT,
    stratified: bool = False,
    n_strength_bins: int = 4,
) -> ArzDatasetBundle:
    """Generate an exact-Riemann ARZ dataset evaluated at cell midpoints.

    Ground truth is the exact homogeneous (tau=inf) solver sampled at
    x_mid = x_min + (i + 0.5)*dx  for i=0..nx-1, for each output time t_k.
    No FVM discretisation is used.

    If `stratified=True`, IC states are drawn to uniformly cover the
    (1-wave type) x (1-wave strength) x (contact strength) grid
    (`_build_stratified_riemann_plan`) instead of pure uniform i.i.d.
    endpoints. This guarantees the shock/rarefaction corners and the
    weak/strong-jump tails are represented, which uniform sampling
    under-covers. `stratified=False` reproduces the legacy sampler exactly.
    """
    rng = np.random.default_rng(seed)
    dx = (x_max - x_min) / nx
    x_mid = np.array(
        [x_min + (i + 0.5) * dx for i in range(nx)], dtype=np.float32
    )
    t = np.linspace(0.0, t_max, nt, dtype=np.float32)

    rho_all  = np.zeros((num_samples, nt, nx), dtype=np.float32)
    w_all    = np.zeros((num_samples, nt, nx), dtype=np.float32)
    rho0_all = np.zeros((num_samples, nx), dtype=np.float32)
    w0_all   = np.zeros((num_samples, nx), dtype=np.float32)
    v0_all   = np.zeros((num_samples, nx), dtype=np.float32)

    xm = x_mid.astype(np.float64)
    if stratified:
        plan = _build_stratified_riemann_plan(
            xm, rng, num_samples, rho_min, rho_max, v_min, v_max,
            n_strength_bins,
        )
        n_cells = 2 * n_strength_bins ** 2
        sampling_desc = (
            f"STRATIFIED ({n_cells} cells = 2 wave-types x "
            f"{n_strength_bins} 1-wave bins x {n_strength_bins} contact bins)"
        )
    else:
        plan = None
        sampling_desc = "uniform i.i.d. (vacuum-free intermediate-state sampling)"

    print(
        f"[arz riemann exact datagen] N={num_samples}  nx={nx}  nt={nt}  "
        f"x=[{x_min},{x_max}]  t_max={t_max}  "
        f"rho in [{rho_min},{rho_max}]  v* in [{v_min},{v_max}]  "
        f"sampling={sampling_desc}"
    )

    for i in range(num_samples):
        if plan is not None:
            rho_L, v_L, rho_R, v_R, x0 = plan[i]
        else:
            # Vacuum-free: sample rho_L, rho_*, rho_R, v_* directly so rho_* is
            # in range by construction and (w_L - v_R) >= p(rho_min) > 0 always.
            rho_L, v_L, rho_R, v_R, x0 = _sample_riemann_vacuum_free(
                xm, rng, rho_min, rho_max, v_min, v_max
            )
        w_L = v_L + float(P.pressure(np.array(rho_L)))
        w_R = v_R + float(P.pressure(np.array(rho_R)))

        # Build the IC field on the grid from the two states at the interface x0.
        rho0 = np.where(xm <= x0, rho_L, rho_R).astype(np.float64)
        v0   = np.where(xm <= x0, v_L,   v_R  ).astype(np.float64)
        w0   = v0 + P.pressure(rho0)

        rho_hist, w_hist, _ = Rie.solve_riemann_arz_xt(
            rho_L, w_L, rho_R, w_R,
            xm, t.astype(np.float64), x0=x0,
        )
        rho_all[i]  = rho_hist.astype(np.float32)
        w_all[i]    = w_hist.astype(np.float32)
        rho0_all[i] = rho0.astype(np.float32)
        w0_all[i]   = w0.astype(np.float32)
        v0_all[i]   = v0.astype(np.float32)

        if (i + 1) % max(1, num_samples // 10) == 0:
            print(f"  {i+1}/{num_samples}", flush=True)

    v_all = w_all - P.pressure(rho_all)

    return ArzDatasetBundle(
        x=x_mid, t=t,
        rho=rho_all, w=w_all, v=v_all,
        rho0=rho0_all, w0=w0_all, v0=v0_all,
        num_segments=np.ones(num_samples, dtype=np.int64),
        ic_type=np.array(["riemann_stratified"] * num_samples, dtype=np.str_),
        tau=float("inf"),
        p_form=P.get_pressure_form(),
    )


def _worker_generate_sample(args: tuple) -> tuple:
    """Top-level (picklable) worker for parallel datagen.

    Returns (i, rho0, w0, v0, rho_hist, w_hist, seg, fam).
    """
    (i, fam, seg, seed_i,
     x, x_min, x_max, t_max, nt,
     tau, cfl, boundary, refine,
     use_exact_riemann, fv_solver, n_jump_bins,
     rho_min, rho_max, v_min, v_max, wft_rare_delta,
     pressure_form) = args

    # Each spawned worker is a fresh process — set_pressure_form must be called
    # here or it defaults to rho+rho2 regardless of the main process setting.
    P.set_pressure_form(pressure_form)

    rng = np.random.default_rng(seed_i)
    rho0, v0 = _sample_rho_v_pair(
        fam, x.astype(np.float64), seg, rng,
        rho_min, rho_max, v_min, v_max,
        n_jump_bins=n_jump_bins,
    )
    rho0 = np.clip(rho0, P.RHO_MIN, 1.0 - 1e-6)
    w0 = v0 + P.pressure(rho0)
    rho_hist, w_hist = _solve_one(
        fam, rho0, w0,
        x_min=x_min, x_max=x_max, t_max=t_max, nt=nt,
        tau=tau, cfl=cfl, boundary=boundary, refine=refine,
        use_exact_riemann=use_exact_riemann,
        fv_solver=fv_solver,
        wft_rare_delta=wft_rare_delta,
    )
    return i, rho0.astype(np.float32), w0.astype(np.float32), v0.astype(np.float32), rho_hist, w_hist, seg, fam


def generate_arz_dataset(
    num_samples: int,
    nx: int, nt: int,
    x_min: float, x_max: float, t_max: float,
    tau: float,
    families: list[str],
    segments: list[int],
    cfl: float = 0.4,
    boundary: str = "ghost",
    refine: int = 4,
    use_exact_riemann: bool = False,
    fv_solver: str = "hll",
    n_jump_bins: int = 0,
    num_workers: int = 1,
    seed: int = 0,
    rho_min: float = _RHO_MIN_DEFAULT,
    rho_max: float = _RHO_MAX_DEFAULT,
    v_min: float = _V_MIN_DEFAULT,
    v_max: float = _V_MAX_DEFAULT,
    wft_rare_delta: float = 0.0,
) -> ArzDatasetBundle:
    """Generate a stratified ARZ dataset.

    num_samples must be divisible by len(families) * len(segments) (stratified
    quota per cell, matching the LWR datagen convention).

    num_workers: number of parallel processes for solving (default 1). Set to
    the number of CPUs allocated (e.g. 8) to get near-linear speedup on
    WENO5 datagen, which is CPU-bound and embarrassingly parallel.
    """
    n_cells = len(families) * len(segments)
    if num_samples % n_cells != 0:
        raise ValueError(
            f"num_samples ({num_samples}) must be divisible by "
            f"len(families)*len(segments) ({n_cells})."
        )
    per_cell = num_samples // n_cells

    # Use a single master RNG to generate per-sample seeds deterministically,
    # so parallel execution produces the same dataset as serial.
    master_rng = np.random.default_rng(seed)
    sample_seeds = master_rng.integers(0, 2**31, size=num_samples)

    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    t = np.linspace(0.0, t_max, nt, dtype=np.float32)

    rho_all = np.zeros((num_samples, nt, nx), dtype=np.float32)
    w_all = np.zeros((num_samples, nt, nx), dtype=np.float32)
    rho0_all = np.zeros((num_samples, nx), dtype=np.float32)
    w0_all = np.zeros((num_samples, nx), dtype=np.float32)
    v0_all = np.zeros((num_samples, nx), dtype=np.float32)
    seg_meta = np.zeros(num_samples, dtype=np.int64)
    ic_meta: list[str] = [""] * num_samples

    print(
        f"[arz datagen] N={num_samples}  cells={n_cells}  per_cell={per_cell}  "
        f"families={families}  segments={segments}  tau={tau}  refine={refine}  "
        f"fv_solver={fv_solver}  use_exact_riemann={use_exact_riemann}  "
        f"n_jump_bins={n_jump_bins}  num_workers={num_workers}  "
        f"rho in [{rho_min}, {rho_max}]  v in [{v_min}, {v_max}]"
    )

    # Build flat list of (global_index, family, segments) in deterministic order.
    work_items = []
    idx = 0
    for seg in segments:
        for fam in families:
            for _ in range(per_cell):
                work_items.append((
                    idx, fam, seg, int(sample_seeds[idx]),
                    x, x_min, x_max, t_max, nt,
                    tau, cfl, boundary, refine,
                    use_exact_riemann, fv_solver, n_jump_bins,
                    rho_min, rho_max, v_min, v_max, wft_rare_delta,
                    P.get_pressure_form(),
                ))
                idx += 1

    def _collect(result):
        i, rho0, w0, v0, rho_hist, w_hist, seg, fam = result
        rho_all[i] = rho_hist
        w_all[i] = w_hist
        rho0_all[i] = rho0
        w0_all[i] = w0
        v0_all[i] = v0
        seg_meta[i] = seg
        ic_meta[i] = fam

    if num_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            done = 0
            log_every = max(1, num_samples // 10)
            for result in pool.imap_unordered(_worker_generate_sample, work_items, chunksize=4):
                _collect(result)
                done += 1
                if done % log_every == 0:
                    print(f"  {done}/{num_samples}", flush=True)
    else:
        for k, item in enumerate(work_items):
            _collect(_worker_generate_sample(item))
            if (k + 1) % max(1, num_samples // 10) == 0:
                print(f"  {k+1}/{num_samples}", flush=True)

    # v field from (rho, w).
    v_all = w_all - P.pressure(rho_all)

    return ArzDatasetBundle(
        x=x, t=t,
        rho=rho_all, w=w_all, v=v_all,
        rho0=rho0_all, w0=w0_all, v0=v0_all,
        num_segments=seg_meta,
        ic_type=np.array(ic_meta, dtype=np.str_),
        tau=float(tau),
        p_form=P.get_pressure_form(),
    )


def save_arz_dataset(bundle: ArzDatasetBundle, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=bundle.x, t=bundle.t,
        rho=bundle.rho, w=bundle.w, v=bundle.v,
        rho0=bundle.rho0, w0=bundle.w0, v0=bundle.v0,
        num_segments=bundle.num_segments,
        ic_type=bundle.ic_type,
        tau=np.array(bundle.tau, dtype=np.float32),
        p_form=np.array(P.get_pressure_form(), dtype=np.str_),
    )


def load_arz_dataset(path: Path) -> ArzDatasetBundle:
    path = Path(path)
    d = np.load(path, allow_pickle=False)
    # p_form may be absent in pre-toggle datasets; default to the original closure.
    p_form = str(d["p_form"].item()) if "p_form" in d.files else "rho+rho2"
    return ArzDatasetBundle(
        x=d["x"], t=d["t"],
        rho=d["rho"], w=d["w"], v=d["v"],
        rho0=d["rho0"], w0=d["w0"], v0=d["v0"],
        num_segments=d["num_segments"],
        ic_type=np.array([str(s) for s in d["ic_type"]]),
        tau=float(d["tau"]),
        p_form=p_form,
    )


# CLI -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ARZ dataset.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument(
        "--families", type=str,
        default="riemann_stratified,piecewise_constant_stratified,piecewise_sine",
    )
    parser.add_argument("--segments", type=str, default="2,3")
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--boundary", type=str, default="ghost")
    parser.add_argument("--refine", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel worker processes for solving. "
                             "Set to the number of CPUs allocated (e.g. 8).")
    parser.add_argument("--n-jump-bins", type=int, default=0,
                        help="Number of jump-magnitude bins for riemann_stratified ICs "
                             "in the multi-disc path. 0 = legacy U(0.03,0.95)*span draw "
                             "(under-samples weak jumps). >0 = bins of equal width over "
                             "[0, span] chosen uniformly, guaranteeing weak and strong "
                             "jumps are equally represented. Recommended: 8.")
    parser.add_argument("--fv-solver", type=str, default="hll",
                        choices=["hll", "weno5", "wft"],
                        help="Solver for non-exact-Riemann samples: "
                             "'hll' (default, Strang-split HLL), "
                             "'weno5' (WENO5+SSP-RK3+Strang-split relaxation), or "
                             "'wft' (wave-front tracking, MACHINE-PRECISION GT for "
                             "piecewise-constant ICs; homogeneous tau=inf only; "
                             "piecewise_sine falls back to HLL). "
                             "Riemann ICs with --use-exact-riemann still use the "
                             "exact solver regardless of this flag.")
    parser.add_argument("--wft-rare-delta", type=float, default=0.0,
                        help="WFT rarefaction-fan resolution (absolute, in rho units) "
                             "for --fv-solver wft. 0 (default) = auto = 0.1*dx (a tenth "
                             "of a cell; finer than the output grid, still cheap). "
                             "Smaller = sharper rarefactions but more fronts and slower "
                             "on busy multi-segment ICs (rare_delta << dx can make the "
                             "front tracker O(n^2)-explode -- watch the per-sample time).")
    parser.add_argument("--use-exact-riemann", action="store_true",
                        help="Use exact homogeneous (tau=inf) solver for Riemann ICs.")
    parser.add_argument("--exact-riemann-only", action="store_true",
                        help="Generate a pure exact-Riemann dataset evaluated at cell midpoints "
                             "(no FVM; overrides --families/--segments/--tau).")
    parser.add_argument("--stratified-riemann", action="store_true",
                        help="With --exact-riemann-only: stratify IC sampling across "
                             "(1-wave type) x (1-wave strength) x (contact strength) so "
                             "shock/rarefaction corners and weak/strong-jump tails are "
                             "evenly covered (vs uniform i.i.d.).")
    parser.add_argument("--n-strength-bins", type=int, default=4,
                        help="Number of jump-strength bins per axis for "
                             "--stratified-riemann (grid = 2 x bins^2 cells).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho-min", type=float, default=_RHO_MIN_DEFAULT)
    parser.add_argument("--rho-max", type=float, default=_RHO_MAX_DEFAULT)
    parser.add_argument("--v-min",   type=float, default=_V_MIN_DEFAULT)
    parser.add_argument("--v-max",   type=float, default=_V_MAX_DEFAULT)
    parser.add_argument("--pressure-form", type=str, default=None,
                        help="Pressure closure: 'rho' or 'rho+rho2' (default: current module default).")
    args = parser.parse_args()

    if args.pressure_form is not None:
        P.set_pressure_form(args.pressure_form)
    print(f"[arz datagen] pressure_form={P.get_pressure_form()}")

    if args.exact_riemann_only:
        bundle = generate_arz_riemann_exact_dataset(
            num_samples=args.N, nx=args.nx, nt=args.nt,
            x_min=args.x_min, x_max=args.x_max, t_max=args.t_max,
            seed=args.seed,
            rho_min=args.rho_min, rho_max=args.rho_max,
            v_min=args.v_min,     v_max=args.v_max,
            stratified=args.stratified_riemann,
            n_strength_bins=args.n_strength_bins,
        )
    else:
        families = [s.strip() for s in args.families.split(",") if s.strip()]
        segments = [int(s) for s in args.segments.split(",") if s.strip()]
        bundle = generate_arz_dataset(
            num_samples=args.N, nx=args.nx, nt=args.nt,
            x_min=args.x_min, x_max=args.x_max, t_max=args.t_max,
            tau=args.tau, families=families, segments=segments,
            cfl=args.cfl, boundary=args.boundary, refine=args.refine,
            use_exact_riemann=args.use_exact_riemann,
            fv_solver=args.fv_solver,
            n_jump_bins=args.n_jump_bins,
            num_workers=args.num_workers,
            seed=args.seed,
            rho_min=args.rho_min, rho_max=args.rho_max,
            v_min=args.v_min,     v_max=args.v_max,
            wft_rare_delta=args.wft_rare_delta,
        )
    save_arz_dataset(bundle, args.out)
    print(f"[arz datagen] saved {args.out}  shapes: rho={bundle.rho.shape}")
