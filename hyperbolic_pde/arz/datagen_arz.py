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

import numpy as np

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz import reference_arz as Ref
from hyperbolic_pde.arz import riemann_arz as Rie
from hyperbolic_pde.data.fvm import (
    piecewise_constant_stratified_ic,
    piecewise_sine_ic,
    riemann_stratified_ic,
)


# Default value ranges (mirror LWR config: u in [0.1, 0.9]).
_RHO_MIN_DEFAULT = 0.1
_RHO_MAX_DEFAULT = 0.9
_V_MIN_DEFAULT = 0.1
_V_MAX_DEFAULT = 0.9


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


def _sample_rho_v_pair(
    ic_name: str, x: np.ndarray, num_segments: int, rng: np.random.Generator,
    rho_min: float, rho_max: float, v_min: float, v_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Independently sample rho and v fields with the chosen IC family.

    The two fields share the family/num_segments but use independent value
    draws (the rng is shared; we just call the sampler twice in sequence).
    """
    if ic_name not in IC_FUNCS:
        raise ValueError(f"Unknown IC family {ic_name!r}; available: {list(IC_FUNCS)}")
    ic_fn = IC_FUNCS[ic_name]
    rho0 = ic_fn(x, num_segments, rho_min, rho_max, rng)
    v0 = ic_fn(x, num_segments, v_min, v_max, rng)
    return rho0.astype(np.float64), v0.astype(np.float64)


def _solve_one(
    ic_name: str,
    rho0: np.ndarray, w0: np.ndarray,
    x_min: float, x_max: float, t_max: float, nt: int,
    tau: float, cfl: float, boundary: str, refine: int,
    use_exact_riemann: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return rho_hist, w_hist of shape (nt, nx)."""
    nx = rho0.size
    if use_exact_riemann and ic_name == "riemann_stratified":
        # Detect the single jump location: pick the cell with the largest
        # adjacent jump in rho0 (or v0). riemann_stratified_ic builds a 2-segment
        # field, so this is well-defined.
        x = np.linspace(x_min, x_max, nx, dtype=np.float64)
        d = np.abs(np.diff(rho0)) + np.abs(np.diff(w0))
        j = int(np.argmax(d))
        x0 = 0.5 * (x[j] + x[j + 1])
        rho_L = float(rho0[j]); w_L = float(w0[j])
        rho_R = float(rho0[j + 1]); w_R = float(w0[j + 1])
        t = np.linspace(0.0, t_max, nt, dtype=np.float64)
        rho_hist, w_hist, _ = Rie.solve_riemann_arz_xt(
            rho_L, w_L, rho_R, w_R, x, t, x0=x0,
        )
        return rho_hist.astype(np.float32), w_hist.astype(np.float32)
    # Default: Strang-split FV with the configured tau.
    _, _, rho_hist, w_hist = Ref.solve_arz_reference(
        rho0, w0, x_min, x_max, t_max, nt_out=nt,
        tau=tau, cfl=cfl, boundary=boundary, refine=refine,
    )
    return rho_hist, w_hist


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
    seed: int = 0,
    rho_min: float = _RHO_MIN_DEFAULT,
    rho_max: float = _RHO_MAX_DEFAULT,
    v_min: float = _V_MIN_DEFAULT,
    v_max: float = _V_MAX_DEFAULT,
) -> ArzDatasetBundle:
    """Generate a stratified ARZ dataset.

    num_samples must be divisible by len(families) * len(segments) (stratified
    quota per cell, matching the LWR datagen convention).
    """
    n_cells = len(families) * len(segments)
    if num_samples % n_cells != 0:
        raise ValueError(
            f"num_samples ({num_samples}) must be divisible by "
            f"len(families)*len(segments) ({n_cells})."
        )
    per_cell = num_samples // n_cells

    rng = np.random.default_rng(seed)
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
        f"rho in [{rho_min}, {rho_max}]  v in [{v_min}, {v_max}]"
    )

    i = 0
    for seg in segments:
        for fam in families:
            for _ in range(per_cell):
                # Sample IC in (rho, v).
                rho0, v0 = _sample_rho_v_pair(
                    fam, x.astype(np.float64), seg, rng,
                    rho_min, rho_max, v_min, v_max,
                )
                rho0 = np.clip(rho0, P.RHO_MIN, 1.0 - 1e-6)
                w0 = v0 + P.pressure(rho0)
                rho_hist, w_hist = _solve_one(
                    fam, rho0, w0,
                    x_min=x_min, x_max=x_max, t_max=t_max, nt=nt,
                    tau=tau, cfl=cfl, boundary=boundary, refine=refine,
                    use_exact_riemann=use_exact_riemann,
                )
                rho_all[i] = rho_hist
                w_all[i] = w_hist
                rho0_all[i] = rho0.astype(np.float32)
                w0_all[i] = w0.astype(np.float32)
                v0_all[i] = v0.astype(np.float32)
                seg_meta[i] = seg
                ic_meta[i] = fam
                i += 1
                if i % max(1, num_samples // 10) == 0:
                    print(f"  {i}/{num_samples}", flush=True)

    # v field from (rho, w).
    v_all = w_all - P.pressure(rho_all)

    return ArzDatasetBundle(
        x=x, t=t,
        rho=rho_all, w=w_all, v=v_all,
        rho0=rho0_all, w0=w0_all, v0=v0_all,
        num_segments=seg_meta,
        ic_type=np.array(ic_meta, dtype=np.str_),
        tau=float(tau),
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
        p_form=np.array("rho+rho2", dtype=np.str_),
    )


def load_arz_dataset(path: Path) -> ArzDatasetBundle:
    path = Path(path)
    d = np.load(path, allow_pickle=False)
    return ArzDatasetBundle(
        x=d["x"], t=d["t"],
        rho=d["rho"], w=d["w"], v=d["v"],
        rho0=d["rho0"], w0=d["w0"], v0=d["v0"],
        num_segments=d["num_segments"],
        ic_type=np.array([str(s) for s in d["ic_type"]]),
        tau=float(d["tau"]),
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
    parser.add_argument("--use-exact-riemann", action="store_true",
                        help="Use exact homogeneous (tau=inf) solver for Riemann ICs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho-min", type=float, default=_RHO_MIN_DEFAULT)
    parser.add_argument("--rho-max", type=float, default=_RHO_MAX_DEFAULT)
    parser.add_argument("--v-min",   type=float, default=_V_MIN_DEFAULT)
    parser.add_argument("--v-max",   type=float, default=_V_MAX_DEFAULT)
    args = parser.parse_args()

    families = [s.strip() for s in args.families.split(",") if s.strip()]
    segments = [int(s) for s in args.segments.split(",") if s.strip()]
    bundle = generate_arz_dataset(
        num_samples=args.N, nx=args.nx, nt=args.nt,
        x_min=args.x_min, x_max=args.x_max, t_max=args.t_max,
        tau=args.tau, families=families, segments=segments,
        cfl=args.cfl, boundary=args.boundary, refine=args.refine,
        use_exact_riemann=args.use_exact_riemann, seed=args.seed,
        rho_min=args.rho_min, rho_max=args.rho_max,
        v_min=args.v_min,     v_max=args.v_max,
    )
    save_arz_dataset(bundle, args.out)
    print(f"[arz datagen] saved {args.out}  shapes: rho={bundle.rho.shape}")
