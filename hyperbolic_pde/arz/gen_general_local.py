"""Generate the GENERAL multi-discontinuity homogeneous ARZ dataset locally.

Convenience wrapper around datagen_arz.generate_arz_dataset with the exact recipe
from slurm/arz_gen_stratified_homog.sh -- the set the original HypNO-ARZ trains on:

    3 families (piecewise_constant_stratified + piecewise_sine + riemann_stratified)
    x num_segments [2,3,5,7,10,25] x 300/cell = 5400 samples,
    nx=128, nt=128, x in [-1,1], t in [0,1], tau=inf (homogeneous), p=rho,
    rho in [0.1,0.9], v in [0.1,0.9], cfl=0.4, refine=4, ghost BC, seed=42.

This is CPU-bound (Strang-split FV ground truth per sample). On a laptop the full
5400-sample set takes a while; use --N for a smaller set while iterating, but keep
N divisible by 18 (3 families x 6 segments) or generation errors out.

Usage:
    python -m hyperbolic_pde.arz.gen_general_local            # full 5400 set
    python -m hyperbolic_pde.arz.gen_general_local --N 540    # quick 540 subset
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.datagen_arz import generate_arz_dataset, save_arz_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the general homogeneous ARZ dataset locally.")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "arz" / "data" / "arz_general_local.npz")
    ap.add_argument("--N", type=int, default=5400,
                    help="Number of samples (must be divisible by 18 = 3 families x 6 segments).")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--x-min", type=float, default=-1.0)
    ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--t-max", type=float, default=1.0)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--refine", type=int, default=4)
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rho-min", type=float, default=0.1)
    ap.add_argument("--rho-max", type=float, default=0.9)
    ap.add_argument("--v-min", type=float, default=0.1)
    ap.add_argument("--v-max", type=float, default=0.9)
    args = ap.parse_args()

    P.set_pressure_form("rho")
    print(f"[gen_general_local] pressure_form={P.get_pressure_form()}")

    t0 = time.perf_counter()
    bundle = generate_arz_dataset(
        num_samples=args.N, nx=args.nx, nt=args.nt,
        x_min=args.x_min, x_max=args.x_max, t_max=args.t_max,
        tau=float("inf"),
        families=["piecewise_constant_stratified", "piecewise_sine", "riemann_stratified"],
        segments=[2, 3, 5, 7, 10, 25],
        cfl=args.cfl, boundary=args.boundary, refine=args.refine,
        use_exact_riemann=False, seed=args.seed,
        rho_min=args.rho_min, rho_max=args.rho_max,
        v_min=args.v_min, v_max=args.v_max,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_arz_dataset(bundle, args.out)
    dt = time.perf_counter() - t0
    print(f"[gen_general_local] saved {args.out}  rho={bundle.rho.shape}  "
          f"p_form={bundle.p_form}  tau={bundle.tau}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
