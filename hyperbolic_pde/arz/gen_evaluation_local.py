"""Generate the dedicated ARZ_Evaluation set LOCALLY (WFT machine-precision GT).

Local companion to slurm/arz_gen_evaluation.sh -- same recipe, repo-relative
output path, your local python. This is the ARZ analogue of the LWR paper_eval
set: stratified by (ic_type, num_segments) with the SAME num_segments grid story
(some ID, some OOD) so ARZ and LWR results are directly comparable.

  families : riemann_stratified, piecewise_constant_stratified, piecewise_sine
             (piecewise_sine -> sine STAIRCASE; all WFT-able) -- the families the
             clean WFT orig set trains on.
  segments : 2,3,5,7,8,10,20,30. vs clean-WFT training segs [2,3,5,7,10]:
             {2,3,5,7,10} ID, {8,20,30} OOD. (40 dropped: rho+v independently
             segmented -> ~2x interfaces; 40 segs trips the WFT smoothness guard
             >nx/2 at nx=128. 30 is safe at ~51 interfaces.)
  N = 480  = 3 families x 8 segments x 20/cell (divisible by 24).
  GT       : WFT (homogeneous tau=inf), p=rho, rho in [0.1,0.9], v in [0,1].
  seed 2026 (!= training seed 42) -> ID-segment cells are held-out too.

Usage:
    python -m hyperbolic_pde.arz.gen_evaluation_local
    python -m hyperbolic_pde.arz.gen_evaluation_local --N 480 --num-workers 8
    python -m hyperbolic_pde.arz.gen_evaluation_local --out path.npz --seed 7
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.datagen_arz import generate_arz_dataset, save_arz_dataset

ROOT = Path(__file__).resolve().parents[1]

# Match the clean WFT orig training families (and slurm/arz_gen_evaluation.sh).
FAMILIES = ["riemann_stratified", "piecewise_constant_stratified", "piecewise_sine"]
TRAIN_SEED = 42  # the clean WFT training seed; the eval set MUST differ from it.


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate the ARZ_Evaluation set locally (WFT GT, held-out seed)."
    )
    ap.add_argument("--out", type=Path,
                    default=ROOT / "arz" / "data" / "arz_evaluation_prho.npz")
    ap.add_argument("--N", type=int, default=480,
                    help="Samples (must be divisible by n_families * n_segments = 24).")
    ap.add_argument("--segments", type=str, default="2,3,5,7,8,10,20,30",
                    help="Comma list of num_segments (default mirrors the cluster set). "
                         "Keep max <=~30 at nx=128: rho+v are independently segmented, "
                         "so ~2x num_segments interfaces must stay under the WFT guard (nx/2).")
    ap.add_argument("--seed", type=int, default=2026,
                    help=f"RNG seed; MUST differ from the training seed ({TRAIN_SEED}).")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="Parallel solver processes (WFT is CPU-bound). Drop to 1 "
                         "if Windows multiprocessing misbehaves.")
    ap.add_argument("--wft-rare-delta", type=float, default=0.0,
                    help="WFT fan resolution. 0 = auto (0.1*dx) = fewest fronts = "
                         "fastest/safest. Do NOT push below ~5e-4 on the 20/30-seg "
                         "cells or front collisions blow up (O(n^2)).")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--x-min", type=float, default=-1.0)
    ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--t-max", type=float, default=1.0)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--rho-min", type=float, default=0.1)
    ap.add_argument("--rho-max", type=float, default=0.9)
    ap.add_argument("--v-min", type=float, default=0.0)
    ap.add_argument("--v-max", type=float, default=1.0)
    args = ap.parse_args()

    if args.seed == TRAIN_SEED:
        ap.error(f"--seed {args.seed} equals the training seed; pick a different "
                 f"seed or the eval set will overlap the training set.")

    segments = [int(s) for s in args.segments.split(",") if s.strip()]
    n_cells = len(FAMILIES) * len(segments)
    if args.N % n_cells != 0:
        ap.error(f"--N {args.N} is not divisible by n_families*n_segments={n_cells}.")

    P.set_pressure_form("rho")
    print(f"[gen_evaluation_local] pressure_form={P.get_pressure_form()}  "
          f"seed={args.seed} (train={TRAIN_SEED})  segments={segments}  "
          f"N={args.N} ({args.N // n_cells}/cell)  WFT GT")

    t0 = time.perf_counter()
    bundle = generate_arz_dataset(
        num_samples=args.N, nx=args.nx, nt=args.nt,
        x_min=args.x_min, x_max=args.x_max, t_max=args.t_max,
        tau=float("inf"),
        families=FAMILIES,
        segments=segments,
        cfl=args.cfl, boundary=args.boundary, refine=args.refine,
        use_exact_riemann=True,         # riemann_stratified -> exact analytic GT
        fv_solver="wft",                # staircase families -> WFT machine precision
        wft_rare_delta=args.wft_rare_delta,
        n_jump_bins=8,
        num_workers=args.num_workers,
        seed=args.seed,
        rho_min=args.rho_min, rho_max=args.rho_max,
        v_min=args.v_min, v_max=args.v_max,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_arz_dataset(bundle, args.out)
    dt = time.perf_counter() - t0
    print(f"[gen_evaluation_local] saved {args.out}  rho={bundle.rho.shape}  "
          f"p_form={bundle.p_form}  tau={bundle.tau}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
