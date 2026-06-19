"""Generate a HELD-OUT / OOD ARZ eval set locally (companion to gen_general_local).

The mark-0 model trains on arz_general_local.npz (seed=42). Evaluating on that
file is in-sample. This wrapper produces a separate stratified .npz with the SAME
physics (homogeneous tau=inf, p=rho, same grid/families) but a DIFFERENT seed, so
every sample is fresh -- a genuine held-out set with no train overlap.

Two flavours of "out-of-distribution":
  * Same families/segments, new seed   -> held-out, same distribution. The
    per-(family, num_segments) tables line up cell-for-cell with training, so
    this is the clean apples-to-apples eval. (default)
  * Add --segments with UNSEEN values  -> genuinely OOD cells (e.g. 4,6,8,15,40
    were not in the training [2,3,5,7,10,25]).

N must stay divisible by (n_families * n_segments). Defaults: 3 families x 6
segments x 20/cell = 360 samples -- plenty for stratified MAE tables and far
faster than the 5400-sample training set.

Usage:
    python -m hyperbolic_pde.arz.gen_ood_eval_local
    python -m hyperbolic_pde.arz.gen_ood_eval_local --N 360 --seed 1234
    python -m hyperbolic_pde.arz.gen_ood_eval_local --segments 4,6,8,15,40 --N 300
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.datagen_arz import generate_arz_dataset, save_arz_dataset

ROOT = Path(__file__).resolve().parents[1]

# Families fixed to the training recipe so cells are comparable.
FAMILIES = ["piecewise_constant_stratified", "piecewise_sine", "riemann_stratified"]
TRAIN_SEED = 42  # the seed gen_general_local uses; we must differ from it.


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a held-out / OOD ARZ eval set (different seed from training)."
    )
    ap.add_argument("--out", type=Path,
                    default=ROOT / "arz" / "data" / "arz_ood_eval_local.npz")
    ap.add_argument("--N", type=int, default=360,
                    help="Number of samples (must be divisible by n_families * n_segments).")
    ap.add_argument("--segments", type=str, default="2,3,5,7,10,25",
                    help="Comma list of num_segments. Use UNSEEN values "
                         "(e.g. 4,6,8,15,40) for genuinely OOD cells.")
    ap.add_argument("--seed", type=int, default=2026,
                    help=f"RNG seed; MUST differ from the training seed ({TRAIN_SEED}).")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--x-min", type=float, default=-1.0)
    ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--t-max", type=float, default=1.0)
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--refine", type=int, default=4)
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--rho-min", type=float, default=0.1)
    ap.add_argument("--rho-max", type=float, default=0.9)
    ap.add_argument("--v-min", type=float, default=0.1)
    ap.add_argument("--v-max", type=float, default=0.9)
    args = ap.parse_args()

    if args.seed == TRAIN_SEED:
        ap.error(f"--seed {args.seed} equals the training seed; pick a different "
                 f"seed or the eval set will overlap the training set.")

    segments = [int(s) for s in args.segments.split(",") if s.strip()]
    n_cells = len(FAMILIES) * len(segments)
    if args.N % n_cells != 0:
        ap.error(f"--N {args.N} is not divisible by n_families*n_segments={n_cells}.")

    P.set_pressure_form("rho")
    print(f"[gen_ood_eval_local] pressure_form={P.get_pressure_form()}  "
          f"seed={args.seed} (train={TRAIN_SEED})  segments={segments}  "
          f"N={args.N} ({args.N // n_cells}/cell)")

    t0 = time.perf_counter()
    bundle = generate_arz_dataset(
        num_samples=args.N, nx=args.nx, nt=args.nt,
        x_min=args.x_min, x_max=args.x_max, t_max=args.t_max,
        tau=float("inf"),
        families=FAMILIES,
        segments=segments,
        cfl=args.cfl, boundary=args.boundary, refine=args.refine,
        use_exact_riemann=False, seed=args.seed,
        rho_min=args.rho_min, rho_max=args.rho_max,
        v_min=args.v_min, v_max=args.v_max,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_arz_dataset(bundle, args.out)
    dt = time.perf_counter() - t0
    print(f"[gen_ood_eval_local] saved {args.out}  rho={bundle.rho.shape}  "
          f"p_form={bundle.p_form}  tau={bundle.tau}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
