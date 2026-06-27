"""Precompute & cache godunov / hll / weno5 solutions for a FIXED ARZ eval set.

The FV schemes are deterministic functions of the IC, so for a frozen eval set
their solutions never change -- compute them ONCE here, then pass the resulting
.npz to arz_mark0_paper_eval / arz_mark0_shock_paper_export via --baselines-npz
so they don't re-solve the schemes every run.

WENO5 robustness: methods are solved in the given order and the cache is
RE-SAVED after each method finishes. So if you list weno5 LAST and it hard-crashes
(uncatchable native access violation) on a busy cell, godunov+hll are already on
disk. Re-running with `--baselines weno5 --out <same.npz>` MERGES weno5 into the
existing cache (existing methods are preserved unless re-listed).

Usage:
    python -m hyperbolic_pde.arz.arz_precompute_baselines \\
        --data hyperbolic_pde/arz/data/arz_evaluation_prho.npz \\
        --baselines godunov,hll,weno5
    # -> hyperbolic_pde/arz/data/arz_evaluation_prho_baselines.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.eval_vs_numerical_arz import _run_baseline
from hyperbolic_pde.arz.arz_baseline_cache import (
    save_baseline_cache, load_baseline_cache,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute numerical-baseline solutions for a fixed ARZ eval set."
    )
    ap.add_argument("--data", type=Path, required=True,
                    help="Eval .npz (must carry rho0/w0 + tau/p_form).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Cache .npz (default: <data stem>_baselines.npz beside --data).")
    ap.add_argument("--baselines", type=str, default="godunov,hll,weno5",
                    help="Comma list; put weno5 LAST (crash-isolation, see module doc).")
    ap.add_argument("--boundary", type=str, default="ghost")
    ap.add_argument("--tau", type=float, default=None,
                    help="Override relaxation tau for baselines (default: bundle.tau).")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    bundle = load_arz_dataset(args.data)
    P.set_pressure_form(bundle.p_form)
    print(f"[precompute] pressure_form={bundle.p_form}  N={bundle.rho.shape[0]}")

    tau = float(args.tau) if args.tau is not None else float(bundle.tau)
    baseline_tau = tau if np.isfinite(tau) else 1e6

    out = args.out or args.data.with_name(args.data.stem + "_baselines.npz")
    methods = [m.strip() for m in args.baselines.split(",") if m.strip()]

    N, nt, nx = bundle.rho.shape

    # Start from an existing cache (if any) so a re-run MERGES instead of clobbering.
    solutions: dict = {}
    if out.exists():
        try:
            existing = load_baseline_cache(out, bundle, strict=True)
            for m, d in existing.items():
                solutions[m] = {"rho": np.asarray(d["rho"], np.float32),
                                "w": np.asarray(d["w"], np.float32)}
            print(f"[precompute] merging into existing cache (has: {sorted(solutions)})")
        except Exception as e:
            print(f"[precompute] existing cache at {out} unusable ({e}); overwriting.")

    for m in methods:
        print(f"=== baseline '{m}' over {N} samples ===", flush=True)
        rho_all = np.full((N, nt, nx), np.nan, dtype=np.float32)
        w_all = np.full((N, nt, nx), np.nan, dtype=np.float32)
        t0 = time.perf_counter()
        fails = 0
        for i in range(N):
            rho0 = bundle.rho0[i].astype(np.float64)
            w0 = bundle.w0[i].astype(np.float64)
            try:
                rho_b, w_b = _run_baseline(m, rho0, w0, bundle, baseline_tau, args.boundary)
                rho_all[i] = rho_b
                w_all[i] = w_b
            except Exception as e:
                fails += 1
                print(f"  [warn] {m} failed on sample {i}: {e}", flush=True)
            if args.progress_every and (i + 1) % args.progress_every == 0:
                print(f"  {m}: {i+1}/{N}", flush=True)
        solutions[m] = {"rho": rho_all, "w": w_all}
        dt = time.perf_counter() - t0
        # Re-save after EACH method so a later crash can't lose earlier ones.
        save_baseline_cache(out, bundle, solutions)
        print(f"  {m}: done ({dt:.1f}s, {fails} fails). cache -> {out} "
              f"(methods: {sorted(solutions)})", flush=True)

    print(f"[precompute] complete: {out}")


if __name__ == "__main__":
    main()
