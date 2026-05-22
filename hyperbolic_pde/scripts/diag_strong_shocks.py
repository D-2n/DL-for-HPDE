"""Diagnostic: HypNO-ST3 on weak Riemann shocks at the u-extrema.

Generates a small deterministic grid of *small-jump* Riemann ICs whose
states both live near 0 (low-density extremum) or both near 1 (high-
density extremum). Tests whether the model handles weak shocks at the
edges of the training distribution, where (a) the characteristic speed
``a = 1 - 2u`` is itself extremal and (b) the upwind direction is
unambiguous but |a| is large in absolute value.

ICs (small jumps, u_R > u_L strictly, 6 samples by default):
    low-extremum  pairs: (0.00, 0.05), (0.00, 0.10), (0.05, 0.10)
    high-extremum pairs: (0.85, 0.90), (0.85, 1.00), (0.90, 1.00)
Step location ``x_0`` is drawn uniformly in ``[-0.5, 0.5]`` per sample
with a fixed seed for reproducibility, so the shock is genuinely
off-center for every IC (rules out left/right boundary-effect
confounders).

Outputs land in ``<run-dir>/diag_strong_shocks/``:
    sample_{k}_uL{...}_uR{...}.png   -- 2x4 heatmap per sample
    summary.txt                       -- per-sample MAE for each method
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from hyperbolic_pde.utils.runtime import apply_runtime_overrides

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import solve_conservation_fvm
from hyperbolic_pde.data.lax_hopf import solve_lax_hopf
from hyperbolic_pde.models.hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3
from hyperbolic_pde.scripts.final_comparison import build_model, mae

SOLVERS = ["Lax-Hopf", "HypNO-ST3", "WENO5", "Godunov"]


def riemann_ic(x: np.ndarray, u_left: float, u_right: float, x0: float) -> np.ndarray:
    """Piecewise-constant u_L for x<x0, u_R for x>=x0."""
    return np.where(x < x0, u_left, u_right).astype(np.float32)


def build_sample_grid(seed: int) -> list[dict]:
    """Deterministic small-jump pairs at the low and high u-extrema.

    Every pair satisfies u_R > u_L (entropy-admissible shock for LWR's
    concave flux). Three pairs cluster near u=0, three near u=1.
    """
    rng = np.random.default_rng(seed)
    pairs = [
        (0.00, 0.05),
        (0.00, 0.10),
        (0.05, 0.10),
        (0.85, 0.90),
        (0.85, 1.00),
        (0.90, 1.00),
    ]
    samples = []
    for uL, uR in pairs:
        x0 = float(rng.uniform(-0.5, 0.5))
        samples.append({"u_L": float(uL), "u_R": float(uR), "x0": x0})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic comparison of HypNO-ST3, WENO5, Godunov, and Lax-Hopf "
            "on weak Riemann shocks at the u-extrema (both states near 0 or "
            "both near 1, with u_R > u_L)."
        )
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Run directory containing config.yaml and model_final.pt.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for x_0 placement and torch/numpy/cuDNN determinism.",
    )
    args = parser.parse_args()

    # --- determinism (matches shock_paper_export.py policy) ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"seed={args.seed}, cudnn.deterministic=True, benchmark=False")

    run_dir = Path(args.run_dir)
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_runtime_overrides(cfg)
    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st")))
    data_cfg = cfg["data"]
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # --- grid (match training resolution exactly) ---
    nx = int(data_cfg["nx"]); nt = int(data_cfg["nt"])
    x_min = float(data_cfg.get("x_min", -1.0))
    x_max = float(data_cfg.get("x_max",  1.0))
    t_max = float(data_cfg.get("t_max",  1.0))
    cfl = float(data_cfg.get("cfl", 0.25))
    boundary = str(data_cfg.get("boundary", "ghost"))
    x_np = np.linspace(x_min, x_max, nx, dtype=np.float32)
    t_np = np.linspace(0.0, t_max, nt, dtype=np.float32)
    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)

    # --- model ---
    model = build_model(model_cfg, device)
    weights_path = run_dir / "model_final.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"No model_final.pt in {run_dir}")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded HypNO-ST3 from {weights_path}")

    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"

    def run_hypno(u0_np: np.ndarray) -> np.ndarray:
        u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device).unsqueeze(0)
        if not skip_ef:
            ef_adj, ef_nonadj = precompute_lwr_edge_features_v3(u0_t, x_grid, stencil_k_x)
            ef_adj = ef_adj.to(device)
            ef_nonadj = ef_nonadj.to(device) if ef_nonadj.numel() > 0 else None
        else:
            ef_adj, ef_nonadj = None, None
        with torch.no_grad():
            pred_t, _, _, _ = model(
                u0_t, x_grid, t_grid,
                edge_feats_adj=ef_adj, edge_feats_nonadj=ef_nonadj,
            )
        return pred_t[0].cpu().numpy()

    out_dir = run_dir / "diag_strong_shocks"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_sample_grid(args.seed)
    summary_lines = ["# diag_strong_shocks summary",
                     "# sample u_L u_R x0 mae_hypno mae_weno mae_godunov"]

    for k, s in enumerate(samples):
        uL, uR, x0 = s["u_L"], s["u_R"], s["x0"]
        u0_np = riemann_ic(x_np, uL, uR, x0)
        print(f"[{k+1}/{len(samples)}] u_L={uL:.2f} u_R={uR:.2f} x0={x0:+.3f}", flush=True)

        lh, _, _ = None, None, None
        # Lax-Hopf exact (returns x, t, u_hist; we already control x_np, t_np)
        _, _, lh = solve_lax_hopf(u0_np, x_min, x_max, t_max, nt, boundary=boundary)
        _, _, weno = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="weno5"
        )
        _, _, godu = solve_conservation_fvm(
            u0_np, x_min, x_max, t_max, nt, cfl=cfl, boundary=boundary, method="godunov"
        )
        hypno = run_hypno(u0_np)

        fields = {"Lax-Hopf": lh, "HypNO-ST3": hypno, "WENO5": weno, "Godunov": godu}

        # MAE relative to Lax-Hopf
        mae_hypno = mae(hypno, lh)
        mae_weno = mae(weno, lh)
        mae_godu = mae(godu, lh)
        summary_lines.append(
            f"{k} {uL:.2f} {uR:.2f} {x0:+.4f} "
            f"{mae_hypno:.4e} {mae_weno:.4e} {mae_godu:.4e}"
        )

        # --- plot: row 0 solutions, row 1 abs errors (Lax-Hopf err is 0 by def) ---
        ncols = len(SOLVERS)
        fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7.5),
                                  constrained_layout=True)
        vmin = min(float(f.min()) for f in fields.values())
        vmax = max(float(f.max()) for f in fields.values())

        for c, name in enumerate(SOLVERS):
            sol = fields[name]
            im = axes[0, c].pcolormesh(
                x_np, t_np, sol, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
            )
            axes[0, c].set_title(name)
            axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
            fig.colorbar(im, ax=axes[0, c], label="u")

        # Shared error vmax across the three numerical/learned methods.
        errs = {n: np.abs(fields[n] - lh) for n in SOLVERS if n != "Lax-Hopf"}
        err_vmax = max(e.max() for e in errs.values())
        axes[1, 0].axis("off")  # Lax-Hopf error column is trivially zero
        for c, name in enumerate(SOLVERS):
            if name == "Lax-Hopf":
                continue
            im = axes[1, c].pcolormesh(
                x_np, t_np, errs[name], shading="auto", cmap="magma",
                vmin=0, vmax=err_vmax,
            )
            axes[1, c].set_title(f"|{name} - Lax-Hopf|  MAE={mae(fields[name], lh):.2e}")
            axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
            fig.colorbar(im, ax=axes[1, c])

        fig.suptitle(
            f"Riemann sample {k}: u_L={uL:.2f}, u_R={uR:.2f}, x_0={x0:+.3f}  "
            f"(jump |u_R - u_L| = {uR - uL:.2f})"
        )
        # Stable, dot-free filename: u_L=0.05 -> 005.
        fname = f"sample_{k:02d}_uL_{int(round(uL*100)):03d}_uR_{int(round(uR*100)):03d}.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {len(samples)} plots + summary.txt to {out_dir}")


if __name__ == "__main__":
    main()
