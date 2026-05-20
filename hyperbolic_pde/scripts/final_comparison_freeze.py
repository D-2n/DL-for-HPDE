"""Single-shock freeze detector for HypNO-ST3.

Runs the model on a single-shock OOD set (riemann_stratified ICs, e.g. the
ood_data_oneshock block) and automatically flags samples where the predicted
shock FREEZES — the front stops tracking the true trajectory and falls
progressively behind.

Freeze score
------------
For each sample the continuous (sub-cell) shock position is found per
timestep for both prediction and Lax-Hopf ground truth (shock_position_error
from eval_vs_numerical). The per-t position error |x_pred - x_gt| is then
linearly fit against t; the SLOPE is the freeze score:

  * slope ~ 0      -> shock tracked correctly (error flat in t)
  * slope > 0      -> error grows with t == the shock is freezing / lagging
  * large slope    -> hard freeze

A constant mis-offset (flat error) scores ~0, so the slope distinguishes a
genuine freeze from a fixed positioning error.

Output (into <run-dir>/plots_freeze/):
  * worst_<rank>_sample_<idx>.png  — comparison plot for the worst-N samples,
    each titled with u_L, u_R and shock speed s = 1-(u_L+u_R)
  * freeze_summary.txt             — every sample ranked by freeze score
"""
from __future__ import annotations

import argparse
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
from hyperbolic_pde.scripts.eval_vs_numerical import shock_position_error


def load_ood_dataset(path: Path):
    """Load a grouped OOD .npz. Returns x, t, u, u0, num_segments, ic_type."""
    from hyperbolic_pde import resolve_data_path

    path = resolve_data_path(path)
    data = np.load(path, allow_pickle=False)
    seg = data["num_segments"].astype(int) if "num_segments" in data else None
    ic = (
        np.array([str(s) for s in data["ic_type"]])
        if "ic_type" in data
        else None
    )
    return data["x"], data["t"], data["u"], data["u0"], seg, ic


def build_model(model_cfg: dict, device: torch.device) -> HypNO_ST3:
    _dhn = model_cfg.get("d_hidden_nonadj", None)
    return HypNO_ST3(
        stencil_k_x=int(model_cfg.get("stencil_k_x", 3)),
        stencil_k_t=int(model_cfg.get("stencil_k_t", 2)),
        d_latent=int(model_cfg.get("d_latent", 128)),
        d_hidden=int(model_cfg.get("d_hidden", 128)),
        n_layers=int(model_cfg.get("n_layers", 6)),
        activation=str(model_cfg.get("activation", "gelu")),
        shock_delta=float(model_cfg.get("shock_delta", 0.01)),
        shock_threshold=float(model_cfg.get("shock_threshold", 0.1)),
        causal_temporal=bool(model_cfg.get("causal_temporal", True)),
        radius_x=None,
        radius_t=None,
        shock_mode=str(model_cfg.get("shock_mode", "physics")),
        weno_eps=float(model_cfg.get("weno_eps", 1e-6)),
        weno_p=float(model_cfg.get("weno_p", 2.0)),
        unified_mp=bool(model_cfg.get("unified_mp", False)),
        readout=str(model_cfg.get("readout", "relu")),
        encoder_scaling=str(model_cfg.get("encoder_scaling", "classic")),
        encoder_type=str(model_cfg.get("encoder_type", "gnn")),
        skip=bool(model_cfg.get("skip", False)),
        use_char_cone=bool(model_cfg.get("use_char_cone", False)),
        detector_path=None,
        detector_cfg={},
        d_hidden_nonadj=int(_dhn) if _dhn is not None else None,
        include_flux=bool(model_cfg.get("include_flux", True)),
        mask_same_t_nonadj=bool(model_cfg.get("mask_same_t_nonadj", True)),
        temporal_gate_type=str(model_cfg.get("temporal_gate_type", "cfl")),
        pure_pairwise_edges=bool(model_cfg.get("pure_pairwise_edges", False)),
        dilated_spatial=bool(model_cfg.get("dilated_spatial", False)),
        normalize_edge_offsets=bool(model_cfg.get("normalize_edge_offsets", False)),
        decoder_depth=int(model_cfg.get("decoder_depth", 3)),
    ).to(device)


def freeze_score(pos_err: np.ndarray, t: np.ndarray) -> float:
    """Slope of shock-position-error vs t. NaN entries (no GT crossing) are
    dropped. Returns the least-squares slope; positive => freezing.
    """
    mask = ~np.isnan(pos_err)
    if mask.sum() < 3:
        return float("nan")
    tt = t[mask]
    ee = pos_err[mask]
    # slope of the linear least-squares fit ee ~ a*tt + b
    slope, _ = np.polyfit(tt, ee, 1)
    return float(slope)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect frozen single-shock predictions of HypNO-ST3."
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Run directory with config.yaml and model_final.pt.",
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the single-shock OOD .npz (e.g. the ood_data_oneshock set).",
    )
    parser.add_argument(
        "--worst_n", type=int, default=8,
        help="Number of worst-freeze samples to plot.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_runtime_overrides(cfg)
    print(f"Using config from {run_dir / 'config.yaml'}")

    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st")))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    x_np, t_np, _u_gt, u0_all, _seg, _ic = load_ood_dataset(Path(args.data_path))
    x_min = float(x_np[0] - 0.5 * (x_np[1] - x_np[0]))
    x_max = float(x_np[-1] + 0.5 * (x_np[1] - x_np[0]))
    t_max = float(t_np[-1])
    nt = len(t_np)
    data_cfg = cfg["data"]
    cfl = float(data_cfg.get("cfl", 0.25))
    boundary = str(data_cfg.get("boundary", "ghost"))
    dx_grid = float(x_np[1] - x_np[0])

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

    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)
    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    skip_ef = str(model_cfg.get("encoder_type", "gnn")) == "mlp"

    plot_dir = run_dir / "plots_freeze"
    plot_dir.mkdir(parents=True, exist_ok=True)

    n = u0_all.shape[0]
    records = []  # (idx, u_L, u_R, s, freeze_slope, mae, hypno, lh)
    for idx in range(n):
        u0_np = u0_all[idx]
        # riemann_stratified IC: left/right states are the domain endpoints.
        u_L = float(u0_np[0])
        u_R = float(u0_np[-1])
        s = 1.0 - (u_L + u_R)  # LWR shock speed for f(u)=u(1-u)

        _, _, lh = solve_lax_hopf(u0_np, x_min, x_max, t_max, nt, boundary=boundary)

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
        hypno = pred_t[0].cpu().numpy()

        pos_mean, _ = shock_position_error(hypno, lh, x_np)
        slope = freeze_score(pos_mean, t_np)
        mae = float(np.abs(hypno - lh).mean())
        records.append((idx, u_L, u_R, s, slope, mae, hypno, lh))
        print(
            f"[{idx + 1}/{n}] u_L={u_L:.3f} u_R={u_R:.3f} s={s:+.3f}  "
            f"freeze_slope={slope:.4e}  MAE={mae:.3e}",
            flush=True,
        )

    # Rank by freeze slope, descending (largest positive slope == worst freeze).
    # NaN slopes (no detectable shock) sort last.
    def _key(r):
        sl = r[4]
        return -1e9 if np.isnan(sl) else sl
    ranked = sorted(records, key=_key, reverse=True)

    # ---- summary table ----
    lines = [
        "Single-shock freeze ranking (freeze_slope = slope of shock-position-",
        "error vs t; positive = freezing). Sorted worst-first.",
        "",
        f"{'rank':>4}  {'idx':>4}  {'u_L':>6}  {'u_R':>6}  {'s':>7}  "
        f"{'freeze_slope':>13}  {'MAE':>10}",
    ]
    for rank, r in enumerate(ranked):
        idx, u_L, u_R, s, slope, mae, _, _ = r
        lines.append(
            f"{rank:>4}  {idx:>4}  {u_L:>6.3f}  {u_R:>6.3f}  {s:>+7.3f}  "
            f"{slope:>13.4e}  {mae:>10.3e}"
        )
    summary = "\n".join(lines)
    print("\n" + summary)
    (plot_dir / "freeze_summary.txt").write_text(summary, encoding="utf-8")

    # ---- freeze-score scatter vs shock speed ----
    sl_all = np.array([r[4] for r in records])
    s_all = np.array([r[3] for r in records])
    ok = ~np.isnan(sl_all)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(s_all[ok], sl_all[ok], c="tab:blue", s=30)
    ax.axhline(0.0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("shock speed  s = 1 - (u_L + u_R)")
    ax.set_ylabel("freeze score (shock-position-error slope vs t)")
    ax.set_title("Freeze score vs shock speed — single-shock OOD")
    ax.grid(True, alpha=0.3)
    fig.savefig(plot_dir / "freeze_vs_speed.png", dpi=150)
    plt.close(fig)

    # ---- worst-N comparison plots ----
    worst = [r for r in ranked if not np.isnan(r[4])][: args.worst_n]
    for rank, r in enumerate(worst):
        idx, u_L, u_R, s, slope, mae, hypno, lh = r
        vmin, vmax = float(lh.min()), float(lh.max())
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for c, (name, sol) in enumerate(
            [("Lax-Hopf (GT)", lh), ("HypNO-ST3", hypno)]
        ):
            im = axes[c].pcolormesh(
                x_np, t_np, sol, shading="auto", cmap="jet", vmin=vmin, vmax=vmax
            )
            axes[c].set_title(name)
            axes[c].set_xlabel("x")
            axes[c].set_ylabel("t")
            fig.colorbar(im, ax=axes[c], label="u")
        err = np.abs(hypno - lh)
        im = axes[2].pcolormesh(x_np, t_np, err, shading="auto", cmap="magma")
        axes[2].set_title("|HypNO-ST3 - GT|")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("t")
        fig.colorbar(im, ax=axes[2])
        fig.suptitle(
            f"FREEZE rank {rank} — sample {idx}  |  "
            f"u_L={u_L:.3f}  u_R={u_R:.3f}  shock speed s={s:+.3f}\n"
            f"freeze score (pos-err slope) = {slope:.4e}   MAE = {mae:.3e}"
        )
        fig.savefig(plot_dir / f"worst_{rank}_sample_{idx}.png", dpi=150)
        plt.close(fig)

    print(f"\nSaved {len(worst)} worst-freeze plots + summary to {plot_dir}")


if __name__ == "__main__":
    main()
