"""Test 1: mirrored Riemann sweep — does freeze direction track shock direction?

Hypothesis: the freeze is one-sided, set by the upstream IC reach. So:
  - right-going shock (u_L=0, u_R>0, s=+1-u_R>0): upstream is LEFT
    => predicted shock should freeze on its RIGHT edge
  - left-going shock  (u_L>0, u_R=0, s=-1+u_L<0): upstream is RIGHT
    => predicted shock should freeze on its LEFT edge

If the freeze formula T_freeze * |s| = L*k_x*dx holds identically in both
sweeps, the receptive-cone hypothesis is robust to direction. If only one
direction freezes, the cone argument is wrong (or my sign convention is).

This is a near-clone of riemann_sweep_freeze.py with two changes:
  1. Adds a `direction` axis: each u_R is run twice — once as right-going
     (u_L=0, u_R=u) and once as left-going (u_L=u, u_R=0).
  2. Saves a side-by-side plot with both directions overlaid.

Edit RUNS_TO_TEST and run:
    cd <repo root>
    python -m hyperbolic_pde.scripts.riemann_sweep_freeze_mirror

Outputs (per run_dir):
    plots_riemann_sweep_mirror/
        t_freeze_vs_inv_s_both.png   — overlaid scatter, both directions
        tracks_right.png             — pcolormesh + tracks, right-going
        tracks_left.png              — pcolormesh + tracks, left-going
        metrics.txt                  — per-direction table
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.utils.runtime import apply_runtime_overrides
from hyperbolic_pde.models.hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3
from hyperbolic_pde.scripts.eval_riemann_shock import riemann_ic


# --------------------------------------------------------------------- #
# EDIT THIS LIST: absolute paths to run directories.                    #
# Each must contain config.yaml and model_final.pt.                     #
# --------------------------------------------------------------------- #
RUNS_TO_TEST: list[str] = [
     "/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_st3/run_20260505_181417",
]

N_U: int                 = 20
U_MIN: float             = 0.05
U_MAX: float             = 0.95
X_SPLIT: float           = 0.0
SHOCK_THRESH_MULT: float = 1.0


def _build_model(model_cfg: dict, device: torch.device) -> HypNO_ST3:
    _dhn = model_cfg.get("d_hidden_nonadj", None)
    d_hidden_nonadj = int(_dhn) if _dhn is not None else None
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
        d_hidden_nonadj=d_hidden_nonadj,
        include_flux=bool(model_cfg.get("include_flux", True)),
        mask_same_t_nonadj=bool(model_cfg.get("mask_same_t_nonadj", True)),
        temporal_gate_type=str(model_cfg.get("temporal_gate_type", "cfl")),
    ).to(device)


def _detect_shock_position(u_field: np.ndarray, x: np.ndarray) -> np.ndarray:
    grad = np.abs(np.diff(u_field, axis=1))
    idx = np.argmax(grad, axis=1)
    return x[idx]


def _detect_freeze(t: np.ndarray, x_pred: np.ndarray, x_truth: np.ndarray,
                   threshold: float) -> float:
    err = np.abs(x_pred - x_truth)
    bad = err > threshold
    if not bad.any():
        return float(t[-1])
    return float(t[np.argmax(bad)])


def _sweep_direction(model, x_grid, t_grid, x_np, t_np, k_x, dx, t_max,
                     direction: str, skip_ef: bool, device: torch.device) -> dict:
    """direction in {'right', 'left'}."""
    u_values = np.linspace(U_MIN, U_MAX, N_U)
    s_arr, tf_arr = [], []
    pred_fields, truth_tracks, pred_tracks = [], [], []
    threshold = SHOCK_THRESH_MULT * k_x * dx

    for u in u_values:
        if direction == "right":
            u_L, u_R = 0.0, float(u)
        elif direction == "left":
            u_L, u_R = float(u), 0.0
        else:
            raise ValueError(direction)

        s = 1.0 - (u_L + u_R)                       # signed; <0 for left-going
        u0_np = riemann_ic(x_np, u_left=u_L, u_right=u_R, x0=X_SPLIT)
        u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device).unsqueeze(0)

        if not skip_ef:
            ef_adj, ef_nonadj = precompute_lwr_edge_features_v3(u0_t, x_grid, k_x)
            ef_adj = ef_adj.to(device)
            ef_nonadj = ef_nonadj.to(device) if ef_nonadj.numel() > 0 else None
        else:
            ef_adj, ef_nonadj = None, None

        with torch.no_grad():
            pred_t, _, _, _ = model(u0_t, x_grid, t_grid,
                                    edge_feats_adj=ef_adj, edge_feats_nonadj=ef_nonadj)
        pred_np = pred_t[0].cpu().numpy()

        x_truth = X_SPLIT + s * t_np
        x_pred  = _detect_shock_position(pred_np, x_np)
        t_freeze = _detect_freeze(t_np, x_pred, x_truth, threshold)

        s_arr.append(float(s)); tf_arr.append(t_freeze)
        pred_fields.append(pred_np)
        truth_tracks.append(x_truth); pred_tracks.append(x_pred)

    return {
        "direction":   direction,
        "u":           u_values,
        "s":           np.asarray(s_arr),
        "t_freeze":    np.asarray(tf_arr),
        "pred_fields": pred_fields,
        "truth_tracks": truth_tracks,
        "pred_tracks":  pred_tracks,
    }


def _plot_tracks(res: dict, x_np, t_np, t_max, out_path: Path, title: str) -> None:
    n_cols = 5
    n_rows = int(np.ceil(N_U / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3 * n_cols, 2.6 * n_rows),
                              constrained_layout=True)
    axes = axes.flatten()
    for i in range(N_U):
        ax = axes[i]
        u_i = float(res["u"][i]); s_i = float(res["s"][i])
        ax.pcolormesh(x_np, t_np, res["pred_fields"][i], shading="auto", cmap="jet",
                      vmin=0.0, vmax=max(0.05, u_i))
        ax.plot(res["truth_tracks"][i], t_np, "k-",  lw=1.2)
        ax.plot(res["pred_tracks"][i],  t_np, "w--", lw=1.0)
        ax.set_title(f"u={u_i:.2f}, s={s_i:+.2f}", fontsize=9)
    for j in range(N_U, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _evaluate_run(run_dir: Path, device: torch.device) -> None:
    cfg_path = run_dir / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = apply_runtime_overrides(yaml.safe_load(f))

    data_cfg  = cfg["data"]
    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st")))

    nx, nt = int(data_cfg["nx"]), int(data_cfg["nt"])
    x_min, x_max = float(data_cfg["x_min"]), float(data_cfg["x_max"])
    t_max = float(data_cfg["t_max"])

    x_np = np.linspace(x_min, x_max, nx, dtype=np.float32)
    t_np = np.linspace(0.0, t_max, nt, dtype=np.float32)
    dx = float(x_np[1] - x_np[0])

    n_layers = int(model_cfg.get("n_layers", 6))
    k_x      = int(model_cfg.get("stencil_k_x", 3))
    cone_reach = n_layers * k_x * dx
    skip_ef = str(model_cfg.get("encoder_type", "gnn")) == "mlp"

    print(f"  config: nx={nx}, nt={nt}, dx={dx:.5f}, t_max={t_max}")
    print(f"  arch:   L={n_layers}, k_x={k_x}, cone_reach={cone_reach:.4f}")

    model = _build_model(model_cfg, device)
    sd = torch.load(run_dir / "model_final.pt", map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd); model.eval()

    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)

    out_dir = run_dir / "plots_riemann_sweep_mirror"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  sweep: right-going (u_L=0, u_R varies)")
    res_R = _sweep_direction(model, x_grid, t_grid, x_np, t_np, k_x, dx, t_max,
                             "right", skip_ef, device)
    print("  sweep: left-going  (u_L varies, u_R=0)")
    res_L = _sweep_direction(model, x_grid, t_grid, x_np, t_np, k_x, dx, t_max,
                             "left",  skip_ef, device)

    # ---- Overlay plot: t_freeze vs 1/|s|, both directions -------------- #
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    dt_min = (t_np[1] - t_np[0])
    for res, color, label in [(res_R, "tab:blue", "right-going (s>0)"),
                              (res_L, "tab:red",  "left-going  (s<0)")]:
        s_abs = np.abs(res["s"]); inv_s = 1.0 / np.maximum(s_abs, 1e-12)
        tf = res["t_freeze"]
        reliable = (tf < t_max - 1e-6) & (tf > 2.0 * dt_min)
        ax.scatter(inv_s[reliable], tf[reliable], c=color, s=40,
                   label=f"{label} — reliable", zorder=3)
        ax.scatter(inv_s[~reliable], tf[~reliable], c=color, s=40, marker="^",
                   alpha=0.4, label=f"{label} — clipped/glitch")

    inv_s_dense = np.linspace(1.0, 1.0 / U_MIN, 200)
    theory = np.minimum(cone_reach * inv_s_dense, t_max)
    ax.plot(inv_s_dense, theory, "k--",
            label=f"theory: L·k_x·dx / |s|  =  {cone_reach:.3f} / |s|")
    ax.set_xlabel("1 / |s|"); ax.set_ylabel("t_freeze")
    ax.set_ylim(0.0, t_max * 1.1)
    ax.set_title(f"{run_dir.name} — both directions  (L={n_layers}, k_x={k_x})")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)
    fig.savefig(out_dir / "t_freeze_vs_inv_s_both.png", dpi=150)
    plt.close(fig)

    _plot_tracks(res_R, x_np, t_np, t_max, out_dir / "tracks_right.png",
                 f"{run_dir.name} — right-going (u_L=0)")
    _plot_tracks(res_L, x_np, t_np, t_max, out_dir / "tracks_left.png",
                 f"{run_dir.name} — left-going (u_R=0)")

    with (out_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"Run: {run_dir}\n")
        f.write(f"L={n_layers}  k_x={k_x}  dx={dx:.6f}  cone_reach={cone_reach:.6f}  "
                f"t_max={t_max:.4f}\n\n")
        for res in (res_R, res_L):
            f.write(f"--- direction = {res['direction']} ---\n")
            f.write(f"{'u':>6}  {'s':>7}  {'1/|s|':>7}  {'t_freeze':>9}  {'theory':>8}  {'ratio':>6}\n")
            for u_i, s_i, tf_i in zip(res["u"], res["s"], res["t_freeze"]):
                inv = 1.0 / max(abs(s_i), 1e-12)
                th = min(cone_reach * inv, t_max)
                ratio = tf_i / th if th > 0 else float("nan")
                f.write(f"{u_i:6.3f}  {s_i:+7.3f}  {inv:7.3f}  "
                        f"{tf_i:9.4f}  {th:8.4f}  {ratio:6.2f}\n")
            f.write("\n")
    print(f"  wrote {out_dir}")


def main() -> None:
    if not RUNS_TO_TEST:
        raise SystemExit(
            "RUNS_TO_TEST is empty.  Edit riemann_sweep_freeze_mirror.py and "
            "add absolute paths to run directories at the top."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    for rs in RUNS_TO_TEST:
        run_dir = Path(rs)
        print(f"\n=== {run_dir} ===")
        if not run_dir.exists():
            print("  SKIP — does not exist"); continue
        _evaluate_run(run_dir, device)


if __name__ == "__main__":
    main()
