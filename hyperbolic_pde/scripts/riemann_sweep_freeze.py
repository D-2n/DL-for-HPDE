"""Riemann shock-speed sweep to test the receptive-cone freeze diagnosis.

Hypothesis: the prediction's shock freezes (goes vertical in the (x,t)
plot) at  t_freeze ≈ n_layers * k_x * dx / |s|  where s = 1 - (u_L+u_R)
is the Rankine-Hugoniot speed for LWR.  Fast shocks (small |du|, large
|s|) freeze early; slow shocks (large |du|, small |s|) don't freeze
within [0, T].

This script sweeps (u_L=0, u_R) for u_R in linspace(0.05, 0.95, N_U_R),
runs each model in RUNS_TO_TEST on every IC, detects the per-sample
freeze time (first t where the predicted shock position deviates from
the analytical truth by more than k_x * dx), and plots t_freeze vs 1/|s|
overlaid with the theoretical cone line.

Edit RUNS_TO_TEST below to control which runs get tested.
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
# EDIT THIS LIST: absolute paths to run directories to evaluate.
# Each entry must contain config.yaml and model_final.pt.
# --------------------------------------------------------------------- #
RUNS_TO_TEST: list[str] = [
    "/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_st3/run_20260505_181417"
]

N_U_R: int               = 20
U_R_MIN: float           = 0.05
U_R_MAX: float           = 0.95
X_SPLIT: float           = 0.0
SHOCK_THRESH_MULT: float = 1.0   # freeze when |x_pred - x_truth| > MULT * k_x * dx


# --------------------------------------------------------------------- #
# Model construction (mirrors eval_ood.py::_build_model verbatim)
# --------------------------------------------------------------------- #
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
    """For each row of u_field [nt, nx], return x at argmax |Δu|.

    Returns array of shape [nt].
    """
    grad = np.abs(np.diff(u_field, axis=1))                # [nt, nx-1]
    idx = np.argmax(grad, axis=1)                          # [nt]
    return x[idx]


def _detect_freeze(
    t: np.ndarray,
    x_pred_track: np.ndarray,
    x_truth_track: np.ndarray,
    threshold: float,
) -> float:
    """Smallest t where |x_pred - x_truth| > threshold; t[-1] if never."""
    err = np.abs(x_pred_track - x_truth_track)
    bad = err > threshold
    if not bad.any():
        return float(t[-1])
    return float(t[np.argmax(bad)])


def _evaluate_run(
    run_dir: Path,
    device: torch.device,
) -> dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_runtime_overrides(cfg)

    data_cfg  = cfg["data"]
    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st")))

    nx     = int(data_cfg["nx"])
    nt     = int(data_cfg["nt"])
    x_min  = float(data_cfg["x_min"])
    x_max  = float(data_cfg["x_max"])
    t_max  = float(data_cfg["t_max"])

    x_np = np.linspace(x_min, x_max, nx, dtype=np.float32)
    t_np = np.linspace(0.0, t_max, nt, dtype=np.float32)
    dx   = float(x_np[1] - x_np[0])

    n_layers   = int(model_cfg.get("n_layers", 6))
    k_x        = int(model_cfg.get("stencil_k_x", 3))
    cone_reach = n_layers * k_x * dx
    threshold  = SHOCK_THRESH_MULT * k_x * dx
    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"

    print(f"  config: nx={nx}, nt={nt}, dx={dx:.5f}, t_max={t_max}")
    print(f"  arch:   n_layers={n_layers}, k_x={k_x}, cone_reach={cone_reach:.4f}")
    print(f"  freeze threshold (1 stencil radius) = {threshold:.4f}")

    # Build & load model
    weights_path = run_dir / "model_final.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"No model_final.pt in {run_dir}")
    model = _build_model(model_cfg, device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  loaded weights ({n_params:,} params)")

    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)

    plot_dir = run_dir / "plots_riemann_sweep"
    plot_dir.mkdir(parents=True, exist_ok=True)

    u_r_values = np.linspace(U_R_MIN, U_R_MAX, N_U_R)
    s_values: list[float] = []
    t_freeze_values: list[float] = []
    t_freeze_theory: list[float] = []
    pred_fields: list[np.ndarray] = []
    truth_tracks: list[np.ndarray] = []
    pred_tracks:  list[np.ndarray] = []

    for i, u_R in enumerate(u_r_values):
        u_L = 0.0
        s = 1.0 - (u_L + u_R)                  # > 0 for u_L=0, u_R<1
        u0_np = riemann_ic(x_np, u_left=u_L, u_right=float(u_R), x0=X_SPLIT)

        u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device).unsqueeze(0)
        if not skip_ef:
            ef_adj, ef_nonadj = precompute_lwr_edge_features_v3(u0_t, x_grid, k_x)
            ef_adj = ef_adj.to(device)
            ef_nonadj = ef_nonadj.to(device) if ef_nonadj.numel() > 0 else None
        else:
            ef_adj, ef_nonadj = None, None

        with torch.no_grad():
            pred_t, _, _, _ = model(
                u0_t, x_grid, t_grid,
                edge_feats_adj=ef_adj, edge_feats_nonadj=ef_nonadj,
            )
        pred_np = pred_t[0].cpu().numpy()                     # [nt, nx]

        x_truth = X_SPLIT + s * t_np                          # [nt]
        x_pred  = _detect_shock_position(pred_np, x_np)       # [nt]

        t_freeze       = _detect_freeze(t_np, x_pred, x_truth, threshold)
        t_freeze_pred  = cone_reach / max(abs(s), 1e-12)
        t_freeze_pred_clipped = min(t_freeze_pred, t_max)

        s_values.append(float(s))
        t_freeze_values.append(t_freeze)
        t_freeze_theory.append(t_freeze_pred_clipped)
        pred_fields.append(pred_np)
        truth_tracks.append(x_truth)
        pred_tracks.append(x_pred)

        print(f"    u_R={u_R:.3f}  s={s:.3f}  t_freeze={t_freeze:.3f}  "
              f"theory={t_freeze_pred_clipped:.3f}")

    s_arr      = np.asarray(s_values)
    tf_arr     = np.asarray(t_freeze_values)
    th_arr     = np.asarray(t_freeze_theory)
    inv_s_arr  = 1.0 / np.abs(s_arr)

    # ---- Plot 1: t_freeze vs 1/|s| with theoretical line --------------- #
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(inv_s_arr, tf_arr, c="tab:blue", s=40, label="measured t_freeze",
               zorder=3)
    inv_s_dense = np.linspace(inv_s_arr.min(), inv_s_arr.max(), 200)
    theory_dense = np.minimum(cone_reach * inv_s_dense, t_max)
    ax.plot(inv_s_dense, theory_dense, "k--",
            label=f"theory: n_layers·k_x·dx / |s|  = {cone_reach:.3f} / |s|  "
                  f"(clipped at t_max={t_max:.2f})")
    # also fit a line through measurements (no intercept) for slope sanity
    not_clipped = tf_arr < t_max - 1e-6
    if not_clipped.sum() >= 2:
        slope_fit = (tf_arr[not_clipped] / inv_s_arr[not_clipped]).mean()
        ax.plot(inv_s_dense, slope_fit * inv_s_dense, "tab:red", alpha=0.5,
                label=f"empirical slope (mean t_f·|s|) = {slope_fit:.3f}")
    ax.set_xlabel("1 / |s|")
    ax.set_ylabel("t_freeze")
    ax.set_title(f"{run_dir.name} — Riemann sweep, u_L=0, u_R∈[{U_R_MIN}, {U_R_MAX}]\n"
                 f"n_layers={n_layers}, k_x={k_x}, dx={dx:.4f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(plot_dir / "t_freeze_vs_inv_s.png", dpi=150)
    plt.close(fig)

    # ---- Plot 2: tracks grid (one panel per u_R) ----------------------- #
    n_cols = 5
    n_rows = int(np.ceil(N_U_R / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3 * n_cols, 2.6 * n_rows),
                              constrained_layout=True)
    axes = axes.flatten()
    for i in range(N_U_R):
        ax = axes[i]
        u_R = u_r_values[i]
        s_i = s_values[i]
        ax.pcolormesh(x_np, t_np, pred_fields[i], shading="auto", cmap="jet",
                      vmin=0.0, vmax=max(0.05, float(u_R)))
        ax.plot(truth_tracks[i], t_np, "k-",  lw=1.2, label="truth")
        ax.plot(pred_tracks[i],  t_np, "w--", lw=1.0, label="pred")
        ax.axhline(t_freeze_values[i], color="red",   lw=0.8, alpha=0.6)
        ax.axhline(th_arr[i],          color="cyan",  lw=0.8, alpha=0.6,
                   linestyle=":")
        ax.set_title(f"u_R={u_R:.2f}, s={s_i:.2f}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("t")
    for j in range(N_U_R, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{run_dir.name} — prediction with truth (black) "
                 f"and predicted shock track (white dashed). "
                 f"Red=measured t_freeze, cyan=theory.",
                 fontsize=10)
    fig.savefig(plot_dir / "tracks.png", dpi=130)
    plt.close(fig)

    # ---- metrics.txt --------------------------------------------------- #
    with (plot_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"Run: {run_dir}\n")
        f.write(f"n_layers={n_layers}  k_x={k_x}  dx={dx:.6f}  "
                f"cone_reach={cone_reach:.6f}  t_max={t_max:.4f}\n")
        f.write(f"freeze threshold = {threshold:.6f} "
                f"(= {SHOCK_THRESH_MULT} * k_x * dx)\n\n")
        f.write(f"{'u_R':>6}  {'s':>7}  {'1/|s|':>7}  "
                f"{'t_freeze':>9}  {'t_theory':>9}  {'ratio':>6}\n")
        for u_R, s, tf, th in zip(u_r_values, s_arr, tf_arr, th_arr):
            ratio = tf / th if th > 0 else float("nan")
            f.write(f"{u_R:6.3f}  {s:7.3f}  {1.0/abs(s):7.3f}  "
                    f"{tf:9.4f}  {th:9.4f}  {ratio:6.2f}\n")

    return {
        "run_dir":   run_dir,
        "n_layers":  n_layers,
        "k_x":       k_x,
        "dx":        dx,
        "t_max":     t_max,
        "u_r":       u_r_values,
        "s":         s_arr,
        "t_freeze":  tf_arr,
        "t_theory":  th_arr,
    }


def main() -> None:
    if not RUNS_TO_TEST:
        raise SystemExit(
            "RUNS_TO_TEST is empty.  Edit riemann_sweep_freeze.py and add "
            "absolute paths to run directories at the top of the file."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results: list[dict] = []
    for run_str in RUNS_TO_TEST:
        run_dir = Path(run_str)
        print(f"\n=== Run: {run_dir} ===")
        if not run_dir.exists():
            print(f"  SKIP — directory does not exist")
            continue
        try:
            results.append(_evaluate_run(run_dir, device))
        except Exception as e:
            print(f"  ERROR — {e}")
            continue

    if not results:
        print("\nNo successful runs.")
        return

    # ---- Combined overlay: all runs on one t_freeze vs 1/|s| ----------- #
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 10)))
    for i, r in enumerate(results):
        inv_s = 1.0 / np.abs(r["s"])
        cone_reach = r["n_layers"] * r["k_x"] * r["dx"]
        label = (f"{r['run_dir'].name}  "
                 f"(n_l={r['n_layers']}, k_x={r['k_x']}, "
                 f"cone={cone_reach:.3f})")
        ax.scatter(inv_s, r["t_freeze"], color=colors[i], s=35,
                   label=label, zorder=3)
        inv_s_dense = np.linspace(inv_s.min(), inv_s.max(), 200)
        theory = np.minimum(cone_reach * inv_s_dense, r["t_max"])
        ax.plot(inv_s_dense, theory, color=colors[i], linestyle="--",
                alpha=0.6)
    ax.set_xlabel("1 / |s|")
    ax.set_ylabel("t_freeze")
    ax.set_title("Combined: t_freeze vs 1/|s|  (dashed = theoretical cone)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    out_path = results[0]["run_dir"].parent / "riemann_sweep_combined.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nCombined overlay saved to {out_path}")

    print("\n=== Summary ===")
    for r in results:
        cone = r["n_layers"] * r["k_x"] * r["dx"]
        not_clipped = r["t_freeze"] < r["t_max"] - 1e-6
        if not_clipped.any():
            slope_fit = (r["t_freeze"][not_clipped]
                         / (1.0 / np.abs(r["s"][not_clipped]))).mean()
            print(f"  {r['run_dir'].name}: cone_reach={cone:.4f}  "
                  f"empirical_slope={slope_fit:.4f}  "
                  f"ratio={slope_fit/cone:.3f}")
        else:
            print(f"  {r['run_dir'].name}: cone_reach={cone:.4f}  "
                  f"(all samples clipped at t_max — no freeze observed)")


if __name__ == "__main__":
    main()
