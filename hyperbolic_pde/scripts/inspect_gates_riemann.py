"""Inspect per-layer physics-gate values around a query point on a Riemann IC.

Hypothesis (from theory note + recent discussion): for a Riemann IC where the
discontinuity at x_0 sits outside the receptive field of a query point
(x_*, T), the model should still be able to recover the shock via "source (2)":
read intermediate-time latents along the shock trajectory inside the spatial
strip. If the physics gates (esp. CFL and upwind on temporal/diagonal edges)
heavily attenuate exactly those messages, source (2) is choked.

This script:
  1. Loads a trained HypNO-ST3 run.
  2. Builds a Riemann IC (u_L=0, u_R=USER_CHOICE).
  3. Runs a forward pass while capturing each MP layer's input latent h^(ell-1).
  4. For each (layer, query_time) chosen by the user, recomputes the per-edge
     physics gate value for every (di, dm) offset in the stencil, producing a
     [k_t+1, 2k_x+1] heatmap. (k_t+1 rows because causal: dm in [-k_t, 0].)
  5. Saves a grid of heatmaps: rows = layers, columns = query times.

Usage:
  python hyperbolic_pde/scripts/inspect_gates_riemann.py \
      --run-dir <path> \
      --u_R 0.2 \
      --query_x 0.5 \
      --times 0.25 0.5 0.75 1.0 \
      [--layers 0 1 2 3 4 5 6 7 8]

Outputs:
  <run_dir>/plots_gate_inspect/gates_uR<u_R>_x<query_x>.png
  <run_dir>/plots_gate_inspect/pred_field_uR<u_R>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.utils.runtime import apply_runtime_overrides
from hyperbolic_pde.models.hypno_st3 import (
    HypNO_ST3,
    _PhysicsSpaceTimeMPLayer,
    _pad_space_time,
    _enumerate_ball_offsets,
    _compute_adj_spatial_edge_feats,
    precompute_lwr_edge_features_v3,
)
from hyperbolic_pde.scripts.eval_riemann_shock import riemann_ic


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
        pure_pairwise_edges=bool(model_cfg.get("pure_pairwise_edges", False)),
    ).to(device)


def _capture_layer_inputs(model: HypNO_ST3) -> list[torch.Tensor]:
    """Hook every _PhysicsSpaceTimeMPLayer to capture its input h^(ell-1).

    Also captures the output of the last MP layer (i.e. h^(L)) by appending it
    to the returned list. Final list length is n_layers + 1:
      captured[0]   = h^(0)   (after lifting, before MP)
      captured[ell] = h^(ell) (after ell MP layers)  for ell = 1..L
    """
    inputs_captured: list[torch.Tensor] = []
    final_output: list[torch.Tensor] = []
    handles = []

    def pre_hook(_module, inputs):
        inputs_captured.append(inputs[0].detach())

    def last_post_hook(_module, _inputs, output):
        final_output.append(output.detach())

    for layer in model.mp_layers:
        handles.append(layer.register_forward_pre_hook(pre_hook))
    # Post-hook on the *last* MP layer to grab h^(L).
    handles.append(model.mp_layers[-1].register_forward_hook(last_post_hook))

    return (inputs_captured, final_output), handles


def _gates_for_query(
    layer: _PhysicsSpaceTimeMPLayer,
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    query_i: int,
    query_n: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Recompute physics-gate values over the neighborhood patch of (query_i, query_n).

    Returns:
        grid:     [k_t+1, 2*k_x+1] heatmap (causal: dm in [-k_t, 0]).
                  rows: dm = 0 (top) to dm = -k_t (bottom)
                  cols: di = -k_x (left) to di = +k_x (right)
        di_axis:  list of di values for x-axis
        dm_axis:  list of dm values for y-axis (top to bottom)
    """
    B, nt, nx, d = h.shape
    dx_val = float((x[0, 1] - x[0, 0]).abs().item())
    k_x = layer.k_x
    k_t = layer.k_t

    u_hat = torch.sigmoid(layer.state_probe(h)).squeeze(-1)             # [B, nt, nx]
    u_hat_pad = _pad_space_time(u_hat.unsqueeze(-1), k_x, k_t).squeeze(-1)
    x_pad = torch.nn.functional.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
    t_pad = torch.nn.functional.pad(
        t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate"
    ).squeeze(0).squeeze(0)

    # Query indices in padded coords
    qi_pad = k_x + query_i
    qn_pad = k_t + query_n

    # Source-node scalars at the query
    u_i = u_hat_pad[0, qn_pad, qi_pad].view(1, 1, 1, 1)                # [1,1,1,1]
    a_i = 1.0 - 2.0 * u_i
    x_i = x[0, query_i].view(1, 1, 1, 1)
    t_i = t[query_n].view(1, 1, 1, 1)

    offsets = _enumerate_ball_offsets(k_x, k_t, layer.causal)

    di_axis = list(range(-k_x, k_x + 1))
    dm_axis = list(range(0, -k_t - 1, -1))           # 0, -1, -2, ..., -k_t  (top to bottom)
    grid = np.full((len(dm_axis), len(di_axis)), np.nan, dtype=np.float32)

    for di, dm in offsets:
        # Source MP layer skips same-time non-adj if masked
        if layer.mask_same_t_nonadj and dm == 0 and abs(di) > 1:
            continue

        u_j = u_hat_pad[0, qn_pad + dm, qi_pad + di].view(1, 1, 1, 1)
        a_j = 1.0 - 2.0 * u_j
        x_j = x_pad[0, qi_pad + di].view(1, 1, 1, 1)
        t_j = t_pad[qn_pad + dm].view(1, 1, 1, 1)
        rel_x = x_j - x_i
        rel_t = t_j - t_i

        is_adj_sp = (dm == 0) and (abs(di) == 1)
        if is_adj_sp:
            _, _, _, _, _, _, a_ij, _sign_a, _upwind = \
                _compute_adj_spatial_edge_feats(u_i, u_j, rel_x)
        else:
            a_ij = torch.zeros_like(rel_x)

        gate = layer._ball_physics_gate(
            di=di, dm=dm,
            u_i=u_i, u_j=u_j,
            rel_x=rel_x, rel_t=rel_t,
            a_i=a_i, a_j=a_j, a_ij=a_ij,
            t_i=t_i, dx_grid=dx_val,
        )
        row = dm_axis.index(dm)
        col = di_axis.index(di)
        grid[row, col] = float(gate.item())

    return grid, di_axis, dm_axis


def main() -> None:
    # ---- HARDCODED PARAMETERS (edit here) ---- #
    RUN_DIR = "/home/dzdrale/DL-for-HPDE/hyperbolic_pde/runs/hypno_st3/run_20260507_123353"
    U_R     = 0.2
    X_0     = 0.0
    QUERY_X = 0.6
    TIMES   = [0.25, 0.5, 0.75, 1.0]
    LAYERS  = None        # None = all MP layers
    # ------------------------------------------ #

    # Compatibility shim so the rest of main() can keep using args.*
    class _Args:
        pass
    args = _Args()
    args.run_dir = RUN_DIR
    args.u_R     = U_R
    args.x_0     = X_0
    args.query_x = QUERY_X
    args.times   = TIMES
    args.layers  = LAYERS

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_runtime_overrides(cfg)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = cfg["data"]
    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st")))

    nx = int(data_cfg["nx"])
    nt = int(data_cfg["nt"])
    x_min = float(data_cfg["x_min"])
    x_max = float(data_cfg["x_max"])
    t_max = float(data_cfg["t_max"])
    x_np = np.linspace(x_min, x_max, nx, dtype=np.float32)
    t_np = np.linspace(0.0, t_max, nt, dtype=np.float32)
    dx = float(x_np[1] - x_np[0])

    n_layers = int(model_cfg.get("n_layers", 6))
    k_x = int(model_cfg.get("stencil_k_x", 3))
    k_t = int(model_cfg.get("stencil_k_t", 2))
    cone_reach = n_layers * k_x * dx
    s = 1.0 - args.u_R
    t_freeze_theory = cone_reach / max(abs(s), 1e-12)
    print(f"Run: {run_dir}")
    print(f"  Architecture: n_layers={n_layers}, k_x={k_x}, k_t={k_t}, dx={dx:.5f}")
    print(f"  Receptive spatial half-reach (per query): L*k_x*dx = {cone_reach:.4f}")
    print(f"  Shock speed s = 1 - (u_L+u_R) = {s:.3f}")
    print(f"  Theoretical t_freeze = L*k_x*dx / |s| = {t_freeze_theory:.4f}")

    # Build & load model
    model = _build_model(model_cfg, device)
    weights_path = run_dir / "model_final.pt"
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    # Build Riemann IC
    u0_np = riemann_ic(x_np, u_left=0.0, u_right=float(args.u_R), x0=float(args.x_0))
    u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device).unsqueeze(0)
    # x_grid is passed 1D to the model; HypNO_ST3.forward auto-expands to [B, nx].
    x_grid_1d = torch.tensor(x_np, dtype=torch.float32, device=device)
    x_grid = x_grid_1d.unsqueeze(0)                  # [1, nx] — matches what MP layer sees
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)

    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"
    if not skip_ef:
        ef_adj, ef_nonadj = precompute_lwr_edge_features_v3(u0_t, x_grid_1d, k_x)
        ef_adj = ef_adj.to(device)
        ef_nonadj = ef_nonadj.to(device) if ef_nonadj.numel() > 0 else None
    else:
        ef_adj, ef_nonadj = None, None

    # Forward pass with hooks; pass x as 1D — model expands internally.
    (inputs_cap, output_cap), handles = _capture_layer_inputs(model)
    with torch.no_grad():
        pred_t, _, _, _ = model(
            u0_t, x_grid_1d, t_grid,
            edge_feats_adj=ef_adj, edge_feats_nonadj=ef_nonadj,
        )
    for h in handles:
        h.remove()
    pred_np = pred_t[0].cpu().numpy()                                  # [nt, nx]
    # Gates need h^(ell-1) -- the input to layer ell -- so we keep inputs_cap as `captured`.
    captured = inputs_cap
    # h-list spanning all stages: h^(0) (input to layer 0) ... h^(L) (output of last layer).
    all_h = inputs_cap + output_cap
    print(f"  Captured h for {len(captured)} MP layers; pred shape {pred_np.shape}")

    # Resolve query indices
    query_i = int(np.argmin(np.abs(x_np - args.query_x)))
    print(f"  Query point: x={x_np[query_i]:.4f} (idx {query_i})")
    print(f"  Receptive strip at query: [{x_np[query_i] - cone_reach:.4f}, "
          f"{x_np[query_i] + cone_reach:.4f}]")
    print(f"  IC discontinuity at x_0={args.x_0:.4f} is "
          f"{'INSIDE' if abs(args.x_0 - x_np[query_i]) <= cone_reach else 'OUTSIDE'} the strip.")

    query_ns = [int(np.argmin(np.abs(t_np - tq))) for tq in args.times]

    layer_indices = args.layers if args.layers is not None else list(range(n_layers))

    # Compute gate grids: shape (n_layers, n_times, k_t+1, 2k_x+1)
    grids = np.full(
        (len(layer_indices), len(query_ns), k_t + 1, 2 * k_x + 1),
        np.nan, dtype=np.float32,
    )
    di_axis = list(range(-k_x, k_x + 1))
    dm_axis = list(range(0, -k_t - 1, -1))

    for li, ell in enumerate(layer_indices):
        layer = model.mp_layers[ell]
        h_in = captured[ell]
        for ti, qn in enumerate(query_ns):
            grid, _, _ = _gates_for_query(layer, h_in, x_grid, t_grid, query_i, qn)
            grids[li, ti] = grid

    # --- Plot 1: gate heatmap grid ---
    out_dir = run_dir / "plots_gate_inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(layer_indices)
    n_cols = len(query_ns)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 + 2.2 * n_cols, 1.0 + 1.6 * n_rows),
        squeeze=False, constrained_layout=True,
    )

    for li, ell in enumerate(layer_indices):
        for ti, qn in enumerate(query_ns):
            ax = axes[li][ti]
            grid = grids[li, ti]
            im = ax.imshow(
                grid, cmap="viridis", vmin=0.0, vmax=1.0,
                aspect="auto",
                extent=[di_axis[0] - 0.5, di_axis[-1] + 0.5,
                        dm_axis[-1] - 0.5, dm_axis[0] + 0.5],
                origin="upper",
            )
            # Annotate values
            for r, dm in enumerate(dm_axis):
                for c, di in enumerate(di_axis):
                    v = grid[r, c]
                    if not np.isnan(v):
                        ax.text(di, dm, f"{v:.2f}", ha="center", va="center",
                                fontsize=6, color="white" if v < 0.5 else "black")
            ax.set_xticks(di_axis)
            ax.set_yticks(dm_axis)
            ax.set_xlabel("di (space offset)")
            ax.set_ylabel("dm (time offset, causal ≤ 0)")
            t_q = t_np[query_ns[ti]]
            x_q = x_np[query_i]
            shock_at_t = args.x_0 + s * t_q
            ax.set_title(
                f"layer {ell} | t={t_q:.3f} (qn={query_ns[ti]})\n"
                f"x*={x_q:.3f} shock@t={shock_at_t:.3f}", fontsize=8,
            )
    fig.colorbar(im, ax=axes, shrink=0.7, label="physics gate value")
    fig.suptitle(
        f"Physics-gate values over neighborhood patch (causal stencil k_x={k_x}, k_t={k_t})\n"
        f"u_R={args.u_R}, s={s:.3f}, query x*={x_np[query_i]:.3f}, "
        f"t_freeze_theory={t_freeze_theory:.3f}",
        fontsize=10,
    )
    fig_path = out_dir / f"gates_uR{args.u_R}_x{args.query_x}.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {fig_path}")

    # --- Plot 2: prediction field with query marker, shock track, and patch outlines ---
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    pcm = ax.pcolormesh(x_np, t_np, pred_np, shading="auto",
                        cmap="jet", vmin=0.0, vmax=max(0.05, float(args.u_R)))
    ax.plot(args.x_0 + s * t_np, t_np, "k-", lw=1.2, label="truth shock")
    ax.axvline(x_np[query_i], color="white", ls=":", lw=1.0, alpha=0.9,
               label=f"query x*={x_np[query_i]:.3f}")
    ax.axvline(x_np[query_i] - cone_reach, color="cyan", lw=1.0, ls="--", alpha=0.8,
               label=f"recept. strip (±{cone_reach:.3f})")
    ax.axvline(x_np[query_i] + cone_reach, color="cyan", lw=1.0, ls="--", alpha=0.8)
    ax.axhline(t_freeze_theory, color="red", lw=0.8, alpha=0.7,
               label=f"t_freeze_theory={t_freeze_theory:.3f}")
    # Draw the neighborhood patch for each (query_x, t_q): exactly the cells
    # whose gate values appear in the heatmap at that (layer, t_q).
    # Patch spans di in [-k_x, +k_x] and dm in [-k_t, 0] from the query node.
    half_dx = 0.5 * dx
    half_dt = 0.5 * (t_np[1] - t_np[0])
    for ti, qn in enumerate(query_ns):
        x_q = x_np[query_i]
        t_q = t_np[qn]
        # Spatial bounds: cells i_q+di for di in [-k_x, k_x]
        left  = x_q - k_x * dx - half_dx
        right = x_q + k_x * dx + half_dx
        # Temporal bounds: cells n_q+dm for dm in [-k_t, 0]  (causal)
        bottom = t_q - k_t * (t_np[1] - t_np[0]) - half_dt
        top    = t_q + half_dt
        rect = plt.Rectangle(
            (left, bottom), right - left, top - bottom,
            fill=False, edgecolor="magenta", lw=1.4, ls="-", alpha=0.9,
        )
        ax.add_patch(rect)
        # Mark the query cell itself with a small dot
        ax.plot(x_q, t_q, marker="o", color="magenta", markersize=4,
                markeredgecolor="black", markeredgewidth=0.4)
        ax.annotate(
            f"t={t_q:.3f}",
            xy=(right, t_q), xytext=(3, 0), textcoords="offset points",
            color="magenta", fontsize=8, va="center",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(
        f"Prediction field — Riemann u_L=0, u_R={args.u_R}, "
        f"x_0={args.x_0}\nmagenta boxes = neighborhood patches "
        f"(±k_x={k_x} in space, k_t={k_t} causal back in time) for each query t")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(pcm, ax=ax, label="u")
    fig_path2 = out_dir / f"pred_field_uR{args.u_R}.png"
    fig.savefig(fig_path2, dpi=140)
    plt.close(fig)
    print(f"  wrote {fig_path2}")

    # --- Plot 3: per-layer probed u_hat^(ell) over space-time ---
    # For each layer, apply that layer's state_probe to h^(ell-1); also probe
    # the final h^(L) using the last layer's probe (a reasonable proxy for the
    # decoder's view, since the decoder operates on h^(L) directly).
    u_hats = []
    for ell, h_in in enumerate(captured):
        layer = model.mp_layers[ell]
        with torch.no_grad():
            uh = torch.sigmoid(layer.state_probe(h_in)).squeeze(-1)[0].cpu().numpy()  # [nt, nx]
        u_hats.append((f"u_hat^({ell})  (before layer {ell})", uh))
    # Final h^(L): use the last layer's state_probe
    if len(output_cap) > 0:
        with torch.no_grad():
            uh_L = torch.sigmoid(
                model.mp_layers[-1].state_probe(output_cap[0])
            ).squeeze(-1)[0].cpu().numpy()
        u_hats.append((f"u_hat^({n_layers})  (after last MP, probe of last layer)", uh_L))
    # Final prediction
    u_hats.append(("u_pred (decoder)", pred_np))

    n_panels = len(u_hats)
    n_cols_uh = min(4, n_panels)
    n_rows_uh = int(np.ceil(n_panels / n_cols_uh))
    fig, axes = plt.subplots(
        n_rows_uh, n_cols_uh,
        figsize=(3.4 * n_cols_uh, 2.6 * n_rows_uh),
        squeeze=False, constrained_layout=True,
    )
    vmax_uh = max(0.05, float(args.u_R))
    for idx, (title, field) in enumerate(u_hats):
        r, c = idx // n_cols_uh, idx % n_cols_uh
        ax = axes[r][c]
        pcm = ax.pcolormesh(x_np, t_np, field, shading="auto",
                            cmap="jet", vmin=0.0, vmax=vmax_uh)
        ax.plot(args.x_0 + s * t_np, t_np, "k-", lw=0.9, alpha=0.8)
        ax.axvline(x_np[query_i], color="white", ls=":", lw=0.6, alpha=0.9)
        ax.axvline(x_np[query_i] - cone_reach, color="cyan", lw=0.6, ls="--", alpha=0.6)
        ax.axvline(x_np[query_i] + cone_reach, color="cyan", lw=0.6, ls="--", alpha=0.6)
        ax.axhline(t_freeze_theory, color="red", lw=0.6, alpha=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("x"); ax.set_ylabel("t")
    for idx in range(n_panels, n_rows_uh * n_cols_uh):
        r, c = idx // n_cols_uh, idx % n_cols_uh
        axes[r][c].axis("off")
    fig.colorbar(pcm, ax=axes, shrink=0.6, label="u_hat / u_pred")
    fig.suptitle(
        f"Per-layer probed u_hat^(ell) — smearing/oversmoothing diagnostic\n"
        f"u_R={args.u_R}, s={s:.3f}, t_freeze_theory={t_freeze_theory:.3f}",
        fontsize=10,
    )
    fig_path3 = out_dir / f"u_hat_per_layer_uR{args.u_R}.png"
    fig.savefig(fig_path3, dpi=140)
    plt.close(fig)
    print(f"  wrote {fig_path3}")

    # --- Plot 4: u_hat^(ell)(x_*, t) at the query column, all layers overlaid ---
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    cmap_layers = plt.get_cmap("viridis", len(u_hats))
    for idx, (title, field) in enumerate(u_hats):
        col = field[:, query_i]                                  # [nt]
        ax.plot(t_np, col, color=cmap_layers(idx),
                lw=1.2, alpha=0.85, label=title.split("  ")[0])
    # Truth: u_L=0 for t < (x* - x_0)/s, then u_R after shock passes.
    t_shock_passes = (x_np[query_i] - args.x_0) / s if s > 0 else np.inf
    truth_col = np.where(t_np < t_shock_passes, 0.0, args.u_R)
    ax.plot(t_np, truth_col, "k--", lw=1.5, label=f"truth (shock passes at t={t_shock_passes:.3f})")
    ax.axvline(t_freeze_theory, color="red", lw=0.8, alpha=0.7,
               label=f"t_freeze_theory={t_freeze_theory:.3f}")
    ax.set_xlabel(f"t (at x_* = {x_np[query_i]:.3f})")
    ax.set_ylabel("u_hat^(ell)")
    ax.set_title(
        f"Per-layer probed u_hat at the query column\n"
        f"u_R={args.u_R}, s={s:.3f}, x_0={args.x_0}, x_*={x_np[query_i]:.3f}"
    )
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig_path4 = out_dir / f"u_hat_query_column_uR{args.u_R}_x{args.query_x}.png"
    fig.savefig(fig_path4, dpi=140)
    plt.close(fig)
    print(f"  wrote {fig_path4}")

    # --- Also print learned gate hyperparams for context ---
    print("\n  Learned gate parameters per MP layer:")
    print("    layer | softplus(temperature) | sigmoid(gamma_entropy) | "
          "softplus(cfl_scale) | softplus(time_decay)")
    for ell, layer in enumerate(model.mp_layers):
        tau = torch.nn.functional.softplus(layer.phys_temperature).item()
        gam = torch.sigmoid(layer.phys_gamma_entropy).item()
        cfl = torch.nn.functional.softplus(layer.phys_cfl_scale).item()
        tdec = torch.nn.functional.softplus(layer.phys_time_decay).item()
        print(f"    {ell:5d} | {tau:.4e}          | {gam:.4f}                  "
              f"| {cfl:.4e}        | {tdec:.4e}")


if __name__ == "__main__":
    main()
