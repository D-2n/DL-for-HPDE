"""Per-layer decoder readout vs Lax-Hopf truth, aggregated over the test set.

For each test sample, this script:
  1. Runs HypNO-ST3 forward.
  2. Captures h^(ell) for every MP layer (and h^(L) after the last one).
  3. Applies model.decoder(h^(ell)) (+ u0 if model.skip) to each captured latent,
     producing a per-layer "predicted u-field".
  4. Compares each layer's field to the Lax-Hopf exact solution: MAE, rL2,
     per-t MAE.
  5. Aggregates across the test set and reports / plots per-layer metrics.

This isolates *what does each MP layer's latent decode to, when read through
the trained decoder?* — orthogonal to the state_probe diagnostic which uses
per-layer linear probes that are barely trained at lambda_probe=0.01.

Usage (CLEPS):
  sbatch slurm/eval_vs_numerical_decoder.sh [run_dir] [n_samples]

Outputs:
  <plot_dir>/per_layer_metrics.txt
  <plot_dir>/per_layer_mae.png           — bar plot, per-layer mean MAE
  <plot_dir>/per_layer_error_vs_time.png — per-layer mean per-t MAE curves
  <plot_dir>/per_layer_sample_<idx>.png  — for up to n_plots samples, all layer fields
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from hyperbolic_pde.utils.runtime import apply_runtime_overrides, resolve_config_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset
from hyperbolic_pde.data.lax_hopf import solve_lax_hopf
from hyperbolic_pde.models.hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> dict:
    base_path = ROOT / "configs" / "hyperbolic_pde.yaml"
    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path.resolve() == base_path.resolve():
        return cfg
    with path.open("r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    return _deep_update(cfg, override or {})


def split_indices(n: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_fraction * n)
    return idx[:n_train], idx[n_train:]


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.abs(pred - truth).mean())


def rel_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(((pred - truth) ** 2).sum() / ((truth ** 2).sum() + 1e-12)))


def per_t_mae(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(pred - truth).mean(axis=1)


def print_large_error_cells(
    pred: np.ndarray,
    truth: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
    max_lines: int = 50,
    exclude_t0: bool = True,
    boundary_buffer: int = 5,
) -> int:
    """Print every cell where |pred - truth| > threshold.

    Filters:
      exclude_t0      — drop the t_idx == 0 row (IC-regression failure,
                        handled separately).
      boundary_buffer — flag cells within `boundary_buffer` of either spatial
                        edge as 'bd' in the output table. Boundary-condition
                        mismatches between Lax-Hopf and the solver under test
                        appear here, not as model failures. Set to 0 to skip
                        the tag. Cells are NOT excluded by this flag.
    """
    err = np.abs(pred - truth)
    bad_mask = err > threshold
    if exclude_t0:
        bad_mask[0, :] = False
    n_bad = int(bad_mask.sum())
    nt_total, nx_total = pred.shape
    n_considered = pred.size - (nx_total if exclude_t0 else 0)
    if boundary_buffer > 0 and n_bad > 0:
        ti_all, xi_all = np.where(bad_mask)
        bd_cnt = int(np.sum(
            (xi_all < boundary_buffer) | (xi_all >= nx_total - boundary_buffer)
        ))
        bd_suffix = f" ({bd_cnt} near boundary, {n_bad - bd_cnt} interior)"
    else:
        bd_suffix = ""
    header = (f"Cells with |pred - truth| > {threshold}"
              f"{(' [' + label + ']') if label else ''}"
              f"{' (excl. t=0)' if exclude_t0 else ''}: "
              f"{n_bad} cells (of {n_considered} considered){bd_suffix}")
    print(header)
    if n_bad == 0:
        return 0
    ti, xi = np.where(bad_mask)
    errs = err[ti, xi]
    order = np.argsort(-errs)
    n_show = min(n_bad, max_lines)
    print(f"  showing top {n_show} (sorted by abs_err desc):")
    print(f"    {'t_idx':>5}  {'x_idx':>5}  {'t':>8}  {'x':>9}  "
          f"{'pred':>10}  {'truth':>10}  {'abs_err':>10}  {'bd':>3}")
    for k in order[:n_show]:
        i_t, i_x = int(ti[k]), int(xi[k])
        near_bd = (i_x < boundary_buffer) or (i_x >= nx_total - boundary_buffer)
        tag = "bd" if near_bd else ""
        print(f"    {i_t:>5d}  {i_x:>5d}  {float(t[i_t]):>8.4f}  "
              f"{float(x[i_x]):>9.4f}  {float(pred[i_t, i_x]):>10.5f}  "
              f"{float(truth[i_t, i_x]):>10.5f}  "
              f"{float(err[i_t, i_x]):>10.5f}  {tag:>3}")
    if n_bad > n_show:
        print(f"    ... {n_bad - n_show} more cells suppressed")
    return n_bad


def _capture_layer_inputs(model: HypNO_ST3):
    """Pre-hook every MP layer to capture its input h^(ell-1); post-hook the last
    MP layer to capture h^(L). Returns (inputs_list, output_list, handles)."""
    inputs_captured: list[torch.Tensor] = []
    final_output: list[torch.Tensor] = []
    handles = []

    def pre_hook(_module, inputs):
        inputs_captured.append(inputs[0].detach())

    def last_post_hook(_module, _inputs, output):
        final_output.append(output.detach())

    for layer in model.mp_layers:
        handles.append(layer.register_forward_pre_hook(pre_hook))
    handles.append(model.mp_layers[-1].register_forward_hook(last_post_hook))
    return inputs_captured, final_output, handles


def _decode_with_skip(model: HypNO_ST3, h: torch.Tensor, u0_t: torch.Tensor) -> np.ndarray:
    """Apply model.decoder(h) (+ u0 if model.skip), return numpy [nt, nx]."""
    with torch.no_grad():
        out = model.decoder(h).squeeze(-1)                    # [B, nt, nx]
    if bool(model.skip):
        u0_exp = u0_t.unsqueeze(1).expand_as(out)
        out = u0_exp + out
    return out[0].cpu().numpy()


def main() -> None:
    # ---- HARDCODED PARAMETERS (edit here) ---- #
    RUN_DIR    = "/home/dzdrale/scratch/runs/hypno_st3/run_20260507_123353"
    N_SAMPLES  = 50
    N_PLOTS    = 3
    DATA_PATH  = None       # None = use the run's config.yaml data.path
    # ------------------------------------------ #

    class _Args:
        pass
    args = _Args()
    args.run_dir   = RUN_DIR
    args.n_samples = N_SAMPLES
    args.n_plots   = N_PLOTS
    args.data_path = DATA_PATH
    args.config    = str(resolve_config_path(ROOT / "configs"))

    # ---- Load config / run dir ----
    if args.run_dir:
        run_dir = Path(args.run_dir)
        with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"Using config from {run_dir / 'config.yaml'}")
    else:
        latest_path = Path("hyperbolic_pde/runs/hypno_st3/latest_run.txt")
        if latest_path.exists():
            run_dir = Path(latest_path.read_text(encoding="utf-8").strip().splitlines()[-1].strip())
            with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            print(f"Using latest run: {run_dir}")
        else:
            cfg = load_config(Path(args.config))
            run_dir = None
    cfg = apply_runtime_overrides(cfg)

    data_cfg = cfg["data"]
    model_cfg = cfg.get("hypno_st3", cfg.get("hypno_st2", cfg.get("hypno_st")))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    data_path = Path(args.data_path) if args.data_path else Path(data_cfg["path"])
    print(f"Loading dataset from {data_path}")
    dataset = load_dataset(data_path)

    _, test_idx = split_indices(
        dataset.u.shape[0],
        float(data_cfg["train_fraction"]),
        int(cfg.get("seed", 42)),
    )
    test_idx = test_idx[:args.n_samples]
    print(f"Test set: {len(test_idx)} samples")

    x_np = dataset.x
    t_np = dataset.t
    x_min = float(x_np[0] - 0.5 * (x_np[1] - x_np[0]))
    x_max = float(x_np[-1] + 0.5 * (x_np[1] - x_np[0]))
    t_max = float(t_np[-1])
    nt = len(t_np)
    boundary = str(data_cfg.get("boundary", "ghost"))

    # ---- Build & load model ----
    _dhn = model_cfg.get("d_hidden_nonadj", None)
    d_hidden_nonadj = int(_dhn) if _dhn is not None else None
    n_layers = int(model_cfg.get("n_layers", 6))
    model = HypNO_ST3(
        stencil_k_x=int(model_cfg.get("stencil_k_x", 3)),
        stencil_k_t=int(model_cfg.get("stencil_k_t", 2)),
        d_latent=int(model_cfg.get("d_latent", 128)),
        d_hidden=int(model_cfg.get("d_hidden", 128)),
        n_layers=n_layers,
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

    if run_dir and (run_dir / "model_final.pt").exists():
        weights_path = run_dir / "model_final.pt"
    else:
        weights_path = Path(model_cfg["save_path"])
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded HypNO-ST3 from {weights_path}  (skip={'on' if model.skip else 'off'})")

    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)
    stencil_k_x = int(model_cfg.get("stencil_k_x", 3))
    encoder_type = str(model_cfg.get("encoder_type", "gnn"))
    skip_ef = encoder_type == "mlp"

    # ---- Output dir ----
    plot_dir = (run_dir / "plots_per_layer_decoder") if run_dir \
        else Path("hyperbolic_pde/runs/plots/per_layer_decoder")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Per-layer accumulators. We have n_layers + 1 readout points:
    #   [0..n_layers-1]: decoder(h^(ell)) for ell = 0..L-1 (input to layer ell)
    #   [n_layers]     : decoder(h^(L))   (output of last MP layer)
    # The final entry should match the model's own u_pred.
    n_readouts = n_layers + 1
    layer_labels = [f"dec(h^({ell}))" for ell in range(n_layers + 1)]
    mae_list:  list[list[float]] = [[] for _ in range(n_readouts)]
    rl2_list:  list[list[float]] = [[] for _ in range(n_readouts)]
    pt_list:   list[list[np.ndarray]] = [[] for _ in range(n_readouts)]

    # Sanity: also track the model's own pred MAE (should match the last readout).
    own_mae: list[float] = []

    for plot_i, idx in enumerate(test_idx):
        u0_np = dataset.u0[idx]
        print(f"[{plot_i + 1}/{len(test_idx)}] sample idx={idx}", flush=True)

        # Lax-Hopf exact
        _, _, lh = solve_lax_hopf(u0_np, x_min, x_max, t_max, nt, boundary=boundary)

        # HypNO forward with capture hooks
        u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device).unsqueeze(0)
        if not skip_ef:
            ef_adj, ef_nonadj = precompute_lwr_edge_features_v3(u0_t, x_grid, stencil_k_x)
            ef_adj = ef_adj.to(device)
            ef_nonadj = ef_nonadj.to(device) if ef_nonadj.numel() > 0 else None
        else:
            ef_adj, ef_nonadj = None, None

        inputs_cap, output_cap, handles = _capture_layer_inputs(model)
        with torch.no_grad():
            pred_t, _, _, _ = model(
                u0_t, x_grid, t_grid,
                edge_feats_adj=ef_adj, edge_feats_nonadj=ef_nonadj,
            )
        for h in handles:
            h.remove()
        pred_np = pred_t[0].cpu().numpy()
        own_mae.append(mae(pred_np, lh))

        # Decoder readout per captured latent
        readouts: list[np.ndarray] = []
        for h_in in inputs_cap:
            readouts.append(_decode_with_skip(model, h_in, u0_t))
        if len(output_cap) > 0:
            readouts.append(_decode_with_skip(model, output_cap[0], u0_t))
        assert len(readouts) == n_readouts, (
            f"Expected {n_readouts} readouts, got {len(readouts)}"
        )

        for ell, field in enumerate(readouts):
            mae_list[ell].append(mae(field, lh))
            rl2_list[ell].append(rel_l2(field, lh))
            pt_list[ell].append(per_t_mae(field, lh))

        # Print cells with very large absolute error for each layer's readout.
        for ell, field in enumerate(readouts):
            print_large_error_cells(
                field, lh, x_np, t_np,
                threshold=0.5,
                label=f"sample {idx} | {layer_labels[ell]}",
                max_lines=30,
            )

        # Per-sample plot: show all layer fields side-by-side vs GT for first n_plots
        if plot_i < args.n_plots:
            vmin, vmax = float(lh.min()), float(lh.max())
            # One row of fields + one row of absolute error maps, n_readouts+1 columns
            # (first column = GT)
            cols = [("GT (Lax-Hopf)", lh)] + list(zip(layer_labels, readouts))
            n_cols = len(cols)
            fig, axes = plt.subplots(
                2, n_cols,
                figsize=(2.6 * n_cols, 5.4),
                constrained_layout=True,
            )
            for c, (name, field) in enumerate(cols):
                im = axes[0, c].pcolormesh(
                    x_np, t_np, field, shading="auto",
                    cmap="jet", vmin=vmin, vmax=vmax,
                )
                axes[0, c].set_title(name, fontsize=8)
                axes[0, c].set_xlabel("x"); axes[0, c].set_ylabel("t")
                if c == 0:
                    axes[1, c].axis("off")
                else:
                    err = np.abs(field - lh)
                    im2 = axes[1, c].pcolormesh(
                        x_np, t_np, err, shading="auto",
                        cmap="magma", vmin=0.0,
                    )
                    axes[1, c].set_title(f"|{name} - GT|  MAE={mae(field, lh):.2e}",
                                         fontsize=7)
                    axes[1, c].set_xlabel("x"); axes[1, c].set_ylabel("t")
            fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.7, label="u")
            fig.suptitle(
                f"Sample {idx} — decoder(h^(ell)) vs Lax-Hopf for each MP layer",
                fontsize=10,
            )
            fig.savefig(plot_dir / f"per_layer_sample_{idx}.png", dpi=120)
            plt.close(fig)

    # ---- Aggregate metrics across the test set ----
    mean_mae = np.array([np.mean(m) for m in mae_list])
    std_mae  = np.array([np.std(m)  for m in mae_list])
    mean_rl2 = np.array([np.mean(m) for m in rl2_list])
    std_rl2  = np.array([np.std(m)  for m in rl2_list])

    print("\n=== Per-layer decoder readout vs Lax-Hopf (mean ± std over test set) ===")
    print(f"  N samples = {len(test_idx)}")
    print(f"  layer label             | mean MAE     ± std       | mean rL2     ± std")
    print(f"  ------------------------+---------------------------+--------------------")
    for ell in range(n_readouts):
        print(f"  {layer_labels[ell]:24s}| {mean_mae[ell]:.4e} ± {std_mae[ell]:.4e}"
              f" | {mean_rl2[ell]:.4e} ± {std_rl2[ell]:.4e}")
    own_arr = np.array(own_mae)
    print(f"\n  Sanity: model own u_pred MAE = {own_arr.mean():.4e} (should equal "
          f"{layer_labels[-1]} MAE = {mean_mae[-1]:.4e})")

    # ---- Save metrics.txt ----
    metrics_path = plot_dir / "per_layer_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"Per-layer decoder readout vs Lax-Hopf\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"N samples: {len(test_idx)}\n")
        f.write(f"skip (model.skip): {bool(model.skip)}\n")
        f.write(f"n_layers: {n_layers}\n\n")
        for ell in range(n_readouts):
            f.write(f"{layer_labels[ell]}\n")
            f.write(f"  MAE: mean={mean_mae[ell]:.6e}  std={std_mae[ell]:.6e}\n")
            f.write(f"  rL2: mean={mean_rl2[ell]:.6e}  std={std_rl2[ell]:.6e}\n")
        f.write(f"\nSanity own u_pred MAE = {own_arr.mean():.6e}\n")
    print(f"\nSaved {metrics_path}")

    # ---- Bar plot: per-layer mean MAE ----
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    xs = np.arange(n_readouts)
    ax.bar(xs, mean_mae, yerr=std_mae, capsize=4,
           color="tab:blue", alpha=0.85, edgecolor="black", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean MAE vs Lax-Hopf")
    ax.set_yscale("log")
    ax.set_title(f"Per-layer decoder readout MAE (N={len(test_idx)} test samples)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(plot_dir / "per_layer_mae.png", dpi=140)
    plt.close(fig)

    # ---- Per-layer error vs time ----
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    cmap = plt.get_cmap("viridis", n_readouts)
    for ell in range(n_readouts):
        curve = np.stack(pt_list[ell]).mean(axis=0)
        ax.plot(t_np, curve, color=cmap(ell), lw=1.2, alpha=0.9,
                label=layer_labels[ell])
    ax.set_xlabel("t")
    ax.set_ylabel("mean MAE vs Lax-Hopf (over test set)")
    ax.set_yscale("log")
    ax.set_title("Per-layer decoder readout: error vs time")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(plot_dir / "per_layer_error_vs_time.png", dpi=140)
    plt.close(fig)

    print(f"Saved per-layer plots to {plot_dir}")


if __name__ == "__main__":
    main()
