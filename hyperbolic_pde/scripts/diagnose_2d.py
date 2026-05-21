"""Diagnose a trained HypNO-ST3-2D checkpoint for prediction collapse.

The training run showed state MAE frozen at ~0.066 from epoch 1. This
script checks the suspected failure mode: the model collapsing to a
constant prediction, or echoing the initial condition u0 at every
timestep (the "skip + zero decoder" trap).

For a few test samples it reports, per sample:
  - MAE(pred, truth)            -- the trained objective
  - MAE(pred, u0 broadcast)     -- if ~0, the model just echoes the IC
  - MAE(pred, mean)             -- if ~0, the model predicts a constant
  - per-t spatial std of pred vs truth  -- does the prediction evolve?
and saves pred / truth / |error| heatmap rows to _sanity_out/.

Usage from repo root:
    python hyperbolic_pde/scripts/diagnose_2d.py --ckpt <path> --config <yaml>
If --ckpt is omitted, the latest checkpoint under the config's run_base
is used.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from hyperbolic_pde.data.fvm_2d import load_dataset_2d
from hyperbolic_pde.models.hypno_st3_2d import HypNO_ST3_2D

OUT_DIR = ROOT / "scripts" / "_sanity_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _find_latest_ckpt(run_base: Path) -> Path | None:
    runs = sorted(run_base.glob("run_*"))
    for run in reversed(runs):
        # Sort checkpoints by numeric epoch (filenames are unpadded, so lex
        # order would pick epoch 90 over epoch 150).
        candidates: list[tuple[int, Path]] = []
        for p in run.glob("checkpoint_epoch*.pt"):
            try:
                ep = int(p.stem.replace("checkpoint_epoch", ""))
            except ValueError:
                continue
            candidates.append((ep, p))
        final = run / "model_final.pt"
        if final.exists():
            return final
        if candidates:
            return max(candidates, key=lambda kv: kv[0])[1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose HypNO-ST3-2D collapse.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "hyperbolic_pde_2d.yaml"))
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--n", type=int, default=3, help="Number of test samples to inspect.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    model_cfg = cfg["hypno_st3_2d"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- locate checkpoint ----
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        run_base = Path(model_cfg.get("run_base", "hyperbolic_pde/runs/hypno_st3_2d"))
        if not run_base.is_absolute():
            run_base = ROOT.parent / run_base
        ckpt_path = _find_latest_ckpt(run_base)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found under {run_base}")
    print(f"checkpoint: {ckpt_path}")

    # ---- data ----
    data_path = Path(data_cfg["path"])
    if not data_path.is_absolute():
        data_path = ROOT.parent / data_path
    bundle = load_dataset_2d(data_path)
    x = torch.tensor(bundle.x, dtype=torch.float32, device=device)
    y = torch.tensor(bundle.y, dtype=torch.float32, device=device)
    t = torch.tensor(bundle.t, dtype=torch.float32, device=device)

    # ---- model ----
    _dhn = model_cfg.get("d_hidden_nonadj", None)
    model = HypNO_ST3_2D(
        stencil_k_x=int(model_cfg.get("stencil_k_x", 3)),
        stencil_k_y=int(model_cfg.get("stencil_k_y", 3)),
        stencil_k_t=int(model_cfg.get("stencil_k_t", 3)),
        d_latent=int(model_cfg.get("d_latent", 64)),
        d_hidden=int(model_cfg.get("d_hidden", 64)),
        d_hidden_nonadj=int(_dhn) if _dhn is not None else None,
        n_layers=int(model_cfg.get("n_layers", 6)),
        activation=str(model_cfg.get("activation", "gelu")),
        causal_temporal=bool(model_cfg.get("causal_temporal", True)),
        readout=str(model_cfg.get("readout", "gelu")),
        encoder_type=str(model_cfg.get("encoder_type", "gnn")),
        skip=bool(model_cfg.get("skip", True)),
        use_checkpoint=False,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # ---- inspect last n samples (test split is the tail of the dataset) ----
    N = bundle.u.shape[0]
    idxs = list(range(N - args.n, N))
    nt = bundle.t.shape[0]

    print(f"\n{'sample':>6} | {'MAE(pred,truth)':>16} | {'MAE(pred,u0)':>13} | "
          f"{'MAE(pred,mean)':>14} | verdict")
    print("-" * 78)

    for s in idxs:
        u0 = torch.tensor(bundle.u0[s : s + 1], dtype=torch.float32, device=device)
        truth = torch.tensor(bundle.u[s], dtype=torch.float32, device=device)   # [nt,ny,nx]
        with torch.inference_mode():
            pred, _ = model(u0, x, y, t)
        pred = pred[0]                                                          # [nt,ny,nx]

        u0_bc = u0[0].unsqueeze(0).expand(nt, -1, -1)
        mae_truth = (pred - truth).abs().mean().item()
        mae_u0 = (pred - u0_bc).abs().mean().item()
        mae_mean = (pred - pred.mean()).abs().mean().item()

        if mae_u0 < 0.01:
            verdict = "IC-ECHO (pred ~= u0)"
        elif mae_mean < 0.01:
            verdict = "CONSTANT (pred ~= scalar)"
        else:
            verdict = "non-trivial"
        print(f"{s:>6} | {mae_truth:>16.4e} | {mae_u0:>13.4e} | "
              f"{mae_mean:>14.4e} | {verdict}")

        # per-t spatial std: does the field evolve in time?
        pred_std_t = pred.reshape(nt, -1).std(dim=1).cpu().numpy()
        truth_std_t = truth.reshape(nt, -1).std(dim=1).cpu().numpy()

        # heatmap row: pred / truth / |error| at 5 timesteps
        frames = np.linspace(0, nt - 1, 5).round().astype(int)
        fig, axes = plt.subplots(3, len(frames), figsize=(3 * len(frames), 8))
        for col, k in enumerate(frames):
            for row, (field, name, vmax) in enumerate([
                (pred[k], "pred", 1.0),
                (truth[k], "truth", 1.0),
                ((pred[k] - truth[k]).abs(), "|err|", 0.3),
            ]):
                ax = axes[row, col]
                im = ax.imshow(field.cpu().numpy(), origin="lower",
                               vmin=0.0, vmax=vmax, cmap="viridis")
                if col == 0:
                    ax.set_ylabel(name)
                if row == 0:
                    ax.set_title(f"t={bundle.t[k]:.2f}")
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(
            f"sample {s} | MAE(pred,truth)={mae_truth:.3e} | "
            f"MAE(pred,u0)={mae_u0:.3e}"
        )
        out = OUT_DIR / f"diagnose_sample_{s}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)

        print(f"         per-t std  pred : {np.array2string(pred_std_t, precision=3, max_line_width=200)}")
        print(f"         per-t std  truth: {np.array2string(truth_std_t, precision=3, max_line_width=200)}")
        print(f"         saved: {out}\n")

    print("Interpretation:")
    print("  - MAE(pred,u0) ~ 0      -> model echoes the IC at every t (skip-collapse).")
    print("  - MAE(pred,mean) ~ 0    -> model predicts a single constant.")
    print("  - pred per-t std flat while truth std varies -> prediction does not evolve.")


if __name__ == "__main__":
    main()
