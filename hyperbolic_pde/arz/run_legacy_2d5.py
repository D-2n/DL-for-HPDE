"""Load a pre-trim (2d+5) ARZ checkpoint with the FROZEN legacy model and dump
its outputs on a dataset -- no config.yaml needed.

Targets the old bare-state_dict checkpoints whose architecture predates commit
cde6c0a (adj edge 2d+5, e.g. runs/hypno_arz_riemann.pt). Builds the model class
HypNO_ARZ_Legacy2d5 with kwargs inferred from the checkpoint's own tensor shapes
(node-MLP in-dim -> use_relaxation_features; nonadj out-dim -> d_hidden_nonadj;
mp_layers.* count -> n_layers; decoder/lifting dims -> d_latent/d_hidden). kx/kt
are NOT recoverable from weights, so they default to the 4ec18fb-era 2/2 (the
arz_st3 legacy args confirm kx=kt=2) and are overridable.

Saves predicted (rho, v, w) fields per sample to an .npz and prints summary
stats + per-channel MAE vs ground truth. Does NOT run any baseline solver.

    python -m hyperbolic_pde.arz.run_legacy_2d5 \
        --ckpt /home/dzdrale/scratch/runs/hypno_arz_riemann.pt \
        --data /home/dzdrale/scratch/arz_1d/arz_riemann_exact_prho.npz \
        --pressure-form rho --samples 8 \
        --out /home/dzdrale/scratch/results/legacy2d5_outputs.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.model_arz_legacy_2d5 import HypNO_ARZ_Legacy2d5


def _strip(sd):
    if isinstance(sd, dict) and any(k.startswith("_orig_mod.") for k in sd):
        return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    return sd


def infer_kwargs(sd: dict, kx: int, kt: int, skip: bool) -> dict:
    """Recover constructor kwargs from the checkpoint tensor shapes.

    `skip` is NOT encoded in the weights (it is a forward-pass flag), so it must
    be supplied to match how the model was trained -- a wrong value adds/removes
    the u0 residual and roughly doubles (or halves) the output.
    """
    node_in = sd["lifting.node_mlp.0.weight"].shape[1]          # 7 or 9
    d_latent = sd["decoder.0.weight"].shape[1]                  # decoder in = d_latent
    d_hidden = sd["lifting.node_mlp.0.weight"].shape[0]         # node MLP hidden width
    nonadj_out = sd["mp_layers.0.nonadj_msg.0.weight"].shape[0]
    n_layers = 1 + max(
        int(k.split(".")[1]) for k in sd if k.startswith("mp_layers.")
    )
    # _make_mlp(layers=L) emits L Linear layers (indices 0,2,..,2(L-1)); the
    # number of decoder .weight keys IS the depth. (Earlier off-by-one used
    # len-1, which built a depth-3 decoder whose output landed one layer early.)
    dec_keys = [k for k in sd if k.startswith("decoder.") and k.endswith(".weight")]
    decoder_depth = len(dec_keys)
    return dict(
        stencil_k_x=kx, stencil_k_t=kt,
        d_latent=int(d_latent), d_hidden=int(d_hidden),
        n_layers=int(n_layers),
        d_hidden_nonadj=int(nonadj_out) if nonadj_out != d_hidden else None,
        decoder_depth=int(decoder_depth),
        skip=skip,
        use_checkpoint=False,
        normalize_edge_offsets=True,
        use_relaxation_features=(node_in == 9),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pressure-form", default="auto",
                    choices=["auto", "rho", "rho+rho2"],
                    help="auto (default) reads it from the dataset bundle's p_form "
                         "so model-forward, GT, and plots are all consistent.")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--kx", type=int, default=2, help="stencil_k_x (not in weights; 4ec18fb default 2)")
    ap.add_argument("--kt", type=int, default=2, help="stencil_k_t (not in weights; 4ec18fb default 2)")
    ap.add_argument("--skip", dest="skip", action="store_true", default=True,
                    help="decoder skip residual u_pred = u0 + decoder(h) (default; "
                         "must match training -- check the run's config.yaml).")
    ap.add_argument("--no-skip", dest="skip", action="store_false",
                    help="u_pred = decoder(h) directly (skip=false).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None, help="optional .npz to save predicted fields")
    ap.add_argument("--figures", type=Path, default=None,
                    help="dir for per-sample space-time PNGs (pred|GT|err for rho,v,w)")
    ap.add_argument("--n-plots", type=int, default=5)
    args = ap.parse_args()

    # Load the dataset FIRST so we can pin the pressure form to its p_form
    # (model-forward + GT + plots then all use the same closure).
    bundle = load_arz_dataset(args.data)
    if args.pressure_form == "auto":
        pform = bundle.p_form
        print(f"[run_legacy_2d5] pressure_form=auto -> dataset p_form={pform!r}")
    else:
        pform = args.pressure_form
        if pform != bundle.p_form:
            print(f"[run_legacy_2d5] WARNING: --pressure-form {pform!r} != dataset "
                  f"p_form {bundle.p_form!r}; GT/forward will be INCONSISTENT.")
    P.set_pressure_form(pform)
    print(f"[run_legacy_2d5] pressure_form={P.get_pressure_form()}")

    raw = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    sd = _strip(raw["model"] if isinstance(raw, dict) and "model" in raw else raw)

    kwargs = infer_kwargs(sd, args.kx, args.kt, args.skip)
    print(f"[run_legacy_2d5] inferred kwargs: {kwargs}")

    model = HypNO_ARZ_Legacy2d5(**kwargs).to(args.device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[run_legacy_2d5] MISSING keys ({len(missing)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"[run_legacy_2d5] UNEXPECTED keys ({len(unexpected)}): {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")
    if not missing and not unexpected:
        print("[run_legacy_2d5] state_dict loaded CLEANLY (all keys matched).")
    model.eval()

    x = torch.tensor(bundle.x, dtype=torch.float32, device=args.device)
    t = torch.tensor(bundle.t, dtype=torch.float32, device=args.device)
    take = min(args.samples, bundle.rho.shape[0])
    print(f"[run_legacy_2d5] data: N={bundle.rho.shape[0]} nx={bundle.x.shape[0]} "
          f"nt={bundle.t.shape[0]} p_form={bundle.p_form}  evaluating {take} samples")

    rho_out, v_out, w_out = [], [], []
    rho_g, v_g, w_g = [], [], []
    mae_rho = mae_w = mae_v = 0.0
    for i in range(take):
        rho0 = torch.tensor(bundle.rho0[i][None], dtype=torch.float32, device=args.device)
        w0 = torch.tensor(bundle.w0[i][None], dtype=torch.float32, device=args.device)
        with torch.no_grad():
            rho_p, w_p, _ = model(rho0, w0, x, t)
        rho_p = rho_p[0].cpu().numpy()
        w_p = w_p[0].cpu().numpy()
        v_p = w_p - P.pressure(rho_p)
        rho_out.append(rho_p); v_out.append(v_p); w_out.append(w_p)

        rho_gt = bundle.rho[i]; w_gt = bundle.w[i]
        v_gt = w_gt - P.pressure(rho_gt)
        rho_g.append(rho_gt); v_g.append(v_gt); w_g.append(w_gt)
        mae_rho += np.mean(np.abs(rho_p - rho_gt))
        mae_w += np.mean(np.abs(w_p - w_gt))
        mae_v += np.mean(np.abs(v_p - v_gt))

    n = float(take)
    print(f"[run_legacy_2d5] MAE over {take} samples:  "
          f"rho={mae_rho/n:.4e}  w={mae_w/n:.4e}  v={mae_v/n:.4e}")
    r = np.stack(rho_out)
    print(f"[run_legacy_2d5] pred rho range [{r.min():.3f}, {r.max():.3f}]  "
          f"mean {r.mean():.3f}")

    # ----- visuals: per-sample space-time panels (pred | GT | error) ------- #
    if args.figures is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = Path(args.figures)
        fig_dir.mkdir(parents=True, exist_ok=True)
        x_np = bundle.x.astype(np.float64)
        t_np = bundle.t.astype(np.float64)
        channels = [("rho", rho_out, rho_g), ("v", v_out, v_g), ("w", w_out, w_g)]
        n_fig = min(args.n_plots, take)
        for i in range(n_fig):
            fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
            for row, (name, preds, gts) in enumerate(channels):
                pred = preds[i]; gt = gts[i]; err = pred - gt
                vmin = min(pred.min(), gt.min()); vmax = max(pred.max(), gt.max())
                im0 = axes[row, 0].pcolormesh(x_np, t_np, pred, shading="auto",
                                              vmin=vmin, vmax=vmax, cmap="viridis")
                axes[row, 0].set_title(f"{name} pred")
                fig.colorbar(im0, ax=axes[row, 0])
                im1 = axes[row, 1].pcolormesh(x_np, t_np, gt, shading="auto",
                                              vmin=vmin, vmax=vmax, cmap="viridis")
                axes[row, 1].set_title(f"{name} GT")
                fig.colorbar(im1, ax=axes[row, 1])
                amax = float(np.abs(err).max()) or 1.0
                im2 = axes[row, 2].pcolormesh(x_np, t_np, err, shading="auto",
                                              vmin=-amax, vmax=amax, cmap="RdBu_r")
                axes[row, 2].set_title(f"{name} err (MAE {np.mean(np.abs(err)):.3e})")
                fig.colorbar(im2, ax=axes[row, 2])
                for c in range(3):
                    axes[row, c].set_xlabel("x"); axes[row, c].set_ylabel("t")
            fig.suptitle(f"legacy2d5  sample {i}  (p_form={bundle.p_form})")
            fig.savefig(fig_dir / f"legacy2d5_sample_{i}.png", dpi=130)
            plt.close(fig)
        print(f"[run_legacy_2d5] wrote {n_fig} figures -> {fig_dir}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            rho=np.stack(rho_out), v=np.stack(v_out), w=np.stack(w_out),
            rho_gt=bundle.rho[:take], w_gt=bundle.w[:take],
            x=bundle.x, t=bundle.t,
        )
        print(f"[run_legacy_2d5] saved predicted fields -> {args.out}")


if __name__ == "__main__":
    main()
