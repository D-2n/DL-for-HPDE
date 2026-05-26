"""Train HypNO-ARZ (plan §7).

Self-contained mirror of the LWR trainer. Config via CLI flags; checkpoints
to a directory; per-channel (rho, w, v) logging and a near-eq vs off-eq split
based on the dataset's tau and equilibrium deviation at t=0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.losses_arz import total_loss
from hyperbolic_pde.arz.model_arz import HypNO_ARZ


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class ArzDataset(Dataset):
    def __init__(self, rho0, w0, rho, w):
        self.rho0 = torch.tensor(rho0, dtype=torch.float32)
        self.w0   = torch.tensor(w0,   dtype=torch.float32)
        self.rho  = torch.tensor(rho,  dtype=torch.float32)
        self.w    = torch.tensor(w,    dtype=torch.float32)

    def __len__(self): return self.rho0.shape[0]
    def __getitem__(self, i):
        return self.rho0[i], self.w0[i], self.rho[i], self.w[i]


def _split_indices(N: int, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = max(1, int(round(val_frac * N)))
    return perm[n_val:], perm[:n_val]


def _diseq_score(rho0: np.ndarray, w0: np.ndarray) -> np.ndarray:
    """Average |v0 - V(rho0)| per sample. >0.1 -> off-equilibrium."""
    v0 = w0 - (rho0 + rho0 * rho0)
    Veq = 1.0 - rho0
    return np.abs(v0 - Veq).mean(axis=1)


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=None,
                    help="Override dataset tau (defaults to bundle.tau).")
    # model
    ap.add_argument("--kx", type=int, default=2)
    ap.add_argument("--kt", type=int, default=2)
    ap.add_argument("--d-latent", type=int, default=64)
    ap.add_argument("--d-hidden", type=int, default=96)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--decoder-depth", type=int, default=3)
    ap.add_argument("--skip", action="store_true", default=True)
    ap.add_argument("--no-skip", dest="skip", action="store_false")
    ap.add_argument("--use-checkpoint", action="store_true")
    ap.add_argument("--normalize-edge-offsets",    dest="normalize_edge_offsets",
                    action="store_true", default=True)
    ap.add_argument("--no-normalize-edge-offsets", dest="normalize_edge_offsets",
                    action="store_false")
    # loss weights
    ap.add_argument("--lambda-state",   type=float, default=1.0)
    ap.add_argument("--lambda-cons",    type=float, default=0.1)
    ap.add_argument("--lambda-balance", type=float, default=0.1)
    ap.add_argument("--lambda-probe",   type=float, default=0.1)
    ap.add_argument("--w-weight",       type=float, default=1.0)
    # optim
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--diseq-threshold", type=float, default=0.1,
                    help="|v0 - V(rho0)| mean threshold for 'off-equilibrium' split.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bundle = load_arz_dataset(args.data)
    tau = float(args.tau if args.tau is not None else bundle.tau)
    print(f"[train_arz] loaded {args.data} -- N={bundle.rho.shape[0]} "
          f"nt={bundle.t.shape[0]} nx={bundle.x.shape[0]} tau={tau}")

    train_idx, val_idx = _split_indices(bundle.rho.shape[0], args.val_frac, args.seed)
    print(f"[train_arz] train/val: {len(train_idx)}/{len(val_idx)}")

    train_ds = ArzDataset(
        bundle.rho0[train_idx], bundle.w0[train_idx],
        bundle.rho[train_idx],  bundle.w[train_idx],
    )
    val_ds = ArzDataset(
        bundle.rho0[val_idx], bundle.w0[val_idx],
        bundle.rho[val_idx],  bundle.w[val_idx],
    )
    diseq_val = _diseq_score(bundle.rho0[val_idx], bundle.w0[val_idx])
    eq_mask  = diseq_val <  args.diseq_threshold
    off_mask = diseq_val >= args.diseq_threshold
    print(f"[train_arz] val split: near-eq={int(eq_mask.sum())}, off-eq={int(off_mask.sum())}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    x = torch.tensor(bundle.x, dtype=torch.float32).to(args.device)
    t = torch.tensor(bundle.t, dtype=torch.float32).to(args.device)

    model = HypNO_ARZ(
        stencil_k_x=args.kx, stencil_k_t=args.kt,
        d_latent=args.d_latent, d_hidden=args.d_hidden,
        n_layers=args.depth, decoder_depth=args.decoder_depth,
        skip=args.skip, use_checkpoint=args.use_checkpoint,
        normalize_edge_offsets=args.normalize_edge_offsets,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_arz] model params = {n_params/1e6:.3f} M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and args.device == "cuda"))

    args.ckpt.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_val = float("inf")
    if args.resume and (args.ckpt / "last.pt").exists():
        ck = torch.load(args.ckpt / "last.pt", map_location=args.device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", best_val)
        print(f"[train_arz] resumed from epoch {start_epoch}, best_val={best_val:.4e}")

    log = []
    for ep in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "L_state": 0.0, "L_cons": 0.0, "L_bal": 0.0, "L_probe": 0.0}
        n_batches = 0
        for rho0, w0, rho_gt, w_gt in train_loader:
            rho0, w0 = rho0.to(args.device), w0.to(args.device)
            rho_gt, w_gt = rho_gt.to(args.device), w_gt.to(args.device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(args.amp and args.device == "cuda")):
                rho_p, w_p, u_hats = model(rho0, w0, x, t)
                terms = total_loss(
                    rho_p, w_p, u_hats, rho_gt, w_gt, t=t, tau=tau,
                    lam_state=args.lambda_state, lam_cons=args.lambda_cons,
                    lam_bal=args.lambda_balance, lam_probe=args.lambda_probe,
                    w_weight=args.w_weight,
                )
            scaler.scale(terms["loss"]).backward()
            scaler.step(opt)
            scaler.update()
            for k in running:
                running[k] += float(terms[k] if k == "loss" else terms[k])
            n_batches += 1
        for k in running:
            running[k] /= max(1, n_batches)

        # validation
        model.eval()
        val_metrics = {"mae_rho": 0.0, "mae_w": 0.0, "mae_v": 0.0,
                       "mae_eq": 0.0, "mae_off": 0.0,
                       "n_eq": 0, "n_off": 0}
        with torch.no_grad():
            idx_start = 0
            mae_per_sample = []
            for rho0, w0, rho_gt, w_gt in val_loader:
                rho0, w0 = rho0.to(args.device), w0.to(args.device)
                rho_gt, w_gt = rho_gt.to(args.device), w_gt.to(args.device)
                rho_p, w_p, _ = model(rho0, w0, x, t)
                v_p = w_p - (rho_p + rho_p * rho_p)
                v_gt = w_gt - (rho_gt + rho_gt * rho_gt)
                B = rho0.shape[0]
                mae_rho_b = (rho_p - rho_gt).abs().mean(dim=(1, 2))
                mae_w_b   = (w_p - w_gt).abs().mean(dim=(1, 2))
                mae_v_b   = (v_p - v_gt).abs().mean(dim=(1, 2))
                val_metrics["mae_rho"] += float(mae_rho_b.sum())
                val_metrics["mae_w"]   += float(mae_w_b.sum())
                val_metrics["mae_v"]   += float(mae_v_b.sum())
                for b in range(B):
                    is_eq = eq_mask[idx_start + b]
                    cmae = float(mae_rho_b[b] + mae_w_b[b])
                    if is_eq:
                        val_metrics["mae_eq"] += cmae
                        val_metrics["n_eq"]   += 1
                    else:
                        val_metrics["mae_off"] += cmae
                        val_metrics["n_off"]   += 1
                idx_start += B

        n_val = max(1, len(val_ds))
        val_metrics["mae_rho"] /= n_val
        val_metrics["mae_w"]   /= n_val
        val_metrics["mae_v"]   /= n_val
        val_metrics["mae_eq"]  /= max(1, val_metrics["n_eq"])
        val_metrics["mae_off"] /= max(1, val_metrics["n_off"])
        val_combined = val_metrics["mae_rho"] + val_metrics["mae_w"]

        dt = time.time() - t0
        line = (
            f"ep {ep:3d}  loss {running['loss']:.4e}  "
            f"[state {running['L_state']:.3e} cons {running['L_cons']:.3e} "
            f"bal {running['L_bal']:.3e} probe {running['L_probe']:.3e}]  "
            f"val rho/w/v {val_metrics['mae_rho']:.3e}/{val_metrics['mae_w']:.3e}/{val_metrics['mae_v']:.3e}  "
            f"eq/off {val_metrics['mae_eq']:.3e}/{val_metrics['mae_off']:.3e}  "
            f"({dt:.1f}s)"
        )
        print(line, flush=True)
        log.append({"epoch": ep, "train": running, "val": val_metrics, "dt_s": dt})

        # checkpoint
        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "epoch": ep, "best_val": best_val,
              "args": vars(args), "tau": tau}
        torch.save(ck, args.ckpt / "last.pt")
        if val_combined < best_val:
            best_val = val_combined
            ck["best_val"] = best_val
            torch.save(ck, args.ckpt / "best.pt")
            print(f"  -> saved best.pt ({best_val:.4e})")

    with (args.ckpt / "log.json").open("w") as f:
        json.dump(log, f, indent=2)
    print(f"[train_arz] done. best val combined MAE = {best_val:.4e}")


if __name__ == "__main__":
    main()
