"""Plot a few random samples from an ARZ dataset for visual inspection.

For each sampled trajectory it draws, per field (rho, v, w):
  * a space-time heatmap  (rows = time, cols = space)
  * the initial condition u(x, t=0) as a line

Usage (CLEPS, headless -> saves PNGs):
    python hyperbolic_pde/arz/plot_dataset_samples.py \
        --data /home/dzdrale/scratch/arz_1d/arz_riemann_mark2_prho.npz \
        --n 3 --out-dir /home/dzdrale/scratch/runs/plots/dataset_inspect

If --indices is given it overrides random sampling, e.g. --indices 0 7 42.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display on the cluster
import matplotlib.pyplot as plt
import numpy as np

from hyperbolic_pde.arz.datagen_arz import load_arz_dataset


def plot_sample(b, idx: int, out_path: Path) -> None:
    """One figure per sample: 3 fields x (heatmap, IC line)."""
    x, t = b.x, b.t
    fields = [
        ("rho", b.rho[idx], b.rho0[idx], "viridis"),
        ("v", b.v[idx], b.v0[idx], "plasma"),
        ("w", b.w[idx], b.w0[idx], "magma"),
    ]
    extent = [x.min(), x.max(), t.max(), t.min()]  # imshow origin upper

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, (name, field, ic, cmap) in enumerate(fields):
        ax_hm = axes[0, col]
        im = ax_hm.imshow(field, aspect="auto", extent=extent, cmap=cmap)
        ax_hm.set_title(f"{name}(x, t)")
        ax_hm.set_xlabel("x")
        ax_hm.set_ylabel("t")
        fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

        ax_ic = axes[1, col]
        ax_ic.plot(x, ic, lw=1.6, label="t=0")
        ax_ic.plot(x, field[-1], lw=1.2, ls="--", label=f"t={t[-1]:.3g}")
        ax_ic.set_title(f"{name}: initial vs final")
        ax_ic.set_xlabel("x")
        ax_ic.set_ylabel(name)
        ax_ic.legend(fontsize=8)
        ax_ic.grid(alpha=0.3)

    ic_type = b.ic_type[idx] if idx < len(b.ic_type) else "?"
    nseg = int(b.num_segments[idx]) if idx < len(b.num_segments) else -1
    fig.suptitle(
        f"sample #{idx}  |  ic_type={ic_type}  num_segments={nseg}  "
        f"tau={b.tau}  p_form={b.p_form}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot random ARZ dataset samples.")
    ap.add_argument("--data", type=Path, required=True, help="path to .npz dataset")
    ap.add_argument("--n", type=int, default=3, help="number of random samples")
    ap.add_argument("--indices", type=int, nargs="*", default=None,
                    help="explicit sample indices (overrides --n random)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("dataset_inspect"))
    args = ap.parse_args()

    b = load_arz_dataset(args.data)
    N = b.rho.shape[0]
    print(f"[plot] loaded {args.data}")
    print(f"[plot] N={N}  nt={b.t.shape[0]}  nx={b.x.shape[0]}  "
          f"tau={b.tau}  p_form={b.p_form}")
    print(f"[plot] rho range [{b.rho.min():.4g}, {b.rho.max():.4g}]  "
          f"v range [{b.v.min():.4g}, {b.v.max():.4g}]  "
          f"w range [{b.w.min():.4g}, {b.w.max():.4g}]")

    if args.indices:
        idxs = [i for i in args.indices if 0 <= i < N]
    else:
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(N, size=min(args.n, N), replace=False).tolist()
    print(f"[plot] sampling indices: {idxs}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in idxs:
        plot_sample(b, i, args.out_dir / f"arz_sample_{i:04d}.png")


if __name__ == "__main__":
    main()
