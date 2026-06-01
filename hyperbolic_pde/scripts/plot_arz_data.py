"""Quick visualisation of an ARZ dataset (.npz).

Renders a small gallery of samples as (rho, w, v) space-time heatmaps plus
their initial conditions. Defaults match the Riemann pretraining dataset.

Usage:
    python hyperbolic_pde/scripts/plot_arz_data.py \
        --npz /path/to/arz_riemann_trial.npz \
        --out figures/arz_riemann_trial \
        --n 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.arz.datagen_arz import load_arz_dataset


def _plot_one(bundle, idx: int, out_dir: Path) -> Path:
    x = bundle.x
    t = bundle.t
    rho = bundle.rho[idx]   # (nt, nx)
    w   = bundle.w[idx]
    v   = bundle.v[idx]
    rho0 = bundle.rho0[idx]
    v0 = bundle.v0[idx]
    ic = str(bundle.ic_type[idx])

    fig, axes = plt.subplots(
        2, 3, figsize=(13, 6.5),
        gridspec_kw={"height_ratios": [1.0, 3.0]},
    )

    # Row 0: IC traces.
    axes[0, 0].plot(x, rho0, lw=1.2)
    axes[0, 0].set_title(r"$\rho_0(x)$")
    axes[0, 0].set_xlabel("x"); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(x, v0, lw=1.2, color="tab:orange")
    axes[0, 1].set_title(r"$v_0(x)$")
    axes[0, 1].set_xlabel("x"); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.0, 0.85,
        f"sample {idx}\n"
        f"ic_type = {ic}\n"
        f"tau     = {bundle.tau}\n"
        f"nx={x.size}  nt={t.size}\n"
        f"x in [{x[0]:.2f},{x[-1]:.2f}]\n"
        f"t in [{t[0]:.2f},{t[-1]:.2f}]\n"
        f"rho range: [{rho.min():.3f}, {rho.max():.3f}]\n"
        f"v   range: [{v.min():.3f}, {v.max():.3f}]\n",
        family="monospace", fontsize=9, va="top",
    )

    # Row 1: space-time heatmaps (t on y, x on x).
    extent = [x[0], x[-1], t[0], t[-1]]
    for ax, field, title, cmap in [
        (axes[1, 0], rho, r"$\rho(x,t)$",  "viridis"),
        (axes[1, 1], w,   r"$w(x,t)$",     "magma"),
        (axes[1, 2], v,   r"$v(x,t)$",     "cividis"),
    ]:
        im = ax.imshow(
            field, extent=extent, origin="lower", aspect="auto",
            cmap=cmap, interpolation="nearest",
        )
        ax.set_xlabel("x"); ax.set_ylabel("t")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"ARZ sample #{idx}  ({ic})", y=0.99)
    fig.tight_layout()
    out_path = out_dir / f"sample_{idx:04d}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise an ARZ .npz dataset.")
    parser.add_argument("--npz", type=Path, required=True,
                        help="Path to ARZ dataset (.npz, as produced by generate_arz_data.py).")
    parser.add_argument("--out", type=Path, default=Path("figures/arz_data"),
                        help="Output directory for PNGs.")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of samples to render (taken evenly across the dataset).")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated explicit sample indices (overrides --n).")
    args = parser.parse_args()

    bundle = load_arz_dataset(args.npz)
    N = bundle.rho.shape[0]
    print(f"[plot_arz_data] loaded {args.npz}: N={N}, nt={bundle.t.size}, nx={bundle.x.size}, tau={bundle.tau}")

    if args.indices:
        idxs = [int(s) for s in args.indices.split(",") if s.strip()]
    else:
        n = min(args.n, N)
        idxs = list(np.linspace(0, N - 1, n, dtype=int))

    args.out.mkdir(parents=True, exist_ok=True)
    for i in idxs:
        path = _plot_one(bundle, i, args.out)
        print(f"  wrote {path}")
    print(f"[plot_arz_data] done -> {args.out}")


if __name__ == "__main__":
    main()
