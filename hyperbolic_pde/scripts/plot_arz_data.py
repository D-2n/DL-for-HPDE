"""Quick visualisation of an ARZ dataset (.npz).

Renders a small gallery of samples as (rho, w, v) space-time heatmaps plus
their initial conditions. Defaults match the Riemann pretraining dataset.

For homogeneous Riemann data, rarefaction fans are hard to spot on a raw
heatmap because:
  - w is *constant* across the 1-rarefaction (Riemann invariant w_L preserved),
    so w(x,t) only ever shows the 2-contact.
  - rho varies smoothly in the fan, which looks like a soft gradient next to
    the sharp shock/contact lines.
This script therefore overlays contour lines on rho(x,t) to make fan
structure visible, and labels each sample with its detected 1-wave type
(shock / rare / degen).

Usage:
    python hyperbolic_pde/scripts/plot_arz_data.py \
        --npz /path/to/arz_riemann_trial.npz \
        --out figures/arz_riemann_trial \
        --n 8
    python hyperbolic_pde/scripts/plot_arz_data.py \
        --npz ... --filter rare --n 12       # only rarefaction samples
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
from hyperbolic_pde.arz import physics_arz as P
from hyperbolic_pde.arz.riemann_arz import _rho_star_from_w_minus_v


def _classify(rho_L, w_L, rho_R, w_R):
    """Return ('shock'|'rare'|'degen', wave_speeds_dict)."""
    v_L = w_L - P.pressure(rho_L)
    v_R = w_R - P.pressure(rho_R)
    rho_star = float(_rho_star_from_w_minus_v(w_L - v_R))
    w_star = w_L
    v_star = v_R
    lam1_L    = v_L    - rho_L    * P.dpressure(rho_L)
    lam1_star = v_star - rho_star * P.dpressure(rho_star)

    info = {
        "rho_L": rho_L, "w_L": w_L, "v_L": v_L,
        "rho_R": rho_R, "w_R": w_R, "v_R": v_R,
        "rho_star": rho_star, "v_star": v_star,
        "lam1_L": lam1_L, "lam1_star": lam1_star,
        "contact_speed": v_star,
    }
    if lam1_L > lam1_star + 1e-12:
        denom = rho_star - rho_L
        if abs(denom) < 1e-14:
            s = lam1_L
        else:
            s = (rho_star * v_star - rho_L * v_L) / denom
        info["shock_speed"] = s
        return "shock", info
    if lam1_L < lam1_star - 1e-12:
        info["fan_head"] = lam1_L     # smaller xi
        info["fan_tail"] = lam1_star  # larger xi
        return "rare", info
    return "degen", info


def _interface_states(bundle, idx):
    rho0 = bundle.rho0[idx]; w0 = bundle.w0[idx]
    d = np.abs(np.diff(rho0)) + np.abs(np.diff(w0))
    j = int(np.argmax(d))
    x_grid = bundle.x
    x0 = 0.5 * (x_grid[j] + x_grid[j + 1])
    rho_L, w_L = float(rho0[j]),     float(w0[j])
    rho_R, w_R = float(rho0[j + 1]), float(w0[j + 1])
    return x0, rho_L, w_L, rho_R, w_R


def _overlay_wave_lines(ax, info, cls, x0, t):
    """Draw analytical wave trajectories x = x0 + speed * t over the heatmap."""
    t_arr = np.array([t[0], t[-1]])
    if cls == "shock":
        s = info["shock_speed"]
        ax.plot(x0 + s * t_arr, t_arr, color="red", lw=1.2, ls="--",
                label=f"1-shock s={s:+.2f}")
    elif cls == "rare":
        ax.plot(x0 + info["fan_head"] * t_arr, t_arr, color="red", lw=1.0, ls=":",
                label=f"fan head λ={info['fan_head']:+.2f}")
        ax.plot(x0 + info["fan_tail"] * t_arr, t_arr, color="red", lw=1.0, ls=":",
                label=f"fan tail λ={info['fan_tail']:+.2f}")
    c = info["contact_speed"]
    ax.plot(x0 + c * t_arr, t_arr, color="white", lw=1.0, ls="-.",
            label=f"contact v*={c:+.2f}")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7)


def _plot_one(bundle, idx: int, out_dir: Path) -> Path:
    x = bundle.x
    t = bundle.t
    rho = bundle.rho[idx]; w = bundle.w[idx]; v = bundle.v[idx]
    rho0 = bundle.rho0[idx]; v0 = bundle.v0[idx]
    ic = str(bundle.ic_type[idx])

    x0, rho_L, w_L, rho_R, w_R = _interface_states(bundle, idx)
    cls, info = _classify(rho_L, w_L, rho_R, w_R)

    fig, axes = plt.subplots(
        2, 3, figsize=(13.5, 7.0),
        gridspec_kw={"height_ratios": [1.0, 3.0]},
    )

    # Row 0: IC traces + metadata.
    axes[0, 0].plot(x, rho0, lw=1.2)
    axes[0, 0].axvline(x0, color="k", lw=0.6, ls=":")
    axes[0, 0].set_title(r"$\rho_0(x)$"); axes[0, 0].set_xlabel("x")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(x, v0, lw=1.2, color="tab:orange")
    axes[0, 1].axvline(x0, color="k", lw=0.6, ls=":")
    axes[0, 1].set_title(r"$v_0(x)$"); axes[0, 1].set_xlabel("x")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].axis("off")
    cls_color = {"shock": "red", "rare": "tab:green", "degen": "gray"}[cls]
    axes[0, 2].text(
        0.0, 0.95,
        f"sample {idx}   ic={ic}   tau={bundle.tau}\n"
        f"1-wave: {cls.upper()}\n"
        f"x0 = {x0:+.3f}\n"
        f"L: rho={rho_L:.3f}  w={w_L:.3f}  v={info['v_L']:+.3f}\n"
        f"*: rho={info['rho_star']:.3f}  v*={info['v_star']:+.3f}\n"
        f"R: rho={rho_R:.3f}  w={w_R:.3f}  v={info['v_R']:+.3f}\n"
        f"lam1_L={info['lam1_L']:+.3f}  lam1*={info['lam1_star']:+.3f}\n"
        + (f"shock_speed = {info.get('shock_speed', float('nan')):+.3f}\n"
           if cls == "shock" else
           f"fan head/tail = {info.get('fan_head', float('nan')):+.3f} / "
           f"{info.get('fan_tail', float('nan')):+.3f}\n"
           if cls == "rare" else "")
        + f"contact_speed = {info['contact_speed']:+.3f}",
        family="monospace", fontsize=9, va="top",
        color=cls_color,
    )

    # Row 1: heatmaps with overlays.
    extent = [x[0], x[-1], t[0], t[-1]]
    panels = [
        (axes[1, 0], rho, r"$\rho(x,t)$", "viridis", True),   # contour overlay
        (axes[1, 1], w,   r"$w(x,t)$",    "magma",   False),
        (axes[1, 2], v,   r"$v(x,t)$",    "cividis", True),
    ]
    for ax, field, title, cmap, do_contour in panels:
        im = ax.imshow(
            field, extent=extent, origin="lower", aspect="auto",
            cmap=cmap, interpolation="nearest",
        )
        if do_contour:
            n_levels = 10
            levels = np.linspace(field.min(), field.max(), n_levels)
            ax.contour(
                np.linspace(x[0], x[-1], field.shape[1]),
                np.linspace(t[0], t[-1], field.shape[0]),
                field,
                levels=levels, colors="white", linewidths=0.4, alpha=0.6,
            )
        _overlay_wave_lines(ax, info, cls, x0, t)
        ax.set_xlim(x[0], x[-1]); ax.set_ylim(t[0], t[-1])
        ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"ARZ sample #{idx}  [{cls.upper()}]  ({ic})", y=0.99, color=cls_color)
    fig.tight_layout()
    tag = cls
    out_path = out_dir / f"sample_{idx:04d}_{tag}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _filter_indices(bundle, mode: str, n: int) -> list[int]:
    N = bundle.rho.shape[0]
    if mode == "all":
        return list(np.linspace(0, N - 1, min(n, N), dtype=int))
    keep = []
    for i in range(N):
        x0, rL, wL, rR, wR = _interface_states(bundle, i)
        cls, _ = _classify(rL, wL, rR, wR)
        if cls == mode:
            keep.append(i)
    if not keep:
        raise SystemExit(f"No samples matched filter={mode!r}.")
    if n >= len(keep):
        return keep
    sel = np.linspace(0, len(keep) - 1, n, dtype=int)
    return [keep[i] for i in sel]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise an ARZ .npz dataset.")
    parser.add_argument("--npz", type=Path, required=True,
                        help="Path to ARZ dataset (.npz, as produced by generate_arz_data.py).")
    parser.add_argument("--out", type=Path, default=Path("figures/arz_data"),
                        help="Output directory for PNGs.")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of samples to render (taken evenly from the matching set).")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated explicit sample indices (overrides --n and --filter).")
    parser.add_argument("--filter", type=str, default="all",
                        choices=("all", "shock", "rare", "degen"),
                        help="Only render samples whose 1-wave matches this class.")
    args = parser.parse_args()

    bundle = load_arz_dataset(args.npz)
    N = bundle.rho.shape[0]
    print(f"[plot_arz_data] loaded {args.npz}: N={N}, nt={bundle.t.size}, nx={bundle.x.size}, tau={bundle.tau}")

    if args.indices:
        idxs = [int(s) for s in args.indices.split(",") if s.strip()]
    else:
        idxs = _filter_indices(bundle, args.filter, args.n)
    print(f"[plot_arz_data] filter={args.filter}  rendering {len(idxs)} sample(s)")

    args.out.mkdir(parents=True, exist_ok=True)
    for i in idxs:
        path = _plot_one(bundle, i, args.out)
        print(f"  wrote {path}")
    print(f"[plot_arz_data] done -> {args.out}")


if __name__ == "__main__":
    main()
