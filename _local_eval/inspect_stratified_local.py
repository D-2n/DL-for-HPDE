"""LOCAL inspection of the stratified exact-Riemann ARZ datagen.

Generates a small batch (default 12) with the SAME settings as the cluster
slurm wrapper (slurm/arz_gen_riemann_exact.sh) -- stratified, p=rho, tau=inf,
v in [0,1], rho in [0.1,0.9] -- then plots rho/v/w space-time heatmaps + IC/final
lines per sample, reusing the convention from
hyperbolic_pde/arz/plot_dataset_samples.py.

Why this file exists: the project package __init__ imports torch (models.pinn),
which isn't installed in the local plotting venv. The datagen + plotting path is
pure NumPy/matplotlib, so we load the arz submodules DIRECTLY by file path with
empty stub parent packages, sidestepping the torch import.

Run (from repo root):
    python _local_eval/inspect_stratified_local.py --n 12 --out-dir _local_eval/strat_inspect
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARZ = REPO / "hyperbolic_pde" / "arz"
DATA = REPO / "hyperbolic_pde" / "data"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap_arz_modules():
    """Stub the torch-importing package inits, then load the pure-NumPy arz +
    data submodules by path so `from hyperbolic_pde.arz import X` resolves."""
    for pkg, p in [
        ("hyperbolic_pde", REPO / "hyperbolic_pde"),
        ("hyperbolic_pde.arz", ARZ),
        ("hyperbolic_pde.data", DATA),
    ]:
        m = types.ModuleType(pkg)
        m.__path__ = [str(p)]
        sys.modules[pkg] = m
    # order matters: dependencies first
    _load("hyperbolic_pde.arz.physics_arz", ARZ / "physics_arz.py")
    _load("hyperbolic_pde.arz.riemann_arz", ARZ / "riemann_arz.py")
    _load("hyperbolic_pde.arz.reference_arz", ARZ / "reference_arz.py")
    _load("hyperbolic_pde.data.fvm", DATA / "fvm.py")
    datagen = _load("hyperbolic_pde.arz.datagen_arz", ARZ / "datagen_arz.py")
    return datagen


def plot_sample(b, idx: int, out_path: Path) -> None:
    """One figure per sample: 3 fields (rho, v, w) x (space-time heatmap, IC/final line).

    Mirrors hyperbolic_pde/arz/plot_dataset_samples.py.
    """
    x, t = b.x, b.t
    fields = [
        ("rho", b.rho[idx], b.rho0[idx], "viridis"),
        ("v",   b.v[idx],   b.v0[idx],   "plasma"),
        ("w",   b.w[idx],   b.w0[idx],   "magma"),
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

    fig.suptitle(
        f"stratified sample #{idx}  |  tau={b.tau}  p_form={b.p_form}", fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _bin_edges(rho_min, rho_max, n_bins):
    """Replicate datagen_arz._build_stratified_riemann_plan bin edges."""
    jump_span = rho_max - rho_min
    edges = np.linspace(0.0, jump_span, n_bins + 1)
    edges[0] = min(0.02 * jump_span, 0.5 * edges[1])
    return [(float(edges[b]), float(edges[b + 1])) for b in range(n_bins)]


def _solve_state(datagen, rho_L, v_L, rho_R, v_R, x0, xm, t):
    """Build IC fields + exact-solve one Riemann state. Returns (rho,w,v hist, rho0,w0,v0)."""
    P = datagen.P
    Rie = sys.modules["hyperbolic_pde.arz.riemann_arz"]
    w_L = v_L + float(P.pressure(np.array(rho_L)))
    w_R = v_R + float(P.pressure(np.array(rho_R)))
    rho0 = np.where(xm <= x0, rho_L, rho_R).astype(np.float64)
    v0 = np.where(xm <= x0, v_L, v_R).astype(np.float64)
    w0 = v0 + P.pressure(rho0)
    rho_hist, w_hist, _ = Rie.solve_riemann_arz_xt(rho_L, w_L, rho_R, w_R, xm, t, x0=x0)
    v_hist = w_hist - P.pressure(rho_hist)
    return (rho_hist.astype(np.float32), w_hist.astype(np.float32), v_hist.astype(np.float32),
            rho0.astype(np.float32), w0.astype(np.float32), v0.astype(np.float32))


def plot_per_bin(datagen, field_name, n_bins, nx, nt, seed, out_path):
    """Grid of ONE representative sample per (wave-type x one-wave strength bin).

    Rows = one-wave strength bins (weak -> strong), columns = [1-shock, 1-rarefaction].
    Contact jump fixed to the MID strength bin so the 1-wave is the variable.
    Shows the space-time heatmap of `field_name` (rho|v|w). Lets you SEE the fan
    (rarefaction column widens over t) vs the sharp front (shock column).
    """
    rng = np.random.default_rng(seed)
    dx = 2.0 / nx
    xm = np.array([-1.0 + (i + 0.5) * dx for i in range(nx)], dtype=np.float64)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    bins = _bin_edges(0.1, 0.9, n_bins)
    contact_bin = bins[n_bins // 2]  # fixed mid-strength contact
    cmap = {"rho": "viridis", "v": "plasma", "w": "magma"}[field_name]
    fidx = {"rho": 0, "v": 2, "w": 1}[field_name]  # index into _solve_state tuple
    extent = [xm.min(), xm.max(), t.max(), t.min()]

    fig, axes = plt.subplots(n_bins, 2, figsize=(8, 3.0 * n_bins), squeeze=False)
    col_label = ["1-SHOCK", "1-RAREFACTION"]
    for r, jb in enumerate(bins):
        for c, is_shock in enumerate((True, False)):
            st = datagen._sample_riemann_stratified_cell(
                xm, rng, 0.1, 0.9, 0.0, 1.0,
                one_wave_is_shock=is_shock,
                one_wave_jump_range=jb, contact_jump_range=contact_bin,
            )
            hists = _solve_state(datagen, *st, xm, t)
            ax = axes[r, c]
            im = ax.imshow(hists[fidx], aspect="auto", extent=extent, cmap=cmap)
            ax.set_title(f"{col_label[c]}  |  1-wave |dρ|∈[{jb[0]:.2f},{jb[1]:.2f}]",
                         fontsize=9)
            ax.set_xlabel("x"); ax.set_ylabel("t")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{field_name}(x,t) per (wave-type x 1-wave strength bin)  "
                 f"[contact fixed mid-bin]  tau=inf p=rho", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Local stratified-datagen inspection.")
    ap.add_argument("--n", type=int, default=12, help="number of samples to generate")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--bins", type=int, default=4, help="n_strength_bins")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-bin", action="store_true",
                    help="Plot ONE sample per (wave-type x strength bin) grid per field, "
                         "instead of N random full samples.")
    ap.add_argument("--out-dir", type=Path, default=REPO / "_local_eval" / "strat_inspect")
    args = ap.parse_args()

    if args.per_bin:
        datagen = _bootstrap_arz_modules()
        datagen.P.set_pressure_form("rho")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for field in ("rho", "v", "w"):
            plot_per_bin(datagen, field, args.bins, args.nx, args.nt, args.seed,
                         args.out_dir / f"per_bin_{field}.png")
        print(f"[inspect] per-bin grids -> {args.out_dir}")
        return

    datagen = _bootstrap_arz_modules()
    datagen.P.set_pressure_form("rho")

    b = datagen.generate_arz_riemann_exact_dataset(
        num_samples=args.n, nx=args.nx, nt=args.nt,
        x_min=-1.0, x_max=1.0, t_max=1.0, seed=args.seed,
        rho_min=0.1, rho_max=0.9, v_min=0.0, v_max=1.0,
        stratified=True, n_strength_bins=args.bins,
    )

    print(f"[inspect] generated N={b.rho.shape[0]} nt={b.t.size} nx={b.x.size} "
          f"tau={b.tau} p_form={b.p_form}")
    print(f"[inspect] rho [{b.rho.min():.4g},{b.rho.max():.4g}]  "
          f"v [{b.v.min():.4g},{b.v.max():.4g}]  w [{b.w.min():.4g},{b.w.max():.4g}]")
    # Sanity: v fully in [0,1]?
    voob = int(((b.v < -1e-6) | (b.v > 1 + 1e-6)).sum())
    print(f"[inspect] v out-of-[0,1] cells: {voob} / {b.v.size}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(b.rho.shape[0]):
        plot_sample(b, i, args.out_dir / f"strat_sample_{i:04d}.png")
    print(f"[inspect] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
