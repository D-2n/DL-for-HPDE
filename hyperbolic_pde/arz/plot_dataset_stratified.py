"""Plot a few samples from each (ic_type x num_segments) bin to eyeball
sharpness/exactness of an ARZ dataset.

For each chosen sample it renders, per field (rho, v):
  - the space-time heatmap (so you see the wave structure),
  - the final-time profile,
  - a ZOOM on the sharpest jump in the final timestep, annotated with the
    transition WIDTH in cells. WFT-exact shocks are ~1 cell wide; HLL-diffused
    shocks smear over 3-5 cells -> this is the visual contamination test.

One PNG per sample, named  <ic_type>_seg<NN>_idx<IIII>.png, plus a printed
per-bin summary of the mean sharpest-jump width.

Usage (CLEPS):
    python -m hyperbolic_pde.arz.plot_dataset_stratified \
        --data /home/dzdrale/scratch/arz_1d/arz_mixed_wft_prho_clean.npz \
        --pressure-form rho \
        --per-bin 2 \
        --out /home/dzdrale/scratch/results/wft_clean_check/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hyperbolic_pde.arz import physics_arz as P


def _transition_width_cells(profile: np.ndarray, frac: float = 0.8) -> tuple[int, int, int]:
    """Locate the sharpest jump in a 1D profile and measure how many cells the
    central `frac` of that jump spans. Returns (center_cell, width_cells, jump_sign).

    Width ~1 => sharp (single-cell transition). Width >=3 => diffused.
    """
    d = np.diff(profile)
    if d.size == 0 or np.allclose(d, 0):
        return 0, 0, 0
    j = int(np.argmax(np.abs(d)))                 # interface of steepest single-cell jump
    lo = max(0, j - 6)
    hi = min(profile.size - 1, j + 7)
    seg = profile[lo:hi + 1]
    span = seg[-1] - seg[0]
    if abs(span) < 1e-6:
        return j, 1, int(np.sign(d[j]))
    # cells whose value lies in the central `frac` band of the jump
    lo_band = seg[0] + (0.5 - frac / 2) * span
    hi_band = seg[0] + (0.5 + frac / 2) * span
    in_band = (seg - min(seg[0], seg[-1])) / span
    mask = (np.minimum(in_band, 1 - in_band) >= (0.5 - frac / 2))
    width = int(mask.sum())
    return j, max(width, 1), int(np.sign(span))


def plot_stratified(data_path: Path, pressure_form: str, per_bin: int,
                    out_dir: Path, seed: int) -> None:
    P.set_pressure_form(pressure_form)
    print(f"[plot] pressure_form = {P.get_pressure_form()}")

    d = np.load(data_path, allow_pickle=True)
    stored_pform = str(d["p_form"]) if "p_form" in d else "unknown"
    print(f"[plot] dataset p_form = {stored_pform}")

    N = d["rho"].shape[0]
    nt, nx = d["rho"].shape[1], d["rho"].shape[2]
    x = d["x"] if "x" in d else np.linspace(-1, 1, nx)
    t = d["t"] if "t" in d else np.linspace(0, 1, nt)
    ic_types = d["ic_type"] if "ic_type" in d else np.array(["?"] * N)
    segs = d["num_segments"].astype(int) if "num_segments" in d else np.zeros(N, dtype=int)

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    bins = sorted({(str(ic_types[i]), int(segs[i])) for i in range(N)})
    print(f"[plot] {len(bins)} (ic_type x segments) bins, {per_bin} samples each")

    bin_widths: dict[tuple, list] = {}

    for ic, seg in bins:
        idx_pool = np.where((ic_types == ic) & (segs == seg))[0]
        if idx_pool.size == 0:
            continue
        chosen = rng.choice(idx_pool, size=min(per_bin, idx_pool.size), replace=False)
        for idx in chosen:
            rho = d["rho"][idx]
            w = d["w"][idx]
            v = d["v"][idx] if "v" in d else w - P.pressure(rho)

            rho_T = rho[-1]
            v_T = v[-1]
            cj, cw, _ = _transition_width_cells(rho_T)
            bin_widths.setdefault((ic, seg), []).append(cw)

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for row, (fname, fld, fT) in enumerate(
                    [("rho", rho, rho_T), ("v", v, v_T)]):
                im = axes[row, 0].imshow(fld, aspect="auto", origin="lower",
                                         extent=[x[0], x[-1], t[0], t[-1]],
                                         cmap="viridis")
                axes[row, 0].set_title(f"{fname} space-time"); axes[row, 0].set_xlabel("x"); axes[row, 0].set_ylabel("t")
                plt.colorbar(im, ax=axes[row, 0])

                axes[row, 1].plot(x, fT, lw=1.2)
                axes[row, 1].set_title(f"{fname} final-time (t={t[-1]:.2f})")
                axes[row, 1].set_xlabel("x")

                # zoom on sharpest jump in this field's final profile
                jj, ww, _ = _transition_width_cells(fT)
                lo = max(0, jj - 8); hi = min(nx - 1, jj + 9)
                axes[row, 2].plot(np.arange(lo, hi + 1), fT[lo:hi + 1],
                                  "o-", ms=4, lw=1.0)
                axes[row, 2].axvline(jj + 0.5, color="r", ls="--", lw=0.8)
                axes[row, 2].set_title(f"{fname} sharpest jump  width~{ww} cell(s)")
                axes[row, 2].set_xlabel("cell index")

            fig.suptitle(f"{ic} | segs={seg} | idx={idx} | p={stored_pform} | "
                         f"rho jump width ~ {cw} cell(s)  "
                         f"({'SHARP' if cw <= 2 else 'SMEARED?'})", fontsize=11)
            plt.tight_layout()
            fname_png = out_dir / f"{ic}_seg{seg:02d}_idx{idx:04d}.png"
            plt.savefig(fname_png, dpi=110)
            plt.close(fig)
            print(f"  {ic:32s} seg{seg:<3d} idx{idx:<5d} rho jump width~{cw} -> {fname_png.name}")

    # ---- per-bin sharpness summary ----
    print("\n[plot] mean sharpest-rho-jump width per bin (cells; ~1 = sharp/WFT, >=3 = smeared/HLL):")
    print(f"  {'ic_type':32s} {'seg':>4s} {'mean_width':>10s}")
    worst = []
    for (ic, seg), ws in sorted(bin_widths.items()):
        mw = float(np.mean(ws))
        flag = "  <-- SMEARED?" if mw >= 3 else ""
        print(f"  {ic:32s} {seg:>4d} {mw:>10.2f}{flag}")
        if mw >= 3:
            worst.append((ic, seg, mw))
    if worst:
        print("\n[plot] WARNING: bins with mean jump width >=3 cells (possible non-WFT/HLL data):")
        for ic, seg, mw in worst:
            print(f"  {ic} seg{seg}: {mw:.2f} cells")
    else:
        print("\n[plot] All bins sharp (mean jump width < 3 cells). Consistent with WFT-exact GT.")
    print(f"\n[plot] figures -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--pressure-form", default="rho", choices=["rho", "rho+rho2"])
    parser.add_argument("--per-bin", type=int, default=2,
                        help="samples per (ic_type x num_segments) bin")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    plot_stratified(args.data, args.pressure_form, args.per_bin, args.out, args.seed)


if __name__ == "__main__":
    main()
