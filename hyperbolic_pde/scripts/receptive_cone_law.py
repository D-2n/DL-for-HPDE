"""Cross-architecture validation of the receptive-cone freeze law.

Hypothesis (from riemann_sweep_freeze.py):
    t_freeze * |s|  ==  L * k_x * dx        (per-sample)

If the receptive cone fully explains freeze, then across DIFFERENT
architectures the per-run empirical slope of (t_freeze vs 1/|s|) must
equal the cone reach L*k_x*dx exactly. Plot empirical vs predicted; the
points should land on y=x.

Usage:
    1. Run riemann_sweep_freeze.py on each architecture you want to test;
       it writes plots_riemann_sweep/metrics.txt under each run_dir.
    2. Edit RUNS_TO_AGGREGATE below.
    3. python -m hyperbolic_pde.scripts.receptive_cone_law

Outputs:
    receptive_cone_summary.png  — slope_emp vs cone_reach across runs
    receptive_cone_summary.txt  — table

What this tests vs. doesn't test:
    + Tests: does the LINEAR LAW slope=L*k_x*dx hold across architectures?
    + Tests: is the law tight (residuals small) or loose (residuals large)?
    - Does NOT test: is k_t binding? (need runs with same L*k_x but
      different L*k_t)
    - Does NOT test: charcone vs rectangle (need charcone runs)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------- #
# EDIT: list of run directories that already have plots_riemann_sweep/  #
# --------------------------------------------------------------------- #
RUNS_TO_AGGREGATE: list[str] = [
    # "/home/dzdrale/scratch/runs/hypno_st3/run_kx3_kt2_L6_xxx",
    # "/home/dzdrale/scratch/runs/hypno_st3/run_kx5_kt4_L7_xxx",
]


def _parse_metrics(metrics_path: Path) -> dict | None:
    """Pull n_layers, k_x, dx, cone_reach, t_max + per-sample (s, t_freeze)."""
    if not metrics_path.exists():
        return None
    text = metrics_path.read_text(encoding="utf-8")
    header = re.search(
        r"n_layers=(\d+)\s+k_x=(\d+)\s+dx=([\d.]+)\s+cone_reach=([\d.]+)\s+t_max=([\d.]+)",
        text,
    )
    if not header:
        return None
    n_layers   = int(header.group(1))
    k_x        = int(header.group(2))
    dx         = float(header.group(3))
    cone_reach = float(header.group(4))
    t_max      = float(header.group(5))

    # Per-sample rows: u_R s 1/|s| t_freeze t_theory ratio
    rows = re.findall(
        r"^\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s*$",
        text, flags=re.MULTILINE,
    )
    if not rows:
        return None
    s_arr  = np.array([float(r[1]) for r in rows])
    tf_arr = np.array([float(r[3]) for r in rows])
    return {
        "n_layers": n_layers, "k_x": k_x, "dx": dx,
        "cone_reach": cone_reach, "t_max": t_max,
        "s": s_arr, "t_freeze": tf_arr,
    }


def _empirical_slope(s: np.ndarray, tf: np.ndarray, t_max: float, dt_min: float) -> dict:
    """Linear fit of t_freeze vs 1/|s| with no intercept (slope only).

    Filters out clipped (t = t_max) and detector-glitch (t < 2*dt) points.
    Returns slope, R^2, n_points used.
    """
    inv_s = 1.0 / np.maximum(np.abs(s), 1e-12)
    reliable = (tf < t_max - 1e-6) & (tf > 2.0 * dt_min)
    if reliable.sum() < 3:
        return {"slope": float("nan"), "r2": float("nan"), "n": int(reliable.sum())}
    x = inv_s[reliable]
    y = tf[reliable]
    slope = float(np.sum(x * y) / np.sum(x * x))   # least-squares, no intercept
    y_hat = slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": slope, "r2": r2, "n": int(reliable.sum())}


def main() -> None:
    if not RUNS_TO_AGGREGATE:
        raise SystemExit(
            "RUNS_TO_AGGREGATE is empty. Edit receptive_cone_law.py and add "
            "absolute paths to run directories that already contain "
            "plots_riemann_sweep/metrics.txt."
        )

    rows = []
    for rs in RUNS_TO_AGGREGATE:
        run_dir = Path(rs)
        m = _parse_metrics(run_dir / "plots_riemann_sweep" / "metrics.txt")
        if m is None:
            print(f"  SKIP {run_dir} (no metrics.txt or unparseable)")
            continue
        # Estimate dt_min from t_freeze samples? Not available; use t_max/200 as a
        # safe lower bound for the glitch filter (matches default nt around 200).
        dt_min = m["t_max"] / 200.0
        fit = _empirical_slope(m["s"], m["t_freeze"], m["t_max"], dt_min)
        row = {
            "name":        run_dir.name,
            "n_layers":    m["n_layers"],
            "k_x":         m["k_x"],
            "dx":          m["dx"],
            "cone_reach":  m["cone_reach"],
            "slope_emp":   fit["slope"],
            "r2":          fit["r2"],
            "n_used":      fit["n"],
        }
        rows.append(row)
        print(f"  {row['name']}: L={row['n_layers']} k_x={row['k_x']} "
              f"cone={row['cone_reach']:.4f}  slope_emp={row['slope_emp']:.4f}  "
              f"R²={row['r2']:.3f}  n={row['n_used']}")

    if not rows:
        raise SystemExit("No runs parsed successfully.")

    cone = np.array([r["cone_reach"] for r in rows])
    slope = np.array([r["slope_emp"] for r in rows])
    r2 = np.array([r["r2"] for r in rows])

    # --------- Plot: slope_emp vs cone_reach (should land on y=x) ------- #
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sc = ax.scatter(cone, slope, c=r2, s=80, cmap="viridis",
                    vmin=0.5, vmax=1.0, edgecolor="k", zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("R² of per-run linear fit")

    lo = float(min(cone.min(), slope[~np.isnan(slope)].min())) * 0.9
    hi = float(max(cone.max(), slope[~np.isnan(slope)].max())) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.7,
            label="receptive-cone law: slope = L·k_x·dx")

    for r in rows:
        ax.annotate(f"L{r['n_layers']}·k{r['k_x']}",
                    (r["cone_reach"], r["slope_emp"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("cone reach  L · k_x · Δx  (predicted slope)")
    ax.set_ylabel("empirical slope  ⟨ t_freeze · |s| ⟩")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("Receptive-cone freeze law across architectures\n"
                 "(points on y=x ⇒ freeze fully explained by spatial reach)")

    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "receptive_cone_summary.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_png}")

    # --------- Text table ----------------------------------------------- #
    out_txt = out_dir / "receptive_cone_summary.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"{'name':40s}  {'L':>3}  {'k_x':>3}  {'dx':>7}  "
                f"{'cone':>7}  {'slope':>7}  {'ratio':>6}  {'R²':>5}  {'n':>3}\n")
        for r in rows:
            ratio = r["slope_emp"] / r["cone_reach"] if r["cone_reach"] > 0 else float("nan")
            f.write(f"{r['name']:40s}  {r['n_layers']:3d}  {r['k_x']:3d}  "
                    f"{r['dx']:7.4f}  {r['cone_reach']:7.4f}  "
                    f"{r['slope_emp']:7.4f}  {ratio:6.2f}  "
                    f"{r['r2']:5.2f}  {r['n_used']:3d}\n")
    print(f"Saved {out_txt}")


if __name__ == "__main__":
    main()
