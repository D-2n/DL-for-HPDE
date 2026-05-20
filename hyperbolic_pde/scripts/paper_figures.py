"""Consolidate paper_eval outputs into a small set of main-text figures.

Inputs (produced by paper_eval.py):
  <run-dir>/<paper_eval_dir>/summary.csv
  <run-dir>/<paper_eval_dir>/figures/compare_<ic_type>_seg<NN>.png

Outputs (written to <run-dir>/<paper_eval_dir>/paper/):
  faceted_mae_vs_num_segments.png   -- 1xK subplot grid, one panel per ic_type.
                                        Mean +/- std error bars per solver, log y.
                                        ID region (train num_segments) shaded.
  qualitative_id.png                -- copy of the chosen ID example.
  qualitative_ood.png               -- copy of the chosen OOD example.

The CSV-driven design means this script does NOT need a model or the dataset:
it can be re-run anytime to regenerate the consolidated views.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

SOLVERS = ["HypNO-ST3", "WENO5", "Godunov"]
COLORS = {"HypNO-ST3": "tab:blue", "WENO5": "tab:orange", "Godunov": "tab:green"}


def load_summary_csv(csv_path: Path):
    """Return per-cell rows only (ic_type != 'ALL') as list of dicts."""
    rows: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["ic_type"] == "ALL":
                continue
            rows.append({
                "ic_type": r["ic_type"],
                "num_segments": int(r["num_segments"]),
                "group": r["group"],
                "solver": r["solver"],
                "mae_mean": float(r["mae_mean"]),
                "mae_std": float(r["mae_std"]),
                "n_samples": int(r["n_samples"]),
            })
    return rows


def get_train_segs(run_dir: Path) -> list[int]:
    """Read training num_segments from the run's snapshot config."""
    cfg_path = run_dir / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [int(s) for s in cfg.get("data", {}).get("num_segments", [])]


def make_faceted_plot(
    rows: list[dict],
    train_segs: list[int],
    out_path: Path,
) -> None:
    """1 x len(ic_types) subplot grid. ID num_segments shaded."""
    ic_types = sorted({r["ic_type"] for r in rows})
    seg_values = sorted({r["num_segments"] for r in rows})

    fig, axes = plt.subplots(
        1, len(ic_types),
        figsize=(4.5 * len(ic_types), 4.5),
        sharey=True,
        constrained_layout=True,
    )
    if len(ic_types) == 1:
        axes = [axes]

    for ax, ic_type in zip(axes, ic_types):
        # ID region shading: span every contiguous run of training segs that
        # also appears on the x-axis.
        train_present = [s for s in train_segs if s in seg_values]
        if train_present:
            ax.axvspan(
                min(train_present) - 0.5,
                max(train_present) + 0.5,
                color="lightgrey", alpha=0.4, zorder=0,
                label="ID region (train num_segments)",
            )
        for name in SOLVERS:
            xs, ys, errs = [], [], []
            for seg in seg_values:
                cand = [
                    r for r in rows
                    if r["ic_type"] == ic_type
                    and r["num_segments"] == seg
                    and r["solver"] == name
                ]
                if not cand:
                    continue
                xs.append(seg)
                ys.append(cand[0]["mae_mean"])
                errs.append(cand[0]["mae_std"])
            ax.errorbar(
                xs, ys, yerr=errs, marker="o", capsize=4,
                label=name, color=COLORS[name], zorder=2,
            )
        ax.set_xlabel("num_segments")
        ax.set_xticks(seg_values)
        ax.set_yscale("log")
        ax.set_title(ic_type)
        ax.grid(True, alpha=0.3, zorder=1)

    axes[0].set_ylabel("MAE vs Lax-Hopf exact")
    # Single legend pulled from the last axes (all axes have same labels).
    handles, labels = axes[-1].get_legend_handles_labels()
    # Deduplicate while preserving order.
    seen = set()
    dedup = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    fig.legend(
        [h for h, _ in dedup],
        [l for _, l in dedup],
        loc="lower center", ncol=len(dedup), bbox_to_anchor=(0.5, -0.02),
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def copy_qualitative(
    figures_dir: Path,
    ic_type: str,
    seg: int,
    out_path: Path,
) -> None:
    src = figures_dir / f"compare_{ic_type}_seg{seg:02d}.png"
    if not src.exists():
        raise FileNotFoundError(
            f"Expected qualitative figure {src} not found. Was paper_eval.py "
            f"run with this (ic_type, num_segments) cell?"
        )
    shutil.copy2(src, out_path)


def parse_cell(spec: str) -> tuple[str, int]:
    """Parse 'ic_type:num_segments' into (ic_type, num_segments)."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"Expected 'ic_type:num_segments', got {spec!r}"
        )
    ic_type, seg = spec.split(":", 1)
    return ic_type.strip(), int(seg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build consolidated paper figures from paper_eval outputs."
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Run directory; must contain <paper_eval_dir>/summary.csv and figures/.",
    )
    parser.add_argument(
        "--paper_eval_dir", type=str, default="paper_eval",
        help="Subdirectory inside run-dir holding paper_eval outputs.",
    )
    parser.add_argument(
        "--id_cell", type=parse_cell, default="piecewise_constant:5",
        help="ID qualitative example as 'ic_type:num_segments'.",
    )
    parser.add_argument(
        "--ood_cell", type=parse_cell, default="sine_staircase:30",
        help="OOD qualitative example as 'ic_type:num_segments'.",
    )
    parser.add_argument(
        "--out_subdir", type=str, default="paper",
        help="Output subdirectory under <run-dir>/<paper_eval_dir>/.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pe_dir = run_dir / args.paper_eval_dir
    csv_path = pe_dir / "summary.csv"
    figures_dir = pe_dir / "figures"
    out_dir = pe_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        sys.exit(
            f"summary.csv not found at {csv_path}. Run paper_eval.py first."
        )
    if not figures_dir.exists():
        sys.exit(
            f"figures/ not found at {figures_dir}. Run paper_eval.py first."
        )

    rows = load_summary_csv(csv_path)
    train_segs = get_train_segs(run_dir)
    print(f"Loaded {len(rows)} per-cell rows from {csv_path}")
    print(f"Training num_segments: {train_segs}")

    # 1. Faceted MAE plot.
    faceted_path = out_dir / "faceted_mae_vs_num_segments.png"
    make_faceted_plot(rows, train_segs, faceted_path)
    print(f"Wrote {faceted_path}")

    # 2. Qualitative ID / OOD copies.
    id_ic, id_seg = args.id_cell
    ood_ic, ood_seg = args.ood_cell
    id_out = out_dir / f"qualitative_id_{id_ic}_seg{id_seg:02d}.png"
    ood_out = out_dir / f"qualitative_ood_{ood_ic}_seg{ood_seg:02d}.png"
    copy_qualitative(figures_dir, id_ic, id_seg, id_out)
    copy_qualitative(figures_dir, ood_ic, ood_seg, ood_out)
    print(f"Wrote {id_out}")
    print(f"Wrote {ood_out}")

    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
