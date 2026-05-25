"""Generate the 2D linear-advection v0 dataset.

Drives ``hyperbolic_pde.data.fvm_2d.generate_dataset_upsampled_2d`` with the
``data`` block of the v0 config — WENO5 + LF + SSP-RK3 at ``upsample_factor``
times the target grid, cell-averaged back down. Writes a compressed NPZ at
``data.path`` with keys: x, y, t, u, u0.

Usage from repo root:
    python hyperbolic_pde/scripts/generate_data_linear_2d_v0.py
    python hyperbolic_pde/scripts/generate_data_linear_2d_v0.py --config <yaml>
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import yaml

from hyperbolic_pde.data.fvm_2d import (
    generate_dataset_upsampled_2d,
    save_dataset_2d,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the 2D linear-advection v0 dataset (up-sampled WENO5)."
    )
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "hyperbolic_pde_linear_2d_v0.yaml"),
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    num_rects = data_cfg["num_rects"]
    if isinstance(num_rects, list):
        num_rects = (int(num_rects[0]), int(num_rects[1]))
    else:
        num_rects = int(num_rects)

    flux_kind = str(data_cfg.get("flux_kind", "linear"))
    a = data_cfg.get("a")
    b = data_cfg.get("b")
    if flux_kind == "linear" and (a is None or b is None):
        raise ValueError("flux_kind=linear requires `a` and `b` in data block")

    t0 = time.time()
    bundle = generate_dataset_upsampled_2d(
        num_samples=int(data_cfg["num_samples"]),
        nx=int(data_cfg["nx"]),
        ny=int(data_cfg["ny"]),
        nt=int(data_cfg["nt"]),
        flux_kind=flux_kind,
        a=float(a) if a is not None else None,
        b=float(b) if b is not None else None,
        upsample_factor=int(data_cfg.get("upsample_factor", 8)),
        x_min=float(data_cfg.get("x_min", 0.0)),
        x_max=float(data_cfg.get("x_max", 1.0)),
        y_min=float(data_cfg.get("y_min", 0.0)),
        y_max=float(data_cfg.get("y_max", 1.0)),
        t_max=float(data_cfg["t_max"]),
        cfl=float(data_cfg.get("cfl", 0.4)),
        num_rects=num_rects,
        u_min=float(data_cfg.get("u_min", 0.1)),
        u_max=float(data_cfg.get("u_max", 0.9)),
        boundary=str(data_cfg.get("boundary", "ghost")),
        seed=int(cfg.get("seed", 42)),
        num_workers=args.num_workers,
        verbose=True,
    )
    elapsed = time.time() - t0

    out_path = Path(data_cfg["path"])
    if not out_path.is_absolute():
        out_path = ROOT.parent / out_path
    save_dataset_2d(bundle, out_path)

    print(f"Saved dataset to {out_path}")
    print(f"u shape : {bundle.u.shape}  ({bundle.u.dtype})")
    print(f"u0 shape: {bundle.u0.shape}")
    print(f"x: {bundle.x.shape}, y: {bundle.y.shape}, t: {bundle.t.shape}")
    print(f"u range : [{bundle.u.min():.4f}, {bundle.u.max():.4f}]")
    print(
        f"elapsed : {elapsed:.1f} s  "
        f"({elapsed / max(1, bundle.u.shape[0]) * 1000:.1f} ms / sample)"
    )


if __name__ == "__main__":
    main()
