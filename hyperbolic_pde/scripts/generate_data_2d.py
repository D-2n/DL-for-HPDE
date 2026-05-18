"""Generate a 2D LWR dataset using WENO5 + Strang + RK3 (see fvm_2d.py).

Usage from repo root:
    python hyperbolic_pde/scripts/generate_data_2d.py
    python hyperbolic_pde/scripts/generate_data_2d.py --config <yaml>

Reads the ``data`` block of the YAML config and writes a compressed NPZ at
``data.path`` with keys: x, y, t, u, u0.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from hyperbolic_pde.data.fvm_2d import generate_dataset_2d, save_dataset_2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2D LWR dataset (WENO5 + Strang + RK3).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "hyperbolic_pde_2d.yaml"),
        help="Path to 2D YAML config.",
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

    t0 = time.time()
    bundle = generate_dataset_2d(
        num_samples=int(data_cfg["num_samples"]),
        nx=int(data_cfg["nx"]),
        ny=int(data_cfg["ny"]),
        nt=int(data_cfg["nt"]),
        x_min=float(data_cfg["x_min"]),
        x_max=float(data_cfg["x_max"]),
        y_min=float(data_cfg["y_min"]),
        y_max=float(data_cfg["y_max"]),
        t_max=float(data_cfg["t_max"]),
        cfl=float(data_cfg["cfl"]),
        num_rects=num_rects,
        u_min=float(data_cfg["u_min"]),
        u_max=float(data_cfg["u_max"]),
        boundary=str(data_cfg.get("boundary", "periodic")),
        seed=int(cfg.get("seed", 42)),
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
    print(f"elapsed : {elapsed:.1f} s  ({elapsed / bundle.u.shape[0] * 1000:.1f} ms / sample)")


if __name__ == "__main__":
    main()
