"""Generate a grouped 256x256 OOD dataset for zero-shot resolution-transfer tests.

Identical in spirit to generate_ood_data.py, but the config block is selected
with --block (default 'ood_data_256_wide'). For every (ic_type, num_segments)
pair in that block this generates `n_samples` samples. The result is saved to a
standalone .npz carrying per-sample `num_segments` and `ic_type` metadata, so
final_comparison_256.py can group / label samples.

Samples are ordered by num_segments (then ic_type).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from hyperbolic_pde.utils.runtime import apply_runtime_overrides, resolve_config_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import generate_dataset


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> dict:
    base_path = ROOT / "configs" / "hyperbolic_pde.yaml"
    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path.resolve() == base_path.resolve():
        return cfg
    with path.open("r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    return _deep_update(cfg, override or {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate grouped 256x256 OOD hyperbolic PDE dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(resolve_config_path(ROOT / "configs")),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--block",
        type=str,
        default="ood_data_256_wide",
        help="Which config block to read (e.g. ood_data_256_wide, "
        "ood_data_256_train).",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    cfg = apply_runtime_overrides(cfg)

    if args.block not in cfg or cfg[args.block] is None:
        raise KeyError(
            f"Config has no '{args.block}:' block. Available 256 blocks: "
            f"{[k for k in cfg if k.startswith('ood_data_256')]}"
        )
    data_cfg = cfg[args.block]

    ic_types = data_cfg.get("ic_types", ["piecewise_constant"])
    if isinstance(ic_types, str):
        ic_types = [ic_types]

    num_segments = data_cfg["num_segments"]
    if not isinstance(num_segments, (list, tuple)):
        num_segments = [num_segments]
    num_segments = [int(s) for s in num_segments]

    # `n_samples` is the count generated per (ic_type, num_segments) pair.
    if "n_samples" in data_cfg:
        n_samples = int(data_cfg["n_samples"])
    else:
        n_samples = int(data_cfg["num_samples"])

    nx = int(data_cfg["nx"])
    nt = int(data_cfg["nt"])
    x_min = float(data_cfg["x_min"])
    x_max = float(data_cfg["x_max"])
    t_max = float(data_cfg["t_max"])
    cfl = float(data_cfg["cfl"])
    u_min = float(data_cfg["u_min"])
    u_max = float(data_cfg["u_max"])
    ic_points = int(data_cfg["ic_points"])
    boundary = str(data_cfg.get("boundary", "periodic"))
    method = str(data_cfg.get("method", "godunov"))
    num_workers = (
        int(data_cfg["num_workers"])
        if data_cfg.get("num_workers", None) is not None
        else None
    )
    base_seed = int(cfg.get("seed", 42))

    print(
        f"[OOD-256] block={args.block}  grid={nx}x{nt}  "
        f"x=[{x_min},{x_max}]  t_max={t_max}  "
        f"dt/dx={ (t_max/(nt-1)) / ((x_max-x_min)/(nx-1)) :.3f}"
    )

    u_blocks: list[np.ndarray] = []
    u0_blocks: list[np.ndarray] = []
    ic_blocks: list[np.ndarray] = []
    seg_meta: list[np.ndarray] = []
    ic_meta: list[str] = []
    x_ref: np.ndarray | None = None
    t_ref: np.ndarray | None = None

    pair_idx = 0
    for seg in num_segments:
        for ic_type in ic_types:
            seed = base_seed + 1000 * pair_idx
            pair_idx += 1
            print(
                f"\n[OOD-256] === pair {pair_idx}: ic_type={ic_type}, "
                f"num_segments={seg}, n_samples={n_samples} (seed={seed}) ==="
            )
            bundle = generate_dataset(
                num_samples=n_samples,
                nx=nx,
                nt=nt,
                x_min=x_min,
                x_max=x_max,
                t_max=t_max,
                cfl=cfl,
                num_segments=seg,
                u_min=u_min,
                u_max=u_max,
                ic_points=ic_points,
                boundary=boundary,
                seed=seed,
                num_workers=num_workers,
                ic_types=[ic_type],
                method=method,
            )
            if x_ref is None:
                x_ref, t_ref = bundle.x, bundle.t
            u_blocks.append(bundle.u)
            u0_blocks.append(bundle.u0)
            ic_blocks.append(bundle.ic)
            seg_meta.append(np.full(n_samples, seg, dtype=np.int64))
            ic_meta.extend([ic_type] * n_samples)

    u = np.concatenate(u_blocks, axis=0)
    u0 = np.concatenate(u0_blocks, axis=0)
    ic = np.concatenate(ic_blocks, axis=0)
    seg_arr = np.concatenate(seg_meta, axis=0)
    ic_arr = np.array(ic_meta, dtype=np.str_)

    out_path = Path(data_cfg["path"])
    from hyperbolic_pde import resolve_data_path

    out_path = resolve_data_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=x_ref,
        t=t_ref,
        u=u,
        u0=u0,
        ic=ic,
        num_segments=seg_arr,
        ic_type=ic_arr,
    )
    print(f"\n[OOD-256] Saved grouped dataset to {out_path}")
    print(f"[OOD-256] total samples: {u.shape[0]}")
    print(f"[OOD-256] u shape: {u.shape}, u0 shape: {u0.shape}, ic shape: {ic.shape}")
    for seg in num_segments:
        n = int((seg_arr == seg).sum())
        print(f"[OOD-256]   num_segments={seg}: {n} samples")


if __name__ == "__main__":
    main()
