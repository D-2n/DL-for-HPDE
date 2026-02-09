from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import generate_dataset, save_dataset


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
    parser = argparse.ArgumentParser(description="Generate hyperbolic PDE dataset via Godunov FVM.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]

    bundle = generate_dataset(
        num_samples=int(data_cfg["num_samples"]),
        nx=int(data_cfg["nx"]),
        nt=int(data_cfg["nt"]),
        x_min=float(data_cfg["x_min"]),
        x_max=float(data_cfg["x_max"]),
        t_max=float(data_cfg["t_max"]),
        cfl=float(data_cfg["cfl"]),
        num_segments=int(data_cfg["num_segments"]),
        u_min=float(data_cfg["u_min"]),
        u_max=float(data_cfg["u_max"]),
        ic_points=int(data_cfg["ic_points"]),
        boundary=str(data_cfg.get("boundary", "periodic")),
        seed=int(cfg.get("seed", 42)),
    )

    out_path = Path(data_cfg["path"])
    save_dataset(bundle, out_path)
    print(f"Saved dataset to {out_path}")
    print(f"u shape: {bundle.u.shape}, u0 shape: {bundle.u0.shape}, ic shape: {bundle.ic.shape}")


if __name__ == "__main__":
    main()
