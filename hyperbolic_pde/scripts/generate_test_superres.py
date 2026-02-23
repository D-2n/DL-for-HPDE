from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import encode_ic, solve_conservation_fvm


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


def sample_piecewise_constant(
    x: np.ndarray,
    cuts: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(cuts, x, side="right")
    return values[idx].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paired 1x/2x test data for super-resolution evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]

    num_samples = int(data_cfg.get("test_num_samples", data_cfg["num_samples"]))
    nx1 = int(data_cfg["nx"])
    nt1 = int(data_cfg["nt"])
    nx2 = nx1 * 2
    nt2 = nt1 * 2
    x_min = float(data_cfg["x_min"])
    x_max = float(data_cfg["x_max"])
    t_max = float(data_cfg["t_max"])
    cfl = float(data_cfg["cfl"])
    num_segments = int(data_cfg["num_segments"])
    u_min = float(data_cfg["u_min"])
    u_max = float(data_cfg["u_max"])
    ic_points = int(data_cfg["ic_points"])
    boundary = str(data_cfg.get("boundary", "periodic"))

    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    x1 = np.linspace(x_min, x_max, nx1, dtype=np.float32)
    t1 = np.linspace(0.0, t_max, nt1, dtype=np.float32)
    x2 = np.linspace(x_min, x_max, nx2, dtype=np.float32)
    t2 = np.linspace(0.0, t_max, nt2, dtype=np.float32)

    u1 = np.zeros((num_samples, nt1, nx1), dtype=np.float32)
    u2 = np.zeros((num_samples, nt2, nx2), dtype=np.float32)
    u0_1 = np.zeros((num_samples, nx1), dtype=np.float32)
    u0_2 = np.zeros((num_samples, nx2), dtype=np.float32)
    ic_1 = np.zeros((num_samples, ic_points), dtype=np.float32)
    ic_2 = np.zeros((num_samples, ic_points), dtype=np.float32)

    for i in range(num_samples):
        cuts = rng.uniform(x_min, x_max, size=max(0, num_segments - 1))
        cuts.sort()
        values = rng.uniform(u_min, u_max, size=num_segments).astype(np.float32)

        u0_coarse = sample_piecewise_constant(x1, cuts, values)
        u0_fine = sample_piecewise_constant(x2, cuts, values)

        u0_1[i] = u0_coarse
        u0_2[i] = u0_fine
        ic_1[i] = encode_ic(u0_coarse, x1, ic_points)
        ic_2[i] = encode_ic(u0_fine, x2, ic_points)

        _, _, u_hist_1 = solve_conservation_fvm(
            u0=u0_coarse,
            x_min=x_min,
            x_max=x_max,
            t_max=t_max,
            nt_out=nt1,
            cfl=cfl,
            boundary=boundary,
        )
        _, _, u_hist_2 = solve_conservation_fvm(
            u0=u0_fine,
            x_min=x_min,
            x_max=x_max,
            t_max=t_max,
            nt_out=nt2,
            cfl=cfl,
            boundary=boundary,
        )
        u1[i] = u_hist_1
        u2[i] = u_hist_2

    out_path = Path(data_cfg.get("test_path", "hyperbolic_pde/data/hyperbolic_test_superres.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x_1x=x1,
        t_1x=t1,
        u_1x=u1,
        u0_1x=u0_1,
        ic_1x=ic_1,
        x_2x=x2,
        t_2x=t2,
        u_2x=u2,
        u0_2x=u0_2,
        ic_2x=ic_2,
    )
    print(f"Saved paired test dataset to {out_path}")
    print(f"u_1x shape: {u1.shape}, u_2x shape: {u2.shape}")


if __name__ == "__main__":
    main()
