from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import load_dataset


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
    parser = argparse.ArgumentParser(description="Plot generated hyperbolic PDE samples.")
    parser.add_argument(
        "--config",
        type=str,
        default="hyperbolic_pde/configs/hyperbolic_pde.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--num", type=int, default=3, help="Number of samples to plot.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="hyperbolic_pde/data/plots",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    dataset = load_dataset(Path(data_cfg["path"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num = min(args.num, dataset.u.shape[0])
    for i in range(num):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot(dataset.x, dataset.u0[i])
        axes[0].set_title(f"u0 sample {i}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("u")

        im = axes[1].pcolormesh(dataset.t, dataset.x, dataset.u[i].T, shading="auto", cmap="jet")
        axes[1].set_title(f"u(x,t) sample {i}")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("x")
        fig.colorbar(im, ax=axes[1], label="u")

        out_path = out_dir / f"pde_sample_{i}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"Saved {num} plots to {out_dir}")


if __name__ == "__main__":
    main()
