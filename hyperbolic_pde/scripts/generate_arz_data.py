"""Generate ARZ dataset (Strang-split FV + relaxation ground truth)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.utils.runtime import apply_runtime_overrides, resolve_config_path
from hyperbolic_pde.arz.datagen_arz import generate_arz_dataset, save_arz_dataset


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> dict:
    base_path = ROOT / "configs" / "hyperbolic_pde_arz.yaml"
    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path.resolve() == base_path.resolve():
        return cfg
    with path.open("r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    return _deep_update(cfg, override or {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ARZ dataset.")
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "hyperbolic_pde_arz.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--section", type=str, default="arz_data",
        help="Which dataset section to read from the config (e.g. arz_data, arz_trial).",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    cfg = apply_runtime_overrides(cfg)
    data_cfg = cfg[args.section]

    # Apply pressure-closure switch *before* any physics computation runs.
    from hyperbolic_pde.arz.physics_arz import set_pressure_form, get_pressure_form
    p_form = str(data_cfg.get("pressure_form", "rho+rho2"))
    set_pressure_form(p_form)
    print(f"[generate_arz_data] config={args.config}  section={args.section}")
    print(f"[generate_arz_data] pressure_form={get_pressure_form()}")
    print(f"[generate_arz_data] target: {data_cfg['path']}")

    bundle = generate_arz_dataset(
        num_samples=int(data_cfg["num_samples"]),
        nx=int(data_cfg["nx"]),
        nt=int(data_cfg["nt"]),
        x_min=float(data_cfg["x_min"]),
        x_max=float(data_cfg["x_max"]),
        t_max=float(data_cfg["t_max"]),
        tau=float(data_cfg["tau"]),
        families=list(data_cfg["ic_types"]),
        segments=list(data_cfg["num_segments"]),
        cfl=float(data_cfg.get("cfl", 0.4)),
        boundary=str(data_cfg.get("boundary", "ghost")),
        refine=int(data_cfg.get("refine", 4)),
        use_exact_riemann=bool(data_cfg.get("use_exact_riemann", False)),
        seed=int(cfg.get("seed", 42)),
        rho_min=float(data_cfg.get("rho_min", 0.1)),
        rho_max=float(data_cfg.get("rho_max", 0.9)),
        v_min=float(data_cfg.get("v_min", 0.1)),
        v_max=float(data_cfg.get("v_max", 0.9)),
    )

    out_path = Path(data_cfg["path"])
    save_arz_dataset(bundle, out_path)
    print(f"[generate_arz_data] saved {out_path}  shapes: rho={bundle.rho.shape}")


if __name__ == "__main__":
    main()
