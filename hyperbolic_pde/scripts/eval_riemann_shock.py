from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))

from hyperbolic_pde.data.fvm import encode_ic
from hyperbolic_pde.models.deeponet import DeepONet
from hyperbolic_pde.models.fno import FNO2d as FNO
from hyperbolic_pde.models.fno_experiment import FNO2d as FNOExperiment
from hyperbolic_pde.models.fluxgnn import FluxGNN1D
from hyperbolic_pde.models.pinn import UniversalPINN
from hyperbolic_pde.models.vpinn import VPINN


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


def flux(u: float | np.ndarray) -> float | np.ndarray:
    return u * (1.0 - u)


def flux_prime(u: float | np.ndarray) -> float | np.ndarray:
    return 1.0 - 2.0 * u


def parse_int_list(raw: str) -> list[int]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        val = int(p)
        if val > 0:
            out.append(val)
    return out


def riemann_ic(x: np.ndarray, u_left: float, u_right: float, x0: float) -> np.ndarray:
    return np.where(x <= x0, u_left, u_right).astype(np.float32)


def riemann_solution(
    x: np.ndarray,
    t: float,
    u_left: float,
    u_right: float,
    x0: float,
) -> tuple[np.ndarray, dict]:
    if t <= 0.0 or math.isclose(u_left, u_right):
        return riemann_ic(x, u_left, u_right, x0), {"type": "constant"}

    s_left = float(flux_prime(u_left))
    s_right = float(flux_prime(u_right))

    if s_left > s_right:
        # Shock
        s = float((flux(u_right) - flux(u_left)) / (u_right - u_left))
        x_shock = x0 + s * t
        u = np.where(x < x_shock, u_left, u_right).astype(np.float32)
        return u, {"type": "shock", "speed": s, "x_shock": x_shock}

    # Rarefaction
    xi = (x - x0) / t
    u = np.empty_like(x, dtype=np.float32)
    u[xi <= s_left] = u_left
    u[xi >= s_right] = u_right
    mask = (xi > s_left) & (xi < s_right)
    u[mask] = 0.5 * (1.0 - xi[mask]).astype(np.float32)
    return u, {"type": "rarefaction", "x_left": x0 + s_left * t, "x_right": x0 + s_right * t}


def _crossings(x: np.ndarray, u: np.ndarray, level: float) -> list[float]:
    vals = u - level
    idx = np.where(vals[:-1] * vals[1:] <= 0.0)[0]
    out: list[float] = []
    for i in idx:
        u0, u1 = u[i], u[i + 1]
        x0, x1 = x[i], x[i + 1]
        if u1 == u0:
            out.append(0.5 * (x0 + x1))
        else:
            out.append(float(x0 + (level - u0) * (x1 - x0) / (u1 - u0)))
    return out


def estimate_shock_position(x: np.ndarray, u: np.ndarray, u_left: float, u_right: float) -> float:
    if u.size < 2:
        return float("nan")
    level = 0.5 * (u_left + u_right)
    grad = np.abs(np.diff(u))
    i = int(np.argmax(grad))
    x_ref = 0.5 * (x[i] + x[i + 1])
    candidates = _crossings(x, u, level)
    if not candidates:
        return x_ref
    return float(min(candidates, key=lambda xc: abs(xc - x_ref)))


def estimate_width(
    x: np.ndarray,
    u: np.ndarray,
    u_left: float,
    u_right: float,
    x_ref: float,
    levels: tuple[float, float],
) -> float:
    if math.isnan(x_ref):
        return float("nan")
    lo, hi = levels
    level_lo = u_left + lo * (u_right - u_left)
    level_hi = u_left + hi * (u_right - u_left)
    c_lo = _crossings(x, u, level_lo)
    c_hi = _crossings(x, u, level_hi)
    if not c_lo or not c_hi:
        return float("nan")
    x_lo = min(c_lo, key=lambda xc: abs(xc - x_ref))
    x_hi = min(c_hi, key=lambda xc: abs(xc - x_ref))
    return float(abs(x_hi - x_lo))


def build_fno_input(x: np.ndarray, t: np.ndarray, u0: np.ndarray) -> torch.Tensor:
    x_t = torch.tensor(x, dtype=torch.float32)
    t_t = torch.tensor(t, dtype=torch.float32)
    X, T = torch.meshgrid(x_t, t_t, indexing="ij")
    u0_t = torch.tensor(u0, dtype=torch.float32).unsqueeze(1).repeat(1, t_t.numel())
    inp = torch.stack([X, T, u0_t], dim=0)
    return inp


def predict_fluxgnn(
    model: FluxGNN1D,
    u0: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    boundary: str,
    device: torch.device,
) -> np.ndarray:
    dt = float(t[1] - t[0]) if t.size > 1 else float(t[-1])
    dx = float(x[1] - x[0])
    n_steps = int(t.size)
    u0_t = torch.tensor(u0, dtype=torch.float32, device=device).unsqueeze(0)
    pred = model(u0_t, dt, dx, n_steps, boundary)[0].detach().cpu().numpy()
    return pred  # [T, X]


def predict_fno(
    model: torch.nn.Module,
    u0: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    inp = build_fno_input(x, t, u0).unsqueeze(0).to(device)
    pred = model(inp)[0, 0].detach().cpu().numpy()  # [X, T]
    return pred.T  # [T, X]


def predict_deeponet(
    model: DeepONet,
    ic: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    X, T = torch.meshgrid(x_t, t_t, indexing="ij")
    trunk = torch.stack([X, T], dim=-1).reshape(-1, 2)
    branch = torch.tensor(ic, dtype=torch.float32, device=device)
    pred = model(branch, trunk).detach().cpu().numpy().reshape(x.size, t.size)
    return pred.T


def predict_pinn(
    model: torch.nn.Module,
    ic: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    X, T = torch.meshgrid(x_t, t_t, indexing="ij")
    Xf = X.reshape(-1, 1)
    Tf = T.reshape(-1, 1)
    cond = torch.tensor(ic, dtype=torch.float32, device=device).unsqueeze(0).repeat(Xf.size(0), 1)
    pred = model(Xf, Tf, cond).detach().cpu().numpy().reshape(x.size, t.size)
    return pred.T


def load_model(model_name: str, cfg: dict, data_cfg: dict, device: torch.device) -> torch.nn.Module:
    model_name = model_name.lower()
    if model_name == "fluxgnn":
        flux_cfg = cfg["fluxgnn"]
        model = FluxGNN1D(
            hidden=int(flux_cfg["hidden"]),
            layers=int(flux_cfg["layers"]),
            activation=str(flux_cfg.get("activation", "gelu")),
            latent_dim=flux_cfg.get("latent_dim"),
            flux_hidden=flux_cfg.get("flux_hidden"),
            use_base_flux=bool(flux_cfg.get("use_base_flux", True)),
            base_flux_weight=float(flux_cfg.get("base_flux_weight", 0.5)),
            flux_scale=float(flux_cfg.get("flux_scale", 0.25)),
        ).to(device)
        model.load_state_dict(torch.load(Path(flux_cfg["save_path"]), map_location=device))
        model.eval()
        return model
    if model_name == "fno":
        fno_cfg = cfg["fno"]
        model = FNO(
            in_channels=3,
            out_channels=1,
            width=int(fno_cfg["width"]),
            modes_x=int(fno_cfg["modes_x"]),
            modes_t=int(fno_cfg["modes_t"]),
            layers=int(fno_cfg["layers"]),
        ).to(device)
        model.load_state_dict(torch.load(Path(fno_cfg["save_path"]), map_location=device))
        model.eval()
        return model
    if model_name == "fno_experiment":
        fno_cfg = cfg["fno_experiment"]
        model = FNOExperiment(
            in_channels=3,
            out_channels=1,
            width=int(fno_cfg["width"]),
            modes_x=int(fno_cfg["modes_x"]),
            modes_t=int(fno_cfg["modes_t"]),
            layers=int(fno_cfg["layers"]),
            band_start=int(fno_cfg.get("band_start", 50)),
        ).to(device)
        model.load_state_dict(torch.load(Path(fno_cfg["save_path"]), map_location=device))
        model.eval()
        return model
    if model_name == "deeponet":
        deep_cfg = cfg["deeponet"]
        model = DeepONet(
            branch_in=int(data_cfg["ic_points"]),
            trunk_in=2,
            hidden_width=int(deep_cfg["hidden_width"]),
            branch_layers=int(deep_cfg["branch_layers"]),
            trunk_layers=int(deep_cfg["trunk_layers"]),
            latent_dim=int(deep_cfg["latent_dim"]),
            activation=str(deep_cfg.get("activation", "tanh")),
            use_bias=bool(deep_cfg.get("use_bias", True)),
        ).to(device)
        model.load_state_dict(torch.load(Path(deep_cfg["save_path"]), map_location=device))
        model.eval()
        return model
    if model_name == "pinn":
        pinn_cfg = cfg["pinn"]
        model = UniversalPINN(
            hidden_layers=int(pinn_cfg["hidden_layers"]),
            hidden_width=int(pinn_cfg["hidden_width"]),
            cond_dim=int(data_cfg["ic_points"]),
            activation=str(pinn_cfg.get("activation", "tanh")),
            hard_boundary=bool(pinn_cfg.get("hard_boundary", False)),
            x_min=float(data_cfg["x_min"]),
            x_max=float(data_cfg["x_max"]),
        ).to(device)
        model.load_state_dict(torch.load(Path(pinn_cfg["save_path"]), map_location=device))
        model.eval()
        return model
    if model_name == "vpinn":
        vpinn_cfg = cfg["vpinn"]
        model = VPINN(
            hidden_layers=int(vpinn_cfg["hidden_layers"]),
            hidden_width=int(vpinn_cfg["hidden_width"]),
            cond_dim=int(data_cfg["ic_points"]),
            activation=str(vpinn_cfg.get("activation", "tanh")),
            hard_boundary=bool(vpinn_cfg.get("hard_boundary", False)),
            x_min=float(data_cfg["x_min"]),
            x_max=float(data_cfg["x_max"]),
            t_min=0.0,
            t_max=float(data_cfg.get("t_max", 1.0)),
            n_test=int(vpinn_cfg.get("n_test", 2)),
        ).to(device)
        model.load_state_dict(torch.load(Path(vpinn_cfg["save_path"]), map_location=device))
        model.eval()
        return model

    raise ValueError(f"Unknown model '{model_name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Riemann shock/rarefaction evaluation across nt.")
    parser.add_argument("--config", type=str, default="hyperbolic_pde/configs/hyperbolic_pde.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="fluxgnn",
        choices=["fluxgnn", "fno", "fno_experiment", "deeponet", "pinn", "vpinn"],
    )
    parser.add_argument("--u_left", type=float, default=0.2)
    parser.add_argument("--u_right", type=float, default=0.8)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--t_final", type=float, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--x_min", type=float, default=None)
    parser.add_argument("--x_max", type=float, default=None)
    parser.add_argument("--nt_list", type=str, default="11,21,41")
    parser.add_argument("--width_levels", type=str, default="0.1,0.9")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    x_min = float(args.x_min) if args.x_min is not None else float(data_cfg["x_min"])
    x_max = float(args.x_max) if args.x_max is not None else float(data_cfg["x_max"])
    nx = int(args.nx) if args.nx is not None else int(data_cfg["nx"])
    t_final = float(args.t_final) if args.t_final is not None else float(data_cfg.get("t_max", 0.5))

    nt_list = parse_int_list(args.nt_list)
    if not nt_list:
        nt_list = [int(data_cfg.get("nt", 11))]

    width_parts = [float(p.strip()) for p in args.width_levels.split(",") if p.strip()]
    if len(width_parts) != 2:
        raise ValueError("width_levels must be two comma-separated values like '0.1,0.9'")
    width_levels = (width_parts[0], width_parts[1])

    x = np.linspace(x_min, x_max, nx, dtype=np.float32)
    u0 = riemann_ic(x, args.u_left, args.u_right, args.x0)
    ic = encode_ic(u0, x, int(data_cfg["ic_points"]))

    model = load_model(args.model, cfg, data_cfg, device)

    _, info = riemann_solution(x, t_final, args.u_left, args.u_right, args.x0)
    print(f"[Riemann] type={info['type']} | uL={args.u_left} uR={args.u_right} | x0={args.x0} | t_final={t_final}")
    if info["type"] == "shock":
        print(f"[Riemann] true shock speed={info['speed']:.6f} | true shock pos={info['x_shock']:.6f}")
    elif info["type"] == "rarefaction":
        print(f"[Riemann] true fan edges: xL={info['x_left']:.6f}, xR={info['x_right']:.6f}")

    header = "nt, pred_pos, pred_speed, width"
    if info["type"] == "shock":
        header += ", pos_err"
    print(header)

    boundary = str(data_cfg.get("boundary", "ghost"))

    for nt in nt_list:
        t = np.linspace(0.0, t_final, int(nt), dtype=np.float32)

        if args.model == "fluxgnn":
            pred_hist = predict_fluxgnn(model, u0, x, t, boundary, device)
        elif args.model in ("fno", "fno_experiment"):
            pred_hist = predict_fno(model, u0, x, t, device)
        elif args.model == "deeponet":
            pred_hist = predict_deeponet(model, ic, x, t, device)
        elif args.model in ("pinn", "vpinn"):
            pred_hist = predict_pinn(model, ic, x, t, device)
        else:
            raise ValueError(f"Unsupported model '{args.model}'")

        u_pred = pred_hist[-1]
        x_pred = estimate_shock_position(x, u_pred, args.u_left, args.u_right)
        width = estimate_width(x, u_pred, args.u_left, args.u_right, x_pred, width_levels)
        speed_pred = (x_pred - args.x0) / t_final if not math.isnan(x_pred) else float("nan")

        row = [f"{nt:d}", f"{x_pred:.6f}", f"{speed_pred:.6f}", f"{width:.6f}"]
        if info["type"] == "shock":
            pos_err = x_pred - info["x_shock"]
            row.append(f"{pos_err:.6f}")
        print(", ".join(row))


if __name__ == "__main__":
    main()
