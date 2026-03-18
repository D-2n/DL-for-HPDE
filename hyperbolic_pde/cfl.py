from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def cfl_dt_max(dx: float, cfl: float, u_min: float, u_max: float) -> tuple[float, float]:
    """
    Compute the CFL-limited dt_max for the LWR flux f(u)=u(1-u).

    The characteristic speed is f'(u)=1-2u, so the max wave speed on
    [u_min, u_max] is max(|1-2u_min|, |1-2u_max|).
    """
    amax = max(abs(1.0 - 2.0 * u_min), abs(1.0 - 2.0 * u_max))
    amax = max(1e-6, amax)
    return amax, cfl * dx / amax


def compute_cfl_metrics(
    data_cfg: Mapping[str, Any],
    x: np.ndarray | None = None,
    t: np.ndarray | None = None,
    t_final: float | None = None,
) -> dict[str, float | int | None]:
    cfl = float(data_cfg.get("cfl", 0.0))
    u_min = float(data_cfg.get("u_min", 0.0))
    u_max = float(data_cfg.get("u_max", 1.0))

    dx = None
    if x is not None and x.size >= 2:
        dx = float(x[1] - x[0])
    elif all(k in data_cfg for k in ("x_min", "x_max", "nx")):
        nx = int(data_cfg["nx"])
        if nx > 1:
            dx = (float(data_cfg["x_max"]) - float(data_cfg["x_min"])) / (nx - 1)

    nt = None
    if t is not None:
        nt = int(t.size)
    elif "nt" in data_cfg:
        nt = int(data_cfg["nt"])

    t_final_use = t_final
    if t_final_use is None:
        if t is not None and t.size >= 1:
            t_final_use = float(t[-1])
        elif "t_max" in data_cfg:
            t_final_use = float(data_cfg["t_max"])

    dt_grid = None
    if t is not None and t.size >= 2:
        dt_grid = float(t[1] - t[0])
    elif t_final_use is not None and nt is not None and nt > 1:
        dt_grid = float(t_final_use / (nt - 1))

    amax = None
    dt_max = None
    cfl_eff = None
    nt_min = None
    if dx is not None:
        amax, dt_max = cfl_dt_max(dx, cfl, u_min, u_max)
        if dt_grid is not None:
            cfl_eff = dt_grid * amax / dx
        if t_final_use is not None and dt_max is not None:
            nt_min = int(math.ceil(t_final_use / dt_max) + 1)

    return {
        "cfl_cfg": cfl,
        "dx": dx,
        "amax": amax,
        "dt_max": dt_max,
        "dt_grid": dt_grid,
        "cfl_eff": cfl_eff,
        "nt_min": nt_min,
        "nt": nt,
        "t_final": t_final_use,
    }


def cfl_plot_label(metrics: Mapping[str, Any], prefix: str | None = None) -> str:
    parts: list[str] = []
    if metrics.get("cfl_eff") is not None:
        parts.append(f"cfl_eff={float(metrics['cfl_eff']):.4g}")
    if metrics.get("nt_min") is not None:
        parts.append(f"nt_min={int(metrics['nt_min'])}")
    if metrics.get("nt") is not None:
        parts.append(f"nt={int(metrics['nt'])}")
    label = " | ".join(parts)
    if prefix:
        return f"{prefix}: {label}" if label else prefix
    return label


def annotate_cfl(
    fig: Any,
    metrics: Mapping[str, Any],
    prefix: str | None = None,
    y: float = 0.01,
    fontsize: int = 9,
) -> None:
    label = cfl_plot_label(metrics, prefix=prefix)
    if not label:
        return
    fig.text(0.5, y, label, ha="center", va="bottom", fontsize=fontsize)


def print_cfl_report(
    data_cfg: Mapping[str, Any],
    x: np.ndarray | None = None,
    t: np.ndarray | None = None,
    label: str | None = None,
    t_final: float | None = None,
) -> dict[str, float | int | None]:
    metrics = compute_cfl_metrics(data_cfg, x=x, t=t, t_final=t_final)
    label_str = f" {label}" if label else ""

    parts = [f"cfl={float(metrics['cfl_cfg']):.6g}"]
    if metrics.get("dx") is not None:
        parts.append(f"dx={float(metrics['dx']):.6g}")
    if metrics.get("amax") is not None:
        parts.append(f"amax≈{float(metrics['amax']):.6g}")
    if metrics.get("dt_max") is not None:
        parts.append(f"dt_max≈{float(metrics['dt_max']):.6g}")
    if metrics.get("dt_grid") is not None:
        parts.append(f"dt_grid={float(metrics['dt_grid']):.6g}")
    if metrics.get("cfl_eff") is not None:
        parts.append(f"cfl_eff≈{float(metrics['cfl_eff']):.6g}")
    if metrics.get("nt_min") is not None:
        parts.append(f"nt_min={int(metrics['nt_min'])}")
    if metrics.get("nt") is not None:
        parts.append(f"nt={int(metrics['nt'])}")

    print(f"[CFL{label_str}] " + " | ".join(parts))
    return metrics
