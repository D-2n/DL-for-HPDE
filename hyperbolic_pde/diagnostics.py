from __future__ import annotations

from typing import Any

import torch


def _godunov_flux_lwr(u_left: torch.Tensor, u_right: torch.Tensor) -> torch.Tensor:
    f_left = u_left * (1.0 - u_left)
    f_right = u_right * (1.0 - u_right)
    f_min = torch.minimum(f_left, f_right)
    f_max = torch.maximum(f_left, f_right)

    u_lo = torch.minimum(u_left, u_right)
    u_hi = torch.maximum(u_left, u_right)
    has_mid = (u_lo <= 0.5) & (0.5 <= u_hi)
    f_max = torch.where(has_mid, torch.maximum(f_max, f_max.new_full((), 0.25)), f_max)

    return torch.where(u_left <= u_right, f_min, f_max)


def godunov_one_step(u: torch.Tensor, dt: float, dx: float, boundary: str = "ghost") -> torch.Tensor:
    """
    Single Godunov step for f(u)=u(1-u) with optional boundary condition.
    u shape: [B, N] or [N].
    """
    if u.dim() == 1:
        u = u.unsqueeze(0)
    if u.dim() != 2:
        raise ValueError("u must have shape [B, N] or [N]")

    if boundary == "periodic":
        u_right = torch.roll(u, shifts=-1, dims=1)
        fhat = _godunov_flux_lwr(u, u_right)
        return u - (dt / dx) * (fhat - torch.roll(fhat, shifts=1, dims=1))

    if boundary == "ghost":
        u_ext = torch.empty(u.size(0), u.size(1) + 2, device=u.device, dtype=u.dtype)
        u_ext[:, 1:-1] = u
        u_ext[:, 0] = u[:, 0]
        u_ext[:, -1] = u[:, -1]
        fhat = _godunov_flux_lwr(u_ext[:, :-1], u_ext[:, 1:])
        return u - (dt / dx) * (fhat[:, 1:] - fhat[:, :-1])

    if boundary == "fixed":
        u_left = u[:, :-1]
        u_right = u[:, 1:]
        fhat = _godunov_flux_lwr(u_left, u_right)
        u_new = u.clone()
        u_new[:, 1:-1] = u[:, 1:-1] - (dt / dx) * (fhat[:, 1:] - fhat[:, :-1])
        u_new[:, 0] = u[:, 0]
        u_new[:, -1] = u[:, -1]
        return u_new

    raise ValueError("boundary must be 'periodic', 'ghost', or 'fixed'")


def model_one_step(
    model: Any,
    u: torch.Tensor,
    dt: float,
    dx: float,
    boundary: str = "ghost",
) -> torch.Tensor:
    """
    One-step update using the learned model.
    Assumes FluxGNN-style forward(u0, dt, dx, n_steps, boundary).
    Returns u_next with shape [B, N].
    """
    if u.dim() == 1:
        u = u.unsqueeze(0)
    if u.dim() != 2:
        raise ValueError("u must have shape [B, N] or [N]")
    out = model(u, dt, dx, 2, boundary)
    return out[:, 1]


def compare_one_step_update_sizes(
    model: Any,
    u: torch.Tensor,
    dt: float,
    dx: float,
    boundary: str = "ghost",
    label: str | None = None,
) -> None:
    """
    Compare one-step update magnitudes between learned model and Godunov.
    """
    with torch.no_grad():
        u_next_model = model_one_step(model, u, dt, dx, boundary=boundary)
        u_next_god = godunov_one_step(u, dt, dx, boundary=boundary)

    delta_model = u_next_model - u
    delta_god = u_next_god - u

    max_model = float(delta_model.abs().max().item())
    mean_model = float(delta_model.abs().mean().item())
    max_god = float(delta_god.abs().max().item())
    mean_god = float(delta_god.abs().mean().item())

    ratio_max = max_model / max_god if max_god > 0 else float("inf")
    ratio_mean = mean_model / mean_god if mean_god > 0 else float("inf")

    prefix = f"[OneStep{': ' + label if label else ''}]"
    print(f"{prefix} dt={dt:.6g} | dx={dx:.6g} | dt/dx={dt/dx:.6g}")
    print(
        f"{prefix} model update: max={max_model:.6g} | mean={mean_model:.6g}"
    )
    print(
        f"{prefix} godunov update: max={max_god:.6g} | mean={mean_god:.6g}"
    )
    print(
        f"{prefix} ratios (model/godunov): max={ratio_max:.6g} | mean={ratio_mean:.6g}"
    )
