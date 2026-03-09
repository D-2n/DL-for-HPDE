from __future__ import annotations

import torch
from torch import nn


def flux_lwr(u: torch.Tensor) -> torch.Tensor:
    return u * (1.0 - u)


def godunov_flux(u_left: torch.Tensor, u_right: torch.Tensor) -> torch.Tensor:
    f_left = flux_lwr(u_left)
    f_right = flux_lwr(u_right)
    f_min = torch.minimum(f_left, f_right)
    f_max = torch.maximum(f_left, f_right)

    u_lo = torch.minimum(u_left, u_right)
    u_hi = torch.maximum(u_left, u_right)
    has_mid = (u_lo <= 0.5) & (0.5 <= u_hi)
    f_max = torch.where(has_mid, torch.maximum(f_max, f_max.new_full((), 0.25)), f_max)

    return torch.where(u_left <= u_right, f_min, f_max)


def _make_mlp(in_dim: int, hidden: int, out_dim: int, layers: int, activation: str) -> nn.Sequential:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    act = nn.GELU if activation == "gelu" else nn.Tanh
    mods: list[nn.Module] = []
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(act())
    return nn.Sequential(*mods)


class FluxGNN1D(nn.Module):
    """
    Minimal FluxGNN-style model for 1D conservation laws.

    It learns a numerical flux on cell interfaces and updates cell averages
    by a conservative finite-volume step.
    """

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 3,
        activation: str = "gelu",
        use_base_flux: bool = True,
        base_flux_weight: float = 0.5,
        flux_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.flux_mlp = _make_mlp(2, hidden, 1, layers, activation)
        self.use_base_flux = use_base_flux
        self.base_flux_weight = float(base_flux_weight)
        self.flux_scale = float(flux_scale)

    def compute_flux(self, u_left: torch.Tensor, u_right: torch.Tensor) -> torch.Tensor:
        edge = torch.stack([u_left, u_right], dim=-1)
        flat = edge.reshape(-1, 2)
        flux_learned = self.flux_mlp(flat).reshape_as(u_left)
        flux_learned = torch.tanh(flux_learned) * self.flux_scale
        if self.use_base_flux: # for training stability, exclude by default
            flux_base = godunov_flux(u_left, u_right)
            w = self.base_flux_weight
            return (1.0 - w) * flux_base + w * flux_learned
        return flux_learned

    def step(self, u: torch.Tensor, dt: float, dx: float, boundary: str) -> torch.Tensor:
        if boundary == "periodic":
            u_right = torch.roll(u, shifts=-1, dims=1)
            flux = self.compute_flux(u, u_right)
            return u - (dt / dx) * (flux - torch.roll(flux, shifts=1, dims=1))
        if boundary == "ghost":
            u_ext = torch.empty(u.size(0), u.size(1) + 2, device=u.device, dtype=u.dtype)
            u_ext[:, 1:-1] = u
            u_ext[:, 0] = u[:, 0]
            u_ext[:, -1] = u[:, -1]
            flux = self.compute_flux(u_ext[:, :-1], u_ext[:, 1:])
            return u - (dt / dx) * (flux[:, 1:] - flux[:, :-1])
        if boundary == "fixed":
            u_left = u[:, :-1]
            u_right = u[:, 1:]
            flux = self.compute_flux(u_left, u_right)
            u_new = u.clone()
            u_new[:, 1:-1] = u[:, 1:-1] - (dt / dx) * (flux[:, 1:] - flux[:, :-1])
            u_new[:, 0] = u[:, 0]
            u_new[:, -1] = u[:, -1]
            return u_new
        raise ValueError("boundary must be 'periodic', 'ghost', or 'fixed'")

    def forward(self, u0: torch.Tensor, dt: float, dx: float, n_steps: int, boundary: str) -> torch.Tensor:
        u = u0
        outputs = [u]
        for _ in range(1, n_steps):
            u = self.step(u, dt, dx, boundary)
            outputs.append(u)
        return torch.stack(outputs, dim=1)
