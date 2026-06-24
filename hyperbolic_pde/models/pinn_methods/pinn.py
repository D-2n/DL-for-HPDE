from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn


def make_activation(activation: Union[str, nn.Module]) -> nn.Module:
    if isinstance(activation, nn.Module):
        return activation.__class__()
    key = str(activation).lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError("activation must be relu, gelu, or tanh")


class UniversalPINN(nn.Module):
    """
    PINN conditioned on a fixed-length encoding of the initial condition.
    """

    def __init__(
        self,
        hidden_layers: int,
        hidden_width: int,
        cond_dim: int,
        activation: Union[str, nn.Module] = "tanh",
        hard_boundary: bool = False,
        x_min: float = -1.0,
        x_max: float = 1.0,
    ) -> None:
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        if hidden_width < 1:
            raise ValueError("hidden_width must be >= 1")
        if cond_dim < 0:
            raise ValueError("cond_dim must be >= 0")

        self.cond_dim = cond_dim
        self.hard_boundary = hard_boundary
        self.x_min = x_min
        self.x_max = x_max

        layers = []
        in_features = 2 + cond_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_width))
            layers.append(make_activation(activation))
            in_features = hidden_width
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond must be provided when cond_dim > 0")
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)
            inputs = torch.cat([x, t, cond], dim=1)
        else:
            inputs = torch.cat([x, t], dim=1)

        out = self.model(inputs)
        if self.hard_boundary:
            xi = (x - self.x_min) / (self.x_max - self.x_min)
            out = xi * (1.0 - xi) * out
        return out


def hyperbolic_residual(
    model: UniversalPINN,
    x: torch.Tensor,
    t: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model(x, t, cond)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    return u_t + (1.0 - 2.0 * u) * u_x


def repeat_cond(cond: torch.Tensor, n: int) -> torch.Tensor:
    """Repeat condition vector for n samples (cond shape: [cond_dim])."""
    if cond.dim() != 1:
        raise ValueError("cond must be a 1D tensor")
    return cond.unsqueeze(0).repeat(n, 1)


def sample_uniform(
    n: int,
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 1, device=device) * (x_max - x_min) + x_min
    t = torch.rand(n, 1, device=device) * (t_max - t_min) + t_min
    return x, t


def sample_initial(
    n: int,
    x_min: float,
    x_max: float,
    t_min: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 1, device=device) * (x_max - x_min) + x_min
    t = torch.zeros_like(x) + t_min
    return x, t


def sample_boundary(
    n: int,
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t_left = torch.rand(n, 1, device=device) * (t_max - t_min) + t_min
    t_right = torch.rand(n, 1, device=device) * (t_max - t_min) + t_min
    x_left = torch.zeros_like(t_left) + x_min
    x_right = torch.zeros_like(t_right) + x_max
    return x_left, t_left, x_right, t_right
