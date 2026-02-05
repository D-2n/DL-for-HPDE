from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn

from . import config as cfg

alpha = cfg.ALPHA
xmin, xmax = cfg.X_MIN, cfg.X_MAX
tmin, tmax = cfg.T_MIN, cfg.T_MAX


def _make_activation(activation: Union[str, nn.Module]) -> nn.Module:
    if isinstance(activation, nn.Module):
        return activation.__class__()

    key = str(activation).lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError("Choose from: relu, gelu, tanh.")


class BurgersPINN(nn.Module):
    def __init__(
        self,
        layers: List[int],
        hard_boundary: bool = False,
        hard_init: bool = False,
        activation: Union[str, nn.Module] = "tanh",
    ) -> None:
        super().__init__()

        if not layers:
            raise ValueError("`layers` must contain at least one hidden width.")

        blocks = []
        in_features = 2
        for width in layers:
            blocks.append(nn.Linear(in_features, width))
            blocks.append(_make_activation(activation))
            in_features = width
        blocks.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*blocks)
        self.hard_boundary = hard_boundary
        self.hard_init = hard_init

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, t], dim=1)
        out = self.model(inputs)
        if self.hard_boundary:
            xi = (x - xmin) / (xmax - xmin)
            out = xi * (1.0 - xi) * out
        if self.hard_init:
            out = -torch.sin(torch.pi * x) + t * out
        return out


def burgers_residual(model: BurgersPINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    u = model(x, t)
    ones = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, ones, create_graph=True)[0]

    v = alpha
    return u_t + u * u_x - v * u_xx

