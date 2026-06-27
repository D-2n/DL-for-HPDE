from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn

from .pinn import UniversalPINN, make_activation


def flux(u: torch.Tensor) -> torch.Tensor:
    return u * (1.0 - u)


class VPINN(nn.Module):
    """
    Variational PINN for u_t + f(u)_x = 0 with f(u)=u(1-u).
    Uses sinusoidal test functions on normalized (x,t).
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
        t_min: float = 0.0,
        t_max: float = 1.0,
        n_test: int = 2,
    ) -> None:
        super().__init__()
        self.model = UniversalPINN(
            hidden_layers=hidden_layers,
            hidden_width=hidden_width,
            cond_dim=cond_dim,
            activation=activation,
            hard_boundary=hard_boundary,
            x_min=x_min,
            x_max=x_max,
        )
        self.cond_dim = cond_dim
        self.x_min = x_min
        self.x_max = x_max
        self.t_min = t_min
        self.t_max = t_max
        self.n_test = int(n_test)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(x, t, cond)

    def weak_residual_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.n_test < 1:
            raise ValueError("n_test must be >= 1")
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        u = self.forward(x, t, cond)
        ones = torch.ones_like(u)
        u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]

        xi = (x - self.x_min) / (self.x_max - self.x_min)
        tau = (t - self.t_min) / (self.t_max - self.t_min)

        loss = torch.zeros((), device=x.device)
        for m in range(1, self.n_test + 1):
            for n in range(1, self.n_test + 1):
                phi = torch.sin(m * math.pi * xi) * torch.sin(n * math.pi * tau)
                phi_x = (
                    (m * math.pi / (self.x_max - self.x_min))
                    * torch.cos(m * math.pi * xi)
                    * torch.sin(n * math.pi * tau)
                )
                integrand = u_t * phi + flux(u) * phi_x
                loss = loss + integrand.mean().pow(2)

        return loss / float(self.n_test * self.n_test)


__all__ = ["VPINN", "flux"]
