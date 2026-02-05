from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from . import config as cfg
from .pinn import BurgersPINN


X_MIN, X_MAX = cfg.X_MIN, cfg.X_MAX
T_MIN, T_MAX = cfg.T_MIN, cfg.T_MAX


def sample_interior(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    t = torch.rand(n, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    return x, t


def sample_initial(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    t = torch.zeros_like(x) + T_MIN
    return x, t


def sample_boundary(
    n: int, device: torch.device
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    t_left = torch.rand(n, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    t_right = torch.rand(n, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    x_left = torch.zeros_like(t_left) + X_MIN
    x_right = torch.zeros_like(t_right) + X_MAX
    return (x_left, t_left), (x_right, t_right)


def burgers_initial_condition(x: torch.Tensor) -> torch.Tensor:
    return -torch.sin(torch.pi * x)


class BurgersVPINN(torch.nn.Module):
    def __init__(
        self,
        layers: List[int],
        hard_boundary: bool = False,
        hard_init: bool = False,
        activation: str = "tanh",
        viscosity: float = cfg.ALPHA,
        n_test: int = 2,
    ) -> None:
        super().__init__()
        self.pinn = BurgersPINN(
            layers=layers,
            hard_boundary=hard_boundary,
            hard_init=hard_init,
            activation=activation,
        )
        self.viscosity = float(viscosity)
        self.n_test = int(n_test)
        self.hard_boundary = hard_boundary
        self.hard_init = hard_init

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.pinn(x, t)

    def weak_residual_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        u = self.forward(x, t)
        ones = torch.ones_like(u)
        u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]

        xi = (x - X_MIN) / (X_MAX - X_MIN)
        tau = (t - T_MIN) / (T_MAX - T_MIN)
        loss = torch.zeros((), device=x.device)

        for m in range(1, self.n_test + 1):
            for n in range(1, self.n_test + 1):
                phi = torch.sin(m * math.pi * xi) * torch.sin(n * math.pi * tau)
                phi_x = (
                    (m * math.pi / (X_MAX - X_MIN))
                    * torch.cos(m * math.pi * xi)
                    * torch.sin(n * math.pi * tau)
                )
                weak_integrand = (
                    u_t * phi
                    - 0.5 * u.pow(2) * phi_x
                    + self.viscosity * u_x * phi_x
                )
                loss = loss + weak_integrand.mean().pow(2)

        return loss / float(self.n_test * self.n_test)


def train_vpinn(
    layers: List[int],
    steps: int = 2000,
    lr: float = 1e-3,
    interior_samples: int = 1024,
    boundary_samples: int = 256,
    initial_samples: int = 256,
    hard_boundary: bool = False,
    hard_init: bool = False,
    activation: str = "tanh",
    viscosity: float = cfg.ALPHA,
    n_test: int = 2,
    device: Optional[torch.device] = None,
    log_every: int = 250,
) -> Tuple[BurgersVPINN, List[Dict[str, float]]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BurgersVPINN(
        layers=layers,
        hard_boundary=hard_boundary,
        hard_init=hard_init,
        activation=activation,
        viscosity=viscosity,
        n_test=n_test,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[Dict[str, float]] = []

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)

        x_i, t_i = sample_interior(interior_samples, device=device)
        loss_weak = model.weak_residual_loss(x_i, t_i)

        loss_bc = torch.zeros((), device=device)
        if not hard_boundary:
            (x_l, t_l), (x_r, t_r) = sample_boundary(boundary_samples, device=device)
            loss_bc = model(x_l, t_l).pow(2).mean() + model(x_r, t_r).pow(2).mean()

        loss_ic = torch.zeros((), device=device)
        if not hard_init:
            x_0, t_0 = sample_initial(initial_samples, device=device)
            target_0 = burgers_initial_condition(x_0)
            loss_ic = (model(x_0, t_0) - target_0).pow(2).mean()

        loss = loss_weak + loss_bc + loss_ic
        loss.backward()
        optimizer.step()

        row = {
            "step": float(step),
            "loss": float(loss.item()),
            "weak": float(loss_weak.item()),
            "boundary": float(loss_bc.item()),
            "initial": float(loss_ic.item()),
        }
        history.append(row)

        if step == 1 or step % log_every == 0:
            print(
                f"[VPINN] step {step:4d} | total={row['loss']:.3e} | "
                f"weak={row['weak']:.3e} | ic={row['initial']:.3e} | bc={row['boundary']:.3e}"
            )

    return model, history


__all__ = [
    "BurgersVPINN",
    "train_vpinn",
    "sample_interior",
    "sample_initial",
    "sample_boundary",
]
