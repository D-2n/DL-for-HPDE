from __future__ import annotations

import torch

from hyperbolic_pde.models.deeponet import DeepONet
from hyperbolic_pde.models.fno import FNO2d
from hyperbolic_pde.models.pinn import UniversalPINN, hyperbolic_residual


def test_universal_pinn_forward_and_residual() -> None:
    model = UniversalPINN(hidden_layers=2, hidden_width=16, cond_dim=4)
    x = torch.rand(10, 1)
    t = torch.rand(10, 1)
    cond = torch.rand(10, 4)
    out = model(x, t, cond)
    res = hyperbolic_residual(model, x, t, cond)
    assert out.shape == (10, 1)
    assert res.shape == (10, 1)
    assert torch.isfinite(res).all()


def test_fno_forward() -> None:
    model = FNO2d(in_channels=3, out_channels=1, width=16, modes_x=4, modes_t=4, layers=2)
    x = torch.rand(2, 3, 16, 8)
    y = model(x)
    assert y.shape == (2, 1, 16, 8)


def test_deeponet_forward() -> None:
    model = DeepONet(branch_in=8, trunk_in=2, hidden_width=16, branch_layers=2, trunk_layers=2, latent_dim=8)
    branch = torch.rand(4, 8)
    trunk = torch.rand(10, 2)
    out = model(branch, trunk)
    assert out.shape == (4, 10)
