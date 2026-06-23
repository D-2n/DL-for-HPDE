"""Standalone PINN shock detector.

Proper PINN: MLP maps (x, t, z) → u(x,t), where z encodes the initial
condition u0.  Trained with PDE residual loss (autograd) + IC loss.
No ground truth solution needed at interior points.

At inference the PDE residual |u_t + (u(1-u))_x| spikes at shocks
because the strong-form PDE doesn't hold across discontinuities.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(in_dim: int, hidden: int, out_dim: int, layers: int, activation: str) -> nn.Sequential:
    act_map = {"gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}
    act = act_map.get(activation, nn.GELU)
    mods: list[nn.Module] = []
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(act())
    return nn.Sequential(*mods)


class ShockDetector(nn.Module):
    """PINN shock detector: (x, t, z) → u(x,t).

    Architecture:
      1. IC encoder: MLP encodes u0 into a fixed-size latent z
      2. PINN trunk: MLP(x, t, z) → u   (with tanh activation for smoothness)
      3. Derivatives via autograd: du/dt, du/dx → PDE residual

    Training losses:
      - PDE: |u_t + (u(1-u))_x|  at random collocation points
      - IC:  |u(x, 0) - u0(x)|
      - BC:  boundary conditions (optional)

    The PINN will be smooth → shocks show up as high PDE residual.
    """

    def __init__(
        self,
        d_latent: int = 64,
        d_hidden: int = 128,
        n_layers: int = 6,
        activation: str = "tanh",
        ic_points: int = 128,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent

        # IC encoder: u0 [nx] → z [d_latent]
        # Simple: flatten u0 and project
        self.ic_encoder = _make_mlp(ic_points, d_hidden, d_latent, 3, "gelu")

        # PINN trunk: (x, t, z) → u
        # Use tanh — smooth activation, standard for PINNs
        self.trunk = _make_mlp(2 + d_latent, d_hidden, 1, n_layers, activation)

    def encode_ic(self, u0: torch.Tensor) -> torch.Tensor:
        """u0: [B, nx] → z: [B, d_latent]."""
        return self.ic_encoder(u0)

    def predict_pointwise(
        self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate u at given (x, t) points.

        x: [B, N] or [B, N, 1]  — spatial coordinates (requires_grad for PDE)
        t: [B, N] or [B, N, 1]  — temporal coordinates (requires_grad for PDE)
        z: [B, d_latent]        — IC encoding (broadcast to all points)

        Returns: u [B, N]
        """
        if x.dim() == 3:
            x = x.squeeze(-1)
        if t.dim() == 3:
            t = t.squeeze(-1)

        B, N = x.shape
        z_exp = z.unsqueeze(1).expand(B, N, self.d_latent)  # [B, N, d_latent]
        inp = torch.cat([x.unsqueeze(-1), t.unsqueeze(-1), z_exp], dim=-1)  # [B, N, 2+d_latent]
        return self.trunk(inp).squeeze(-1)  # [B, N]

    def pde_residual(
        self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Compute PDE residual via autograd: R = u_t + (u(1-u))_x.

        x, t must have requires_grad=True.
        Returns: residual [B, N]
        """
        u = self.predict_pointwise(x, t, z)

        # du/dt and du/dx via autograd
        grad_u = torch.autograd.grad(
            u, [t, x],
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )
        u_t = grad_u[0]  # [B, N]
        u_x = grad_u[1]  # [B, N]

        # LWR flux: f(u) = u(1-u), f'(u) = 1 - 2u
        # d[u(1-u)]/dx = (1 - 2u) * u_x
        f_x = (1.0 - 2.0 * u) * u_x

        return u_t + f_x

    def predict_grid(
        self, u0: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predict u on a full (x, t) grid. For eval/inference (no autograd needed).

        u0: [B, nx], x: [nx] or [B, nx], t: [nt]
        Returns: u_pred [B, nt, nx]
        """
        B, nx = u0.shape
        nt = t.shape[0]

        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)

        z = self.encode_ic(u0)  # [B, d_latent]

        # build grid of (x, t) points: [B, nt*nx]
        x_grid = x.unsqueeze(1).expand(B, nt, nx).reshape(B, nt * nx)
        t_grid = t.unsqueeze(0).unsqueeze(-1).expand(B, nt, nx).reshape(B, nt * nx)

        u_flat = self.predict_pointwise(x_grid, t_grid, z)  # [B, nt*nx]
        return u_flat.reshape(B, nt, nx)

    def compute_shock_indicator_grid(
        self, u0: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute shock indicator on a full grid via autograd.

        Returns: (shock_indicator [B, nt, nx], u_pred [B, nt, nx])
        """
        B, nx = u0.shape
        nt = t.shape[0]

        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)

        z = self.encode_ic(u0)

        # build grid with requires_grad
        x_grid = x.unsqueeze(1).expand(B, nt, nx).reshape(B, nt * nx)
        t_grid = t.unsqueeze(0).unsqueeze(-1).expand(B, nt, nx).reshape(B, nt * nx)

        x_g = x_grid.detach().requires_grad_(True)
        t_g = t_grid.detach().requires_grad_(True)

        residual = self.pde_residual(x_g, t_g, z)  # [B, nt*nx]
        residual = residual.detach().abs().reshape(B, nt, nx)

        u_pred = self.predict_pointwise(x_grid, t_grid, z).detach().reshape(B, nt, nx)

        # normalize to [0, 1] per sample
        r_max = residual.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        shock_indicator = residual / r_max

        return shock_indicator, u_pred

    def forward(
        self, u0: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward: returns (shock_indicator, u_pred) on grid."""
        return self.compute_shock_indicator_grid(u0, x, t)
