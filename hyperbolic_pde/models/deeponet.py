from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from .pinn import make_activation


def _build_mlp(
    in_dim: int,
    hidden_width: int,
    layers: int,
    out_dim: int,
    activation: Union[str, nn.Module],
) -> nn.Sequential:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    if hidden_width < 1:
        raise ValueError("hidden_width must be >= 1")
    modules: list[nn.Module] = []
    dim = in_dim
    for _ in range(layers - 1):
        modules.append(nn.Linear(dim, hidden_width))
        modules.append(make_activation(activation))
        dim = hidden_width
    modules.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*modules)


class DeepONet(nn.Module):
    """
    DeepONet with separate branch and trunk networks.

    branch: encodes the input function (initial conds).
    trunk: encodes the coordinate (x, t).
    Output is inner product of branch and trunk features.
    """

    def __init__(
        self,
        branch_in: int,
        trunk_in: int = 2,
        hidden_width: int = 64,
        branch_layers: int = 3,
        trunk_layers: int = 3,
        latent_dim: int = 64,
        activation: Union[str, nn.Module] = "tanh",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if branch_in < 1:
            raise ValueError("branch_in must be >= 1")
        if trunk_in < 1:
            raise ValueError("trunk_in must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")

        self.branch = _build_mlp(branch_in, hidden_width, branch_layers, latent_dim, activation)
        self.trunk = _build_mlp(trunk_in, hidden_width, trunk_layers, latent_dim, activation)
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None

    def forward(self, branch_in: torch.Tensor, trunk_in: torch.Tensor) -> torch.Tensor:
        if branch_in.dim() == 1:
            branch_in = branch_in.unsqueeze(0)
        if trunk_in.dim() == 1:
            trunk_in = trunk_in.unsqueeze(0)

        b = self.branch(branch_in)

        if trunk_in.dim() == 2:
            t = self.trunk(trunk_in)
            out = b @ t.T
        elif trunk_in.dim() == 3:
            bsz, n_pts, _ = trunk_in.shape
            t = self.trunk(trunk_in.reshape(-1, trunk_in.size(-1))).reshape(bsz, n_pts, -1)
            out = (b.unsqueeze(1) * t).sum(dim=-1)
        else:
            raise ValueError("trunk_in must have shape [P, trunk_in] or [B, P, trunk_in]")

        if self.bias is not None:
            out = out + self.bias
        return out
