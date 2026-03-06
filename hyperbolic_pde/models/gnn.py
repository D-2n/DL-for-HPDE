from __future__ import annotations

import torch
import torch.nn as nn


class GridGNN(nn.Module):
    """
    Simple grid-based GNN using neighbor aggregation on a 2D grid.
    Implemented as depthwise 3x3 message passing + pointwise mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 64,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels, hidden, 1)
        self.msg_layers = nn.ModuleList()
        self.mix_layers = nn.ModuleList()
        for _ in range(layers):
            self.msg_layers.append(nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden))
            self.mix_layers.append(nn.Conv2d(hidden, hidden, 1))
        self.proj = nn.Conv2d(hidden, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for msg, mix in zip(self.msg_layers, self.mix_layers):
            h = torch.nn.functional.gelu(msg(h) + mix(h))
        return self.proj(h)


__all__ = ["GridGNN"]
