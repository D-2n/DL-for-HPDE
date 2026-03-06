from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_t: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_t = modes_t
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_x, self.modes_t, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_x, self.modes_t, dtype=torch.cfloat))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            x.size(0),
            self.out_channels,
            x.size(-2),
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes_x, : self.modes_t] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, : self.modes_x, : self.modes_t],
            self.weights1,
        )
        # low negative x modes
        out_ft[:, :, -self.modes_x :, : self.modes_t] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes_x :, : self.modes_t],
            self.weights2,
        )
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        modes_x: int = 16,
        modes_t: int = 16,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.lift = nn.Sequential(
            nn.Linear(in_channels, width),
        )
        
        self.spectral_layers = nn.ModuleList()
        self.pointwise_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(layers):
            self.spectral_layers.append(SpectralConv2d(width, width, modes_x, modes_t))
            self.pointwise_layers.append(nn.Conv2d(width, width, 1))
            self.norms.append(nn.Identity())
        self.proj = nn.Sequential(
            nn.Linear(width, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        for spec, pw, norm in zip(self.spectral_layers, self.pointwise_layers, self.norms):
            x = norm(spec(x) + pw(x))
            x = torch.nn.functional.gelu(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
