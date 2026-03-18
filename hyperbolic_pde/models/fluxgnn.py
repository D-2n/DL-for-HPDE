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
    def __init__(
        self,
        hidden: int = 64,
        layers: int = 3,
        activation: str = "gelu",
        latent_dim: int | None = None,
        flux_hidden: int | None = None,
        interface_hidden: int | None = None,
        in_dim: int = 1,
        use_base_flux: bool = False,
        base_flux_weight: float = 0.0,
        flux_scale: float = 0.25,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.latent_dim = int(latent_dim or hidden)
        self.flux_hidden = int(flux_hidden or hidden)
        self.interface_hidden = int(interface_hidden or hidden)

        self.encoder = nn.Linear(self.in_dim, self.latent_dim, bias=False)

        # First MLP: build interface representation from neighboring cells
        self.interface_mlp = _make_mlp(
            2 * self.latent_dim,
            self.interface_hidden,
            self.latent_dim,
            layers,
            activation,
        )

        # Second MLP: map interface representation to flux
        self.flux_mlp = _make_mlp(
           2*self.latent_dim,
            self.flux_hidden,
            self.latent_dim,
            layers,
            activation,
        )

        self.use_base_flux = bool(use_base_flux)
        self.base_flux_weight = float(base_flux_weight)
        self.flux_scale = float(flux_scale)
        self.eps = float(eps)

    def _decoder_weight(self) -> torch.Tensor:
        w = self.encoder.weight
        if self.in_dim == 1:
            denom = torch.sum(w * w) + self.eps
            return w.t() / denom
        return torch.linalg.pinv(w)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        return torch.matmul(u, self.encoder.weight.t())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        w_dec = self._decoder_weight()
        return torch.matmul(z, w_dec.t())

    def compute_flux(self, z_left: torch.Tensor, z_right: torch.Tensor) -> torch.Tensor:
        '''
        sym = torch.cat([z_left + z_right, torch.abs(z_left - z_right)], dim=-1)
        flat_sym = sym.reshape(-1, sym.size(-1))

        interface = self.interface_mlp(flat_sym).reshape_as(z_left)
        flat_interface = interface.reshape(-1, interface.size(-1))

        flux_learned = self.flux_mlp(flat_interface).reshape_as(z_left)
        '''
        sym = torch.cat([z_left + z_right, torch.abs(z_left - z_right)], dim=-1)
        flat_sym = sym.reshape(-1, sym.size(-1))

        flux_learned = self.flux_mlp(flat_sym).reshape_as(z_left)

        if self.flux_scale > 0:
            flux_learned = torch.tanh(flux_learned) * self.flux_scale

        #if self.use_base_flux:
           # u_left = self.decode(z_left)
           # u_right = self.decode(z_right)
           # flux_base = godunov_flux(u_left, u_right)
           # flux_base = self.encode(flux_base)
           # w = self.base_flux_weight
           # return (1.0 - w) * flux_base + w * flux_learned

        return flux_learned

    def step(self, z: torch.Tensor, dt: float, dx: float, boundary: str) -> torch.Tensor:
        if boundary == "periodic":
            z_right = torch.roll(z, shifts=-1, dims=1)
            flux = self.compute_flux(z, z_right)
            #return z - (dt / dx) * (flux - torch.roll(flux, shifts=1, dims=1))
            return flux

        if boundary == "ghost":
            repetitions = 3
            sub_dt = dt / repetitions
            for _ in range(repetitions):
                z_ext = torch.empty(z.size(0), z.size(1) + 2, z.size(2), device=z.device, dtype=z.dtype)
                z_ext[:, 1:-1] = z
                z_ext[:, 0] = z[:, 0]
                z_ext[:, -1] = z[:, -1]
                flux = self.compute_flux(z_ext[:, :-1], z_ext[:, 1:])
                z = z - (sub_dt / dx) * (flux[:, 1:] - flux[:, :-1])
            return z
          
 
        if boundary == "fixed":
            z_left = z[:, :-1]
            z_right = z[:, 1:]
            flux = self.compute_flux(z_left, z_right)
            z_new = z.clone()
            z_new[:, 1:-1] = z[:, 1:-1] - (dt / dx) * (flux[:, 1:] - flux[:, :-1])
            z_new[:, 0] = z[:, 0]
            z_new[:, -1] = z[:, -1]
            return z_new

        raise ValueError("boundary must be 'periodic', 'ghost', or 'fixed'")

    def forward(self, u0: torch.Tensor, dt: float, dx: float, n_steps: int, boundary: str) -> torch.Tensor:
        u0 = u0.unsqueeze(-1) if u0.dim() == 2 else u0
       # h = self.encode(u0)
       # z = torch.cat([u0, h], dim=-1)
        z = self.encode(u0)
        outputs = [self.decode(z).squeeze(-1)]
        for _ in range(1, n_steps):
            z = self.step(z, dt, dx, boundary)
            outputs.append(self.decode(z).squeeze(-1))
        return torch.stack(outputs, dim=1)