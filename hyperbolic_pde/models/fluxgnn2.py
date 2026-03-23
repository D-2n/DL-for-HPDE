from __future__ import annotations

import torch
from torch import nn


def _make_mlp(in_dim: int, hidden: int, out_dim: int, layers: int, activation: str) -> nn.Sequential:
    act = nn.GELU if activation == "gelu" else nn.Tanh
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(act())
    return nn.Sequential(*mods)


class ConservativeFluxOperator1D(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden: int = 64,
        layers: int = 3,
        activation: str = "gelu",
        tau_embed_dim: int = 16,
        flux_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dim < 2:
            raise ValueError("latent_dim must be >= 2")

        self.latent_dim = latent_dim
        self.tau_embed_dim = tau_embed_dim
        self.flux_scale = flux_scale

        # Linear encoder for extra latent channels; first channel stays physical u
        self.encoder = nn.Linear(1, latent_dim - 1, bias=False)

        # Time embedding
        self.tau_mlp = _make_mlp(1, hidden, tau_embed_dim, layers, activation)

        # Node update / message passing block
        self.msg_mlp = _make_mlp(
            2 * latent_dim + 1,   # z_i, z_j, relative position sign
            hidden,
            latent_dim,
            layers,
            activation,
        )
        self.node_mlp = _make_mlp(
            2 * latent_dim,       # old node + aggregated messages
            hidden,
            latent_dim,
            layers,
            activation,
        )

        # Interface integrated-flux head: outputs scalar flux
        self.flux_mlp = _make_mlp(
            2 * latent_dim + tau_embed_dim,
            hidden,
            1,
            layers,
            activation,
        )

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, N, 1]
        h = self.encoder(u)                   # [B, N, latent_dim-1]
        z = torch.cat([u, h], dim=-1)        # [B, N, latent_dim]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # exact left inverse of z = [u, W u]
        return z[..., :1]

    def message_passing(self, z: torch.Tensor, boundary: str = "ghost") -> torch.Tensor:
        # z: [B, N, D]
        if boundary != "ghost":
            raise ValueError("This implementation assumes ghost BC only.")

        z_ext = torch.empty(z.size(0), z.size(1) + 2, z.size(2), device=z.device, dtype=z.dtype)
        z_ext[:, 1:-1] = z
        z_ext[:, 0] = z[:, 0]
        z_ext[:, -1] = z[:, -1]

        z_left  = z_ext[:, :-2]   # [B, N, D]
        z_mid   = z_ext[:, 1:-1]  # [B, N, D]
        z_right = z_ext[:, 2:]    # [B, N, D]

        # Two directional messages
        rel_left = -torch.ones_like(z_mid[..., :1])
        rel_right = torch.ones_like(z_mid[..., :1])

        m_left = self.msg_mlp(torch.cat([z_mid, z_left, rel_left], dim=-1))
        m_right = self.msg_mlp(torch.cat([z_mid, z_right, rel_right], dim=-1))

        agg = m_left + m_right
        z_new = self.node_mlp(torch.cat([z_mid, agg], dim=-1))

        # residual
        return z + z_new

    def compute_integrated_flux(self, z: torch.Tensor, tau: torch.Tensor, boundary: str = "ghost") -> torch.Tensor:
        # z: [B, N, D]
        # tau: [B, 1] or [B]
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)

        tau_emb = self.tau_mlp(tau).unsqueeze(1)  # [B, 1, T]
        tau_emb = tau_emb.expand(-1, z.size(1) + 1, -1)

        if boundary != "ghost":
            raise ValueError("This implementation assumes ghost BC only.")

        z_ext = torch.empty(z.size(0), z.size(1) + 2, z.size(2), device=z.device, dtype=z.dtype)
        z_ext[:, 1:-1] = z
        z_ext[:, 0] = z[:, 0]
        z_ext[:, -1] = z[:, -1]

        zL = z_ext[:, :-1]   # [B, N+1, D]
        zR = z_ext[:, 1:]    # [B, N+1, D]

        edge_feat = torch.cat([zL, zR, tau_emb], dim=-1)
        G = self.flux_mlp(edge_feat)  # [B, N+1, 1]

        if self.flux_scale > 0:
            G = torch.tanh(G) * self.flux_scale

        return G

    def forward(self, u_t: torch.Tensor, tau: torch.Tensor, dx: float, boundary: str = "ghost") -> torch.Tensor:
        # u_t: [B, N] or [B, N, 1]
        u_t = u_t.unsqueeze(-1) if u_t.dim() == 2 else u_t

        z = self.encode(u_t)
        z = self.message_passing(z, boundary=boundary)

        G = self.compute_integrated_flux(z, tau, boundary=boundary)   # [B, N+1, 1]

        # one-shot conservative update
        u_next = u_t - (G[:, 1:] - G[:, :-1]) / dx                    # [B, N, 1]
        return u_next.squeeze(-1)