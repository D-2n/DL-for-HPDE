from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class _LiftingLayer(nn.Module):
    """Mesh-invariant encoder: aggregates neighbor info via per-pair MLPs.

    For each cell i, computes:
      h_self = MLP(u_i, x_i)
      h_edge = sum_j MLP(u_i, u_j, (x_j - x_i) / L)
      h_i    = MLP(h_self || h_edge)

    No parameter depends on nx or dx.
    """

    def __init__(self, d_latent: int, d_hidden: int, stencil_k: int, activation: str) -> None:
        super().__init__()
        self.k = stencil_k
        self.node_mlp = _make_mlp(2, d_hidden, d_latent, 2, activation)
        self.edge_mlp = _make_mlp(3, d_hidden, d_latent, 2, activation)
        self.combine = _make_mlp(2 * d_latent, d_hidden, d_latent, 2, activation)

    def forward(self, u0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """u0: [B, nx], x: [B, nx] -> h: [B, nx, d_latent]."""
        B, nx = u0.shape
        k = self.k

        # per-cell features
        h_self = self.node_mlp(torch.stack([u0, x], dim=-1))            # [B, nx, d]

        # pad for neighbor access
        u_pad = F.pad(u0.unsqueeze(1), (k, k), mode="replicate").squeeze(1)
        x_pad = F.pad(x.unsqueeze(1), (k, k), mode="replicate").squeeze(1)

        # aggregate per-neighbor edge features
        agg = torch.zeros_like(h_self)
        for j in range(-k, k + 1):
            u_j = u_pad[:, k + j : k + j + nx]                         # [B, nx]
            dx = x_pad[:, k + j : k + j + nx] - x                      # [B, nx]
            edge_in = torch.stack([u0, u_j, dx], dim=-1)               # [B, nx, 3]
            agg = agg + self.edge_mlp(edge_in)

        return self.combine(torch.cat([h_self, agg], dim=-1))           # [B, nx, d]


class _MessagePassingLayer(nn.Module):
    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        d_time: int,
        stencil_k: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.k = stencil_k
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        # +2 extra features: u0_i - u0_j (raw gradient signal) and |u0_i - u0_j|
        msg_in_dim = 2 * d_latent + 1 + d_time + 2
        # message: (h_i, h_j, rel_x, tau, du0, |du0|) -> m
        self.msg_net = _make_mlp(msg_in_dim, d_hidden, d_latent, 3, activation)
        # attention gate: same input -> scalar gate per neighbor
        self.gate_net = _make_mlp(msg_in_dim, d_hidden, 1, 2, activation)
        # update: (h_i, aggregated_messages, tau) -> h_new  (nonlocal path = K(v))
        self.update_net = _make_mlp(2 * d_latent + d_time, d_hidden, d_latent, 3, activation)
        # local linear transform W — single matrix, mirrors FNO's Conv1x1
        self.W = nn.Linear(d_latent, d_latent)

    def forward(
        self, h: torch.Tensor, x: torch.Tensor, tau: torch.Tensor, u0: torch.Tensor,
    ) -> torch.Tensor:
        """
        h:   [B, nx, d_latent]
        x:   [B, nx]  physical coordinates
        tau: [B, d_time]
        u0:  [B, nx]  initial condition (for shock detection)
        """
        B, nx, d = h.shape
        k = self.k

        # ghost-cell padding along spatial dim
        h_pad = F.pad(h.permute(0, 2, 1), (k, k), mode="replicate").permute(0, 2, 1)
        x_pad = F.pad(x.unsqueeze(1), (k, k), mode="replicate").squeeze(1)
        u0_pad = F.pad(u0.unsqueeze(1), (k, k), mode="replicate").squeeze(1)

        tau_exp = tau.unsqueeze(1).expand(-1, nx, -1)  # [B, nx, d_time]

        # accumulate gated messages from all neighbors in stencil
        agg = h.new_zeros(B, nx, d)
        for j in range(-k, k + 1):
            h_j = h_pad[:, k + j : k + j + nx, :]                     # [B, nx, d]
            x_j = x_pad[:, k + j : k + j + nx]                        # [B, nx]
            u0_j = u0_pad[:, k + j : k + j + nx]                      # [B, nx]
            rel = (x_j - x).unsqueeze(-1)                              # [B, nx, 1]
            du0 = (u0 - u0_j).unsqueeze(-1)                           # [B, nx, 1]
            abs_du0 = du0.abs()                                        # [B, nx, 1]
            msg_in = torch.cat([h, h_j, rel, tau_exp, du0, abs_du0], dim=-1)
            msg = self.msg_net(msg_in)                                 # [B, nx, d]
            gate = torch.sigmoid(self.gate_net(msg_in))                # [B, nx, 1]
            agg = agg + gate * msg

        # nonlocal path: K(v) — gated message passing aggregate
        upd_in = torch.cat([h, agg, tau_exp], dim=-1)
        h_nonlocal = self.update_net(upd_in)                           # K(v)

        # local path: W·v — single linear (like FNO's Conv1x1)
        h_local = self.W(h)                                            # W·v

        return self.act(h_nonlocal + h_local)                          # σ(K(v) + W·v)


class HypGNO(nn.Module):
    """Hyperbolic Graph Neural Operator.

    Mesh-invariant: maps (u0, t, x) -> u(., t) for 1-D hyperbolic conservation laws.
    All operations are per-cell with physical coordinates — no parameter depends on nx.
    """

    def __init__(
        self,
        stencil_k: int = 3,
        d_latent: int = 128,
        d_hidden: int = 128,
        d_time: int = 32,
        n_layers: int = 8,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.stencil_k = stencil_k

        # lifting: mesh-invariant encoder with neighbor aggregation
        self.lifting = _LiftingLayer(d_latent, d_hidden, stencil_k, activation)

        # Fourier time embedding with learnable frequencies
        n_freq = d_time // 2
        self.time_freqs = nn.Parameter(torch.randn(n_freq) * 0.1)
        self.time_proj = nn.Linear(d_time, d_time)

        # message-passing layers
        self.mp_layers = nn.ModuleList(
            [
                _MessagePassingLayer(d_latent, d_hidden, d_time, stencil_k, activation)
                for _ in range(n_layers)
            ]
        )

        # decoder: latent + time -> predicted cell value
        self.decoder = _make_mlp(d_latent + d_time, d_hidden, 1, 3, activation)

    # ------------------------------------------------------------------ #
    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Fourier features of t.  t: [B] -> [B, d_time]."""
        phase = t.unsqueeze(-1) * self.time_freqs.unsqueeze(0)
        return self.time_proj(torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1))

    # ------------------------------------------------------------------ #
    def forward(self, u0: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        u0: [B, nx]   initial condition
        t:  [B]       query time  (scalar is broadcast)
        x:  [nx]      physical cell coordinates (any resolution)
        returns: [B, nx]  predicted solution at time t
        """
        B, nx = u0.shape
        k = self.stencil_k

        # broadcast x to batch: [B, nx]
        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)

        # --- lifting: mesh-invariant encoder with neighbor context ---
        h = self.lifting(u0, x)                                        # [B, nx, d_latent]

        # --- time embedding ---
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        tau = self._time_embed(t)                                      # [B, d_time]

        # --- message passing (u0 passed for shock-aware gating) ---
        for layer in self.mp_layers:
            h = layer(h, x, tau, u0)

        # --- decoder: predict residual, add skip from u0 ---
        tau_exp = tau.unsqueeze(1).expand(-1, nx, -1)
        correction = self.decoder(torch.cat([h, tau_exp], dim=-1)).squeeze(-1)  # [B, nx]
        return u0 + correction
