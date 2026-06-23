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
    """Mesh-invariant spatial encoder with neighbor aggregation.

    Operates on u0 along x only. Output is then tiled across time.
    Supports k-hop stencil or physical radius neighbor selection.
    """

    def __init__(self, d_latent: int, d_hidden: int, stencil_k: int, activation: str,
                 radius_x: float | None = None) -> None:
        super().__init__()
        self.k = stencil_k
        self.radius_x = radius_x
        self.node_mlp = _make_mlp(2, d_hidden, d_latent, 2, activation)
        self.edge_mlp = _make_mlp(4, d_hidden, d_latent, 2, activation)
        self.gate_net = _make_mlp(4, d_hidden, 1, 2, activation)
        self.combine = _make_mlp(2 * d_latent, d_hidden, d_latent, 2, activation)

    def _get_max_k(self, x: torch.Tensor) -> int:
        """Compute max k-hop needed to cover radius_x at the given resolution."""
        dx = (x[0, 1] - x[0, 0]).abs().item()
        return max(1, int(self.radius_x / dx + 0.5))

    def forward(self, u0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """u0: [B, nx], x: [B, nx] -> h: [B, nx, d_latent]."""
        B, nx = u0.shape
        k = self._get_max_k(x) if self.radius_x is not None else self.k

        h_self = self.node_mlp(torch.stack([u0, x], dim=-1))

        u_pad = F.pad(u0.unsqueeze(1), (k, k), mode="replicate").squeeze(1)
        x_pad = F.pad(x.unsqueeze(1), (k, k), mode="replicate").squeeze(1)

        agg = torch.zeros_like(h_self)
        for j in range(-k, k + 1):
            u_j = u_pad[:, k + j : k + j + nx]
            dx = x_pad[:, k + j : k + j + nx] - x
            # If using radius, mask out neighbors beyond the physical distance
            if self.radius_x is not None:
                mask = (dx.abs() <= self.radius_x).unsqueeze(-1)  # [B, nx, 1]
            du = u0 - u_j
            edge_in = torch.stack([u0, u_j, dx, du.abs()], dim=-1)
            msg = self.edge_mlp(edge_in)
            gate = torch.sigmoid(self.gate_net(edge_in))
            contrib = gate * msg
            if self.radius_x is not None:
                contrib = contrib * mask
            agg = agg + contrib

        return self.combine(torch.cat([h_self, agg], dim=-1))


class _SpaceTimeMPLayer(nn.Module):
    """Factored space-time message passing layer.

    Mirrors FNO's 2D spectral conv but with local message passing:
      1. Spatial MP: messages along x (at each t), with shock-aware gating
      2. Temporal MP: messages along t (at each x)
      3. Combine: σ(K_space(v) + K_time(v) + W·v)

    This is the GNO analog of FNO's SpectralConv2d.
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        k_x: int,
        k_t: int,
        activation: str,
        radius_x: float | None = None,
        radius_t: float | None = None,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_t = k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()

        # spatial message: (h_i, h_j, dx, du0, |du0|) -> d_latent
        sp_in = 2 * d_latent + 3
        self.sp_msg = _make_mlp(sp_in, d_hidden, d_latent, 3, activation)
        self.sp_gate = _make_mlp(sp_in, d_hidden, 1, 2, activation)

        # temporal message: (h_i, h_j, dt) -> d_latent
        tp_in = 2 * d_latent + 1
        self.tp_msg = _make_mlp(tp_in, d_hidden, d_latent, 3, activation)
        self.tp_gate = _make_mlp(tp_in, d_hidden, 1, 2, activation)

        # update: (h, spatial_agg + temporal_agg) -> h_nonlocal  (K(v))
        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        # local linear transform W — mirrors FNO's Conv1x1
        self.W = nn.Linear(d_latent, d_latent)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        u0: torch.Tensor,
    ) -> torch.Tensor:
        """
        h:   [B, nt, nx, d_latent]  space-time latent field
        x:   [B, nx]                physical spatial coordinates
        t:   [nt]                   physical time coordinates
        u0:  [B, nx]                initial condition (for shock-aware gating)
        """
        B, nt, nx, d = h.shape

        # Compute effective k from radius if set, otherwise use fixed stencil
        if self.radius_x is not None:
            dx = (x[0, 1] - x[0, 0]).abs().item()
            k_x = max(1, int(self.radius_x / dx + 0.5))
        else:
            k_x = self.k_x
        if self.radius_t is not None:
            dt = (t[1] - t[0]).abs().item()
            k_t = max(1, int(self.radius_t / dt + 0.5))
        else:
            k_t = self.k_t

        # ---- Spatial message passing (along x, for each t) ----
        # pad h along x: flatten (B,nt) -> BN, pad as 3D [BN, d, nx], unflatten
        h_flat = h.reshape(B * nt, nx, d).permute(0, 2, 1)              # [B*nt, d, nx]
        h_xp = F.pad(h_flat, (k_x, k_x), mode="replicate")             # [B*nt, d, nx+2k]
        h_xp = h_xp.permute(0, 2, 1).reshape(B, nt, nx + 2 * k_x, d)  # [B, nt, nx+2k, d]

        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)  # [B, nx+2k]
        u0_pad = F.pad(u0.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)

        sp_agg = h.new_zeros(B, nt, nx, d)
        for j in range(-k_x, k_x + 1):
            h_j = h_xp[:, :, k_x + j : k_x + j + nx, :]               # [B, nt, nx, d]
            # spatial and u0 features: [B, nx] -> [B, 1, nx, 1] broadcast to [B, nt, nx, 1]
            rel_x = (x_pad[:, k_x + j : k_x + j + nx] - x).unsqueeze(1).unsqueeze(-1)
            du0 = (u0 - u0_pad[:, k_x + j : k_x + j + nx]).unsqueeze(1).unsqueeze(-1)
            abs_du0 = du0.abs()
            rel_x = rel_x.expand_as(h[:, :, :, :1])
            du0 = du0.expand_as(rel_x)
            abs_du0 = abs_du0.expand_as(rel_x)

            msg_in = torch.cat([h, h_j, rel_x, du0, abs_du0], dim=-1)
            msg = self.sp_msg(msg_in)
            gate = torch.sigmoid(self.sp_gate(msg_in))
            contrib = gate * msg
            # Mask out neighbors beyond physical radius
            if self.radius_x is not None:
                r_mask = (rel_x.abs() <= self.radius_x)                  # [B, nt, nx, 1]
                contrib = contrib * r_mask
            sp_agg = sp_agg + contrib

        # ---- Temporal message passing (along t, for each x) ----
        # pad h along t: flatten (B,nx) -> BX, pad as 3D [BX, d, nt], unflatten
        h_flat_t = h.permute(0, 2, 1, 3).reshape(B * nx, nt, d).permute(0, 2, 1)  # [B*nx, d, nt]
        h_tp = F.pad(h_flat_t, (k_t, k_t), mode="replicate")            # [B*nx, d, nt+2k]
        h_tp = h_tp.permute(0, 2, 1).reshape(B, nx, nt + 2 * k_t, d)   # [B, nx, nt+2k, d]
        h_tp = h_tp.permute(0, 2, 1, 3)                                 # [B, nt+2k, nx, d]

        t_pad = F.pad(t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate").squeeze(0).squeeze(0)

        tp_agg = h.new_zeros(B, nt, nx, d)
        for j in range(-k_t, k_t + 1):
            h_j = h_tp[:, k_t + j : k_t + j + nt, :, :]               # [B, nt, nx, d]
            t_j = t_pad[k_t + j : k_t + j + nt]                        # [nt]
            # rel_t: [nt] -> [1, nt, 1, 1] broadcast to [B, nt, nx, 1]
            rel_t = (t_j - t).view(1, nt, 1, 1).expand(B, nt, nx, 1)
            msg_in = torch.cat([h, h_j, rel_t], dim=-1)
            msg = self.tp_msg(msg_in)
            gate = torch.sigmoid(self.tp_gate(msg_in))
            contrib = gate * msg
            # Mask out neighbors beyond physical radius
            if self.radius_t is not None:
                t_mask = (rel_t.abs() <= self.radius_t)                  # [B, nt, nx, 1]
                contrib = contrib * t_mask
            tp_agg = tp_agg + contrib

        # ---- Combine: σ(K(v) + W·v) ----
        upd_in = torch.cat([h, sp_agg + tp_agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)                             # K(v)
        h_local = self.W(h)                                              # W·v

        return self.act(h_nonlocal + h_local)                            # σ(K(v) + W·v)


class HypNO(nn.Module):
    """Hyperbolic Neural Operator.

    Space-time neural operator for 1-D hyperbolic conservation laws.
    Maps u0 -> u(x, t) for all (x, t) simultaneously, like FNO.

    Architecture: P -> [σ(K(v) + W·v)]^T -> Q
      - P: mesh-invariant lifting (spatial neighbor aggregation on u0)
      - K: factored space-time message passing (spatial + temporal)
      - W: pointwise linear (like FNO's Conv1x1)
      - Q: pointwise decoder with skip from u0

    The kernel K replaces FNO's spectral convolution with local
    message passing — no FFT, no grid-size parameters, shock-aware gating.
    """

    def __init__(
        self,
        stencil_k_x: int = 3,
        stencil_k_t: int = 2,
        d_latent: int = 128,
        d_hidden: int = 128,
        d_time: int = 32,
        n_layers: int = 6,
        activation: str = "gelu",
        radius_x: float | None = None,
        radius_t: float | None = None,
    ) -> None:
        super().__init__()
        self.stencil_k_x = stencil_k_x
        self.stencil_k_t = stencil_k_t
        self.radius_x = radius_x
        self.radius_t = radius_t

        # P: lifting — spatial encoder on u0
        self.lifting = _LiftingLayer(d_latent, d_hidden, stencil_k_x, activation, radius_x=radius_x)

        # Fourier time embedding with learnable frequencies
        n_freq = d_time // 2
        self.time_freqs = nn.Parameter(torch.randn(n_freq) * 0.1)
        self.time_proj = nn.Linear(d_time, d_time)

        # project (lifted_spatial || time_embedding) -> d_latent
        self.time_mix = _make_mlp(d_latent + d_time, d_hidden, d_latent, 2, activation)

        # space-time message-passing layers (analog of FNO's SpectralConv2d)
        self.mp_layers = nn.ModuleList(
            [
                _SpaceTimeMPLayer(d_latent, d_hidden, stencil_k_x, stencil_k_t, activation,
                                  radius_x=radius_x, radius_t=radius_t)
                for _ in range(n_layers)
            ]
        )

        # Q: decoder — latent -> predicted cell value
        self.decoder = _make_mlp(d_latent, d_hidden, 1, 3, activation)

    # ------------------------------------------------------------------ #
    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Fourier features.  t: [nt] -> [nt, d_time]."""
        phase = t.unsqueeze(-1) * self.time_freqs.unsqueeze(0)
        return self.time_proj(torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1))

    # ------------------------------------------------------------------ #
    def forward(
        self,
        u0: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        u0: [B, nx]   initial condition
        x:  [nx]      physical cell coordinates (any resolution)
        t:  [nt]      physical time coordinates (any temporal grid)
        returns: [B, nt, nx]  predicted solution at all (x, t)
        """
        B, nx = u0.shape
        nt = t.shape[0]

        # broadcast x to batch
        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)                            # [B, nx]

        # --- P: lifting (spatial, on u0) ---
        h_spatial = self.lifting(u0, x)                                  # [B, nx, d_latent]

        # --- time embedding ---
        tau = self._time_embed(t)                                        # [nt, d_time]

        # --- tile spatial features across time, concat time embedding ---
        h_exp = h_spatial.unsqueeze(1).expand(B, nt, nx, -1)            # [B, nt, nx, d_latent]
        tau_exp = tau.unsqueeze(0).unsqueeze(2).expand(B, nt, nx, -1)   # [B, nt, nx, d_time]
        h = self.time_mix(torch.cat([h_exp, tau_exp], dim=-1))          # [B, nt, nx, d_latent]

        # --- space-time message passing ---
        for layer in self.mp_layers:
            h = layer(h, x, t, u0)

        # --- Q: decoder with skip from u0 ---
        correction = self.decoder(h).squeeze(-1)                         # [B, nt, nx]
        u0_exp = u0.unsqueeze(1).expand(B, nt, nx)
        return u0_exp + correction
