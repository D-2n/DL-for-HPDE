from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_mlp(in_dim: int, hidden: int, out_dim: int, layers: int, activation: str) -> nn.Sequential:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    act_map = {"gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}
    act = act_map.get(activation, nn.GELU)
    mods: list[nn.Module] = []
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(act())
    return nn.Sequential(*mods)


# --------------------------------------------------------------------------- #
# lifting layer  (identical to hypno_pinn.py)
# --------------------------------------------------------------------------- #
class _LiftingLayer(nn.Module):
    """Mesh-invariant spatial encoder with neighbor aggregation on u0."""

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
        dx = (x[0, 1] - x[0, 0]).abs().item()
        return max(1, int(self.radius_x / dx + 0.5))

    def forward(self, u0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, nx = u0.shape
        k = self._get_max_k(x) if self.radius_x is not None else self.k

        h_self = self.node_mlp(torch.stack([u0, x], dim=-1))

        u_pad = F.pad(u0.unsqueeze(1), (k, k), mode="replicate").squeeze(1)
        x_pad = F.pad(x.unsqueeze(1), (k, k), mode="replicate").squeeze(1)

        agg = torch.zeros_like(h_self)
        for j in range(-k, k + 1):
            u_j = u_pad[:, k + j : k + j + nx]
            dx_val = x_pad[:, k + j : k + j + nx] - x
            if self.radius_x is not None:
                mask = (dx_val.abs() <= self.radius_x).unsqueeze(-1)
            du = u0 - u_j
            edge_in = torch.stack([u0, u_j, dx_val, du.abs()], dim=-1)
            msg = self.edge_mlp(edge_in)
            gate = torch.sigmoid(self.gate_net(edge_in))
            contrib = gate * msg
            if self.radius_x is not None:
                contrib = contrib * mask
            agg = agg + contrib

        return self.combine(torch.cat([h_self, agg], dim=-1))


# --------------------------------------------------------------------------- #
# flux-based spatial message-passing layer
# --------------------------------------------------------------------------- #
class _FluxSpaceMPLayer(nn.Module):
    """Learned numerical flux at cell interfaces with WENO smoothness gating.

    Physical interpretation:
      F_{i+1/2} = FluxMLP(h_i, h_{i+1}, u0_avg, u0_jump)   — learned interface flux
      h_i -= alpha * (F_{i+1/2} - F_{i-1/2}) / dx           — FVM flux-divergence update

    WENO beta computed from F values (not h values): large beta = flux varies
    sharply = shock region => small omega => suppressed update.

    All nt time slices are processed in parallel (reshape to [B*nt, nx, d]).
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        activation: str,
        weno_eps: float = 1e-6,
        weno_p: float = 2.0,
    ) -> None:
        super().__init__()
        self.weno_eps = weno_eps
        self.weno_p = weno_p

        # FluxMLP: inputs [h_L, h_R, u0_avg, u0_jump] — dim = 2*d_latent + 2
        flux_in = 2 * d_latent + 2
        self.flux_mlp = _make_mlp(flux_in, d_hidden, d_latent, 3, activation)

        # learnable CFL scalar: softplus(raw_alpha) = 0.1 initially
        self.raw_alpha = nn.Parameter(
            torch.tensor(math.log(math.exp(0.1) - 1.0))
        )

    def _flux(
        self,
        h_L: torch.Tensor,    # [BT, nx, d]
        h_R: torch.Tensor,    # [BT, nx, d]
        u0_avg: torch.Tensor, # [BT, nx, 1]
        u0_jump: torch.Tensor,# [BT, nx, 1]
    ) -> torch.Tensor:
        """Antisymmetric flux: F(h_L, h_R) = -F(h_R, h_L).  F(h, h) = 0."""
        inp_fwd = torch.cat([h_L, h_R, u0_avg, u0_jump], dim=-1)
        inp_rev = torch.cat([h_R, h_L, u0_avg, -u0_jump], dim=-1)
        return 0.5 * (self.flux_mlp(inp_fwd) - self.flux_mlp(inp_rev))

    def forward(
        self,
        h: torch.Tensor,   # [B, nt, nx, d]
        x: torch.Tensor,   # [B, nx]
        t: torch.Tensor,   # [nt]   (unused here, carried for API consistency)
        u0: torch.Tensor,  # [B, nx]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sp_agg  : [B, nt, nx, d]   flux-divergence aggregation
            beta_map: [B, nt, nx]      WENO smoothness indicator (zeros if weno_p==0)
        """
        B, nt, nx, d = h.shape
        BT = B * nt

        dx = (x[0, 1] - x[0, 0]).abs()

        # --- reshape to [BT, nx, d] for vectorised processing over all t ---
        h_bt = h.reshape(BT, nx, d)

        # --- expand u0 to [BT, nx] ---
        u0_bt = u0.unsqueeze(1).expand(B, nt, nx).reshape(BT, nx)

        # --- build left/right states at each interface i+1/2 ---
        # h_L[i] = h[i], h_R[i] = h[i+1];  replicate pad at right boundary
        h_L = h_bt                                                        # [BT, nx, d]
        h_R = torch.cat([h_bt[:, 1:, :], h_bt[:, -1:, :]], dim=1)       # [BT, nx, d]

        # u0 interface quantities
        u0_L = u0_bt                                                      # [BT, nx]
        u0_R = torch.cat([u0_bt[:, 1:], u0_bt[:, -1:]], dim=1)          # [BT, nx]
        u0_avg  = ((u0_L + u0_R) / 2.0).unsqueeze(-1)                   # [BT, nx, 1]
        u0_jump = (u0_R - u0_L).unsqueeze(-1)                            # [BT, nx, 1]

        # --- compute antisymmetric numerical flux at each interface ---
        F_half = self._flux(h_L, h_R, u0_avg, u0_jump)                  # [BT, nx, d]

        # --- WENO smoothness weighting (on F values) ---
        if self.weno_p > 0.0:
            # Pad F_half: zero on the left (F_{-1/2}=0), replicate on the right
            F_pad = torch.cat(
                [torch.zeros_like(F_half[:, :1, :]), F_half, F_half[:, -1:, :]],
                dim=1,
            )  # [BT, nx+2, d]
            diff_bwd = F_pad[:, 1:-1, :] - F_pad[:, :-2, :]             # F_{i+1/2} - F_{i-1/2}
            diff_fwd = F_pad[:, 2:,   :] - F_pad[:, 1:-1, :]            # F_{i+3/2} - F_{i+1/2}
            # Use mean over channels to keep beta scale reasonable, then normalise per sample.
            beta = (diff_bwd ** 2 + diff_fwd ** 2).mean(dim=-1)         # [BT, nx]
            beta_scale = beta.median(dim=1, keepdim=True).values.detach().clamp(min=1e-6)
            beta = beta / beta_scale
            omega = 1.0 / (self.weno_eps + beta).pow(self.weno_p)       # [BT, nx]
            omega = omega.clamp(max=1.0)
            F_gated = omega.unsqueeze(-1) * F_half                      # [BT, nx, d]
            beta_map = beta.reshape(B, nt, nx)
        else:
            F_gated = F_half
            beta_map = h.new_zeros(B, nt, nx)

        # --- FVM divergence update: sp_agg = -alpha * (F_{i+1/2} - F_{i-1/2}) / dx ---
        # F_{i-1/2} for each cell i:
        F_left_gated = torch.cat(
            [torch.zeros_like(F_gated[:, :1, :]), F_gated[:, :-1, :]],
            dim=1,
        )  # [BT, nx, d]
        divergence = (F_gated - F_left_gated) / dx                      # [BT, nx, d]

        alpha = F.softplus(self.raw_alpha)
        sp_agg_bt = -alpha * divergence                                   # [BT, nx, d]

        sp_agg = sp_agg_bt.reshape(B, nt, nx, d)

        return sp_agg, beta_map


# --------------------------------------------------------------------------- #
# full space-time layer: flux spatial MP + causal temporal MP
# --------------------------------------------------------------------------- #
class _HypFluxMPLayer(nn.Module):
    """Space-time layer combining learned-flux spatial MP + causal temporal MP."""

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        k_t: int,
        activation: str,
        weno_eps: float = 1e-6,
        weno_p: float = 2.0,
        radius_t: float | None = None,
        causal_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.k_t = k_t
        self.radius_t = radius_t
        self.causal = causal_temporal

        act_map = {"gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}
        self.act = act_map.get(activation, nn.GELU)()

        # spatial: learned flux-divergence
        self.flux_sp = _FluxSpaceMPLayer(d_latent, d_hidden, activation, weno_eps, weno_p)

        # temporal: (h_i, h_j, t_i, t_j, dt, x/t) -> d_latent
        tp_in = 2 * d_latent + 4
        self.tp_msg = _make_mlp(tp_in, d_hidden, d_latent, 3, activation)

        # update + local linear: σ(K(v) + W·v)
        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    def forward(
        self,
        h: torch.Tensor,   # [B, nt, nx, d]
        x: torch.Tensor,   # [B, nx]
        t: torch.Tensor,   # [nt]
        u0: torch.Tensor,  # [B, nx]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_new [B, nt, nx, d], beta_map [B, nt, nx])."""
        B, nt, nx, d = h.shape

        # ---- spatial: flux divergence (parallel over all t) ----
        sp_agg, beta_map = self.flux_sp(h, x, t, u0)                    # [B, nt, nx, d]

        # ---- temporal: causal message passing ----
        if self.radius_t is not None:
            dt_val = (t[1] - t[0]).abs().item()
            k_t = max(1, int(self.radius_t / dt_val + 0.5))
        else:
            k_t = self.k_t

        # pad along t dimension: reshape to [B*nx, nt, d] for F.pad
        h_flat_t = h.permute(0, 2, 1, 3).reshape(B * nx, nt, d).permute(0, 2, 1)
        h_tp = F.pad(h_flat_t, (k_t, k_t), mode="replicate")
        h_tp = h_tp.permute(0, 2, 1).reshape(B, nx, nt + 2 * k_t, d)
        h_tp = h_tp.permute(0, 2, 1, 3)                                 # [B, nt+2k_t, nx, d]

        t_pad = F.pad(
            t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate"
        ).squeeze(0).squeeze(0)

        t_range = range(-k_t, 1) if self.causal else range(-k_t, k_t + 1)

        # self-similarity variable x/t (clamp t to avoid division by zero at t=0)
        t_i_abs = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)
        x_over_t = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1) / t_i_abs.clamp(min=1e-6)

        tp_agg = h.new_zeros(B, nt, nx, d)
        for j in t_range:
            h_j = h_tp[:, k_t + j : k_t + j + nt, :, :]
            t_j = t_pad[k_t + j : k_t + j + nt]
            t_j_abs = t_j.view(1, nt, 1, 1).expand(B, nt, nx, 1)
            rel_t = (t_j - t).view(1, nt, 1, 1).expand(B, nt, nx, 1)

            msg_in = torch.cat([h, h_j, t_i_abs, t_j_abs, rel_t, x_over_t], dim=-1)
            msg = self.tp_msg(msg_in)

            if self.radius_t is not None:
                t_mask = (rel_t.abs() <= self.radius_t)
                msg = msg * t_mask

            tp_agg = tp_agg + msg

        # ---- combine: σ(K(v) + W·v) ----
        upd_in = torch.cat([h, sp_agg + tp_agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)

        return self.act(h_nonlocal + h_local), beta_map


# --------------------------------------------------------------------------- #
# main model
# --------------------------------------------------------------------------- #
class HypFluxNO(nn.Module):
    """Hyperbolic Neural Operator with learned numerical flux spatial messages.

    Space-time operator: takes (u0 [B,nx], x [nx], t [nt]) and returns
    u_pred [B,nt,nx] for all (x,t) simultaneously in a single forward pass.

    Architecture: P -> time_mix -> [_HypFluxMPLayer]^n_layers -> Q(h) + u0
      - Spatial MP: FVM flux-divergence in latent space (antisymmetric flux MLP)
      - Temporal MP: causal message passing over past time steps
      - WENO beta on flux values gates updates near shocks
    """

    def __init__(
        self,
        stencil_k_t: int = 2,
        d_latent: int = 64,
        d_hidden: int = 64,
        d_time: int = 32,
        n_layers: int = 4,
        activation: str = "gelu",
        weno_eps: float = 1e-6,
        weno_p: float = 2.0,
        radius_t: float | None = None,
        causal_temporal: bool = True,
        stencil_k_x: int = 3,        # only used by LiftingLayer
        radius_x: float | None = None,
        readout: str = "gelu",
    ) -> None:
        super().__init__()

        # P: lifting (spatial neighbour aggregation on u0)
        self.lifting = _LiftingLayer(d_latent, d_hidden, stencil_k_x, activation, radius_x=radius_x)

        # Fourier time embedding
        n_freq = d_time // 2
        self.time_freqs = nn.Parameter(torch.randn(n_freq) * 0.1)
        self.time_proj = nn.Linear(d_time, d_time)

        # project (lifted_spatial || time_embedding) -> d_latent
        self.time_mix = _make_mlp(d_latent + d_time, d_hidden, d_latent, 2, activation)

        # space-time MP layers
        self.mp_layers = nn.ModuleList([
            _HypFluxMPLayer(
                d_latent=d_latent,
                d_hidden=d_hidden,
                k_t=stencil_k_t,
                activation=activation,
                weno_eps=weno_eps,
                weno_p=weno_p,
                radius_t=radius_t,
                causal_temporal=causal_temporal,
            )
            for _ in range(n_layers)
        ])

        # Q: decoder (readout activation can differ from internal activation)
        self.decoder = _make_mlp(d_latent, d_hidden, 1, 3, readout)

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        phase = t.unsqueeze(-1) * self.time_freqs.unsqueeze(0)
        return self.time_proj(torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1))

    def forward(
        self,
        u0: torch.Tensor,  # [B, nx]
        x: torch.Tensor,   # [nx] or [B, nx]
        t: torch.Tensor,   # [nt]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            u_pred  : [B, nt, nx]   predicted solution for all (x, t)
            beta_map: [B, nt, nx]   WENO flux smoothness indicator from last layer
        """
        B, nx = u0.shape
        nt = t.shape[0]

        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)

        # --- P: lifting (spatial, on u0) ---
        h_spatial = self.lifting(u0, x)                                  # [B, nx, d]

        # --- time embedding ---
        tau = self._time_embed(t)                                        # [nt, d_time]

        # --- tile + time mix ---
        h_exp = h_spatial.unsqueeze(1).expand(B, nt, nx, -1)
        tau_exp = tau.unsqueeze(0).unsqueeze(2).expand(B, nt, nx, -1)
        h = self.time_mix(torch.cat([h_exp, tau_exp], dim=-1))          # [B, nt, nx, d]

        # --- space-time MP ---
        beta_map = h.new_zeros(B, nt, nx)
        for layer in self.mp_layers:
            h, beta_map = layer(h, x, t, u0)

        # --- Q: decoder with u0 skip ---
        u0_exp = u0.unsqueeze(1).expand(B, nt, nx)
        u_pred = self.decoder(h).squeeze(-1) + u0_exp                   # [B, nt, nx]

        return u_pred, beta_map
