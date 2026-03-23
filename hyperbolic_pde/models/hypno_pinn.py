from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class _LiftingLayer(nn.Module):
    """Mesh-invariant spatial encoder with neighbor aggregation on u0.

    Keeps the learned gate from HypNO — the lifting layer operates on u0 only
    (no time), so the PINN shock detector (which needs space-time) can't help here.
    The u0-difference gate is already physics-grounded at this stage.
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
# PINN shock detector
# --------------------------------------------------------------------------- #
class _ShockDetectorPINN(nn.Module):
    """PINN-based shock detector.

    Produces a coarse solution from lifted space-time features, then computes
    the LWR PDE residual  R = du/dt + d[u(1-u)]/dx  via finite differences.
    High |R| indicates a shock neighbourhood.

    Returns a normalised shock indicator in [0, 1] and the coarse prediction
    (which is trained via an auxiliary L1 loss during training).
    """

    def __init__(self, d_latent: int, d_hidden: int, activation: str) -> None:
        super().__init__()
        self.coarse_decoder = _make_mlp(d_latent, d_hidden, 1, 2, activation)

    def forward(
        self, h: torch.Tensor, dx: float, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """h: [B, nt, nx, d] -> (shock_indicator [B,nt,nx], u_coarse [B,nt,nx])."""
        u_coarse = self.coarse_decoder(h).squeeze(-1)          # [B, nt, nx]

        # LWR flux f(u) = u(1 - u)
        f = u_coarse * (1.0 - u_coarse)

        # df/dx — central differences, one-sided at boundaries
        df_dx = torch.zeros_like(u_coarse)
        df_dx[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dx)
        df_dx[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dx
        df_dx[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dx

        # du/dt — central differences, one-sided at boundaries
        du_dt = torch.zeros_like(u_coarse)
        du_dt[:, 1:-1, :] = (u_coarse[:, 2:, :] - u_coarse[:, :-2, :]) / (2.0 * dt)
        du_dt[:, 0, :] = (u_coarse[:, 1, :] - u_coarse[:, 0, :]) / dt
        du_dt[:, -1, :] = (u_coarse[:, -1, :] - u_coarse[:, -2, :]) / dt

        # PDE residual — should be ~0 away from shocks
        residual = (du_dt + df_dx).abs()

        # normalise to [0, 1] per sample
        r_max = residual.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        shock_indicator = residual / r_max

        return shock_indicator, u_coarse


# --------------------------------------------------------------------------- #
# space-time message-passing layer with PINN shock capping + causal temporal
# --------------------------------------------------------------------------- #
class _PINNSpaceTimeMPLayer(nn.Module):
    """Factored space-time MP with physics-based message capping.

    Differences from HypNO's _SpaceTimeMPLayer:
      1. No learned gate MLP — shock detection is provided by the PINN.
      2. In shock neighbourhood: messages clamped to [0, delta].
      3. Outside shock neighbourhood: messages pass through normally.
      4. Temporal MP is causal — only past time steps contribute.
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        k_x: int,
        k_t: int,
        activation: str,
        shock_delta: float = 0.01,
        shock_threshold: float = 0.1,
        radius_x: float | None = None,
        radius_t: float | None = None,
        causal_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_t = k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.delta = shock_delta
        self.threshold = shock_threshold
        self.causal = causal_temporal
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()

        # spatial message: (h_i, h_j, x_i, x_j, dx, du0, |du0|) -> d_latent
        sp_in = 2 * d_latent + 5
        self.sp_msg = _make_mlp(sp_in, d_hidden, d_latent, 3, activation)

        # temporal message: (h_i, h_j, t_i, t_j, dt, x_i/t_i) -> d_latent
        tp_in = 2 * d_latent + 4
        self.tp_msg = _make_mlp(tp_in, d_hidden, d_latent, 3, activation)

        # update + local linear (same σ(K(v) + W·v) structure)
        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        u0: torch.Tensor,
        shock_indicator: torch.Tensor,
    ) -> torch.Tensor:
        """
        h:                [B, nt, nx, d]
        x:                [B, nx]
        t:                [nt]
        u0:               [B, nx]
        shock_indicator:  [B, nt, nx]   (0 = smooth, 1 = strong shock)
        """
        B, nt, nx, d = h.shape

        if self.radius_x is not None:
            dx_val = (x[0, 1] - x[0, 0]).abs().item()
            k_x = max(1, int(self.radius_x / dx_val + 0.5))
        else:
            k_x = self.k_x
        if self.radius_t is not None:
            dt_val = (t[1] - t[0]).abs().item()
            k_t = max(1, int(self.radius_t / dt_val + 0.5))
        else:
            k_t = self.k_t

        # shock mask at receiver: [B, nt, nx, 1]
        is_shock = (shock_indicator > self.threshold).unsqueeze(-1)

        # ---- spatial message passing (along x, for each t) ----
        h_flat = h.reshape(B * nt, nx, d).permute(0, 2, 1)
        h_xp = F.pad(h_flat, (k_x, k_x), mode="replicate")
        h_xp = h_xp.permute(0, 2, 1).reshape(B, nt, nx + 2 * k_x, d)

        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        u0_pad = F.pad(u0.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)

        # absolute positions for edge features: [B, nx] -> [B, 1, nx, 1]
        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)

        sp_agg = h.new_zeros(B, nt, nx, d)
        for j in range(-k_x, k_x + 1):
            h_j = h_xp[:, :, k_x + j : k_x + j + nx, :]
            x_j_val = x_pad[:, k_x + j : k_x + j + nx].unsqueeze(1).unsqueeze(-1)
            rel_x = (x_j_val - x_i)                                         # dx = x_j - x_i
            du0 = (u0 - u0_pad[:, k_x + j : k_x + j + nx]).unsqueeze(1).unsqueeze(-1)
            abs_du0 = du0.abs()
            x_j_val = x_j_val.expand_as(h[:, :, :, :1])
            x_i_exp = x_i.expand_as(x_j_val)
            rel_x = rel_x.expand_as(x_j_val)
            du0 = du0.expand_as(rel_x)
            abs_du0 = abs_du0.expand_as(rel_x)

            msg_in = torch.cat([h, h_j, x_i_exp, x_j_val, rel_x, du0, abs_du0], dim=-1)
            msg = self.sp_msg(msg_in)

            # physics-based capping: [0, delta] in shock neighbourhood, normal otherwise
            msg_capped = msg.clamp(0.0, self.delta)
            contrib = torch.where(is_shock, msg_capped, msg)

            if self.radius_x is not None:
                r_mask = (rel_x.abs() <= self.radius_x)
                contrib = contrib * r_mask
            sp_agg = sp_agg + contrib

        # ---- temporal message passing (causal: past only) ----
        h_flat_t = h.permute(0, 2, 1, 3).reshape(B * nx, nt, d).permute(0, 2, 1)
        h_tp = F.pad(h_flat_t, (k_t, k_t), mode="replicate")
        h_tp = h_tp.permute(0, 2, 1).reshape(B, nx, nt + 2 * k_t, d)
        h_tp = h_tp.permute(0, 2, 1, 3)

        t_pad = F.pad(
            t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate"
        ).squeeze(0).squeeze(0)

        # causal: only j <= 0 (past and current time)
        t_range = range(-k_t, 1) if self.causal else range(-k_t, k_t + 1)

        # absolute time + self-similarity variable x/t
        t_i_abs = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)                  # [B, nt, nx, 1]
        # x/t: self-similarity variable (for Riemann problems, solution is const along x/t = const)
        # clamp t to avoid division by zero at t=0
        x_over_t = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1) / t_i_abs.clamp(min=1e-6)

        tp_agg = h.new_zeros(B, nt, nx, d)
        for j in t_range:
            h_j = h_tp[:, k_t + j : k_t + j + nt, :, :]
            t_j = t_pad[k_t + j : k_t + j + nt]
            t_j_abs = t_j.view(1, nt, 1, 1).expand(B, nt, nx, 1)
            rel_t = (t_j - t).view(1, nt, 1, 1).expand(B, nt, nx, 1)
            msg_in = torch.cat([h, h_j, t_i_abs, t_j_abs, rel_t, x_over_t], dim=-1)
            msg = self.tp_msg(msg_in)

            # same physics-based capping in temporal direction
            msg_capped = msg.clamp(0.0, self.delta)
            contrib = torch.where(is_shock, msg_capped, msg)

            if self.radius_t is not None:
                t_mask = (rel_t.abs() <= self.radius_t)
                contrib = contrib * t_mask
            tp_agg = tp_agg + contrib

        # ---- combine: σ(K(v) + W·v) ----
        upd_in = torch.cat([h, sp_agg + tp_agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)

        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# main model
# --------------------------------------------------------------------------- #
class HypNO_PINN(nn.Module):
    """Hyperbolic Neural Operator with PINN-based shock detection.

    Key differences from HypNO:
      1. A small PINN produces a coarse prediction whose PDE residual
         reveals shock neighbourhoods — replaces learned gating MLPs.
      2. In the shock neighbourhood, message strength is clamped to
         [0, delta] with delta very small — physics-informed, not black-box.
      3. Temporal message passing is causal (past only).
      4. Physics-enriched edge features:
         - Spatial: (h_i, h_j, x_i, x_j, dx, du0, |du0|)
         - Temporal: (h_i, h_j, t_i, t_j, dt, x/t)
         where x/t is the self-similarity variable for Riemann problems.

    Architecture: P -> shock detect -> [σ(K(v) + W·v)]^T -> Q
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
        shock_delta: float = 0.01,
        shock_threshold: float = 0.1,
        causal_temporal: bool = True,
        radius_x: float | None = None,
        radius_t: float | None = None,
    ) -> None:
        super().__init__()
        self.stencil_k_x = stencil_k_x
        self.stencil_k_t = stencil_k_t
        self.radius_x = radius_x
        self.radius_t = radius_t

        # P: lifting (spatial neighbour aggregation on u0)
        self.lifting = _LiftingLayer(
            d_latent, d_hidden, stencil_k_x, activation, radius_x=radius_x
        )

        # Fourier time embedding
        n_freq = d_time // 2
        self.time_freqs = nn.Parameter(torch.randn(n_freq) * 0.1)
        self.time_proj = nn.Linear(d_time, d_time)

        # project (lifted_spatial || time_embedding) -> d_latent
        self.time_mix = _make_mlp(d_latent + d_time, d_hidden, d_latent, 2, activation)

        # PINN shock detector
        self.shock_detector = _ShockDetectorPINN(d_latent, d_hidden, activation)

        # space-time MP layers with PINN-based capping
        self.mp_layers = nn.ModuleList([
            _PINNSpaceTimeMPLayer(
                d_latent, d_hidden, stencil_k_x, stencil_k_t, activation,
                shock_delta=shock_delta,
                shock_threshold=shock_threshold,
                radius_x=radius_x,
                radius_t=radius_t,
                causal_temporal=causal_temporal,
            )
            for _ in range(n_layers)
        ])

        # Q: decoder
        self.decoder = _make_mlp(d_latent, d_hidden, 1, 3, activation)

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        phase = t.unsqueeze(-1) * self.time_freqs.unsqueeze(0)
        return self.time_proj(torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1))

    def forward(
        self,
        u0: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        u0: [B, nx]   initial condition
        x:  [nx]      spatial coordinates
        t:  [nt]      time coordinates
        Returns:
            u_pred:          [B, nt, nx]  main prediction
            u_coarse:        [B, nt, nx]  PINN coarse prediction (for auxiliary loss)
            shock_indicator: [B, nt, nx]  normalised shock indicator (for visualisation)
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
        h = self.time_mix(torch.cat([h_exp, tau_exp], dim=-1))           # [B, nt, nx, d]

        # --- PINN shock detection ---
        dx_val = (x[0, 1] - x[0, 0]).abs().item()
        dt_val = (t[1] - t[0]).abs().item()
        shock_indicator, u_coarse = self.shock_detector(h, dx_val, dt_val)
        # detach: shock mask is a fixed signal — gradients to the coarse decoder
        # flow only through the auxiliary loss, not through the main loss
        shock_indicator_detached = shock_indicator.detach()

        # --- space-time message passing ---
        for layer in self.mp_layers:
            h = layer(h, x, t, u0, shock_indicator_detached)

        # --- Q: decoder with skip from u0 ---
        correction = self.decoder(h).squeeze(-1)                         # [B, nt, nx]
        u0_exp = u0.unsqueeze(1).expand(B, nt, nx)
        u_pred = u0_exp + correction

        return u_pred, u_coarse, shock_indicator
