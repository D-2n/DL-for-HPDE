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
# space-time MP layer with WENO smoothness-weighted messages + causal temporal
# --------------------------------------------------------------------------- #
class _WENOSpaceTimeMPLayer(nn.Module):
    """Factored space-time MP with WENO-inspired smoothness weighting.

    Instead of a learned or PINN-based shock detector, uses the local
    smoothness of the latent field to weight messages.  Near discontinuities
    the latent field has large jumps → high β → small ω → messages are
    suppressed.  In smooth regions β ≈ 0 → ω is large → full message
    strength.  Computed per-layer so it adapts as the representation evolves.

    WENO weight:  ω_j = 1 / (ε + β_j)^p
    where β_j = ||h_{j+1} - h_j||² + ||h_{j-1} - h_j||²  (spatial)
          β_j = ||h_{t+1} - h_t||² + ||h_{t-1} - h_t||²  (temporal)

    When unified_mp=True, a single MLP handles both spatial and temporal
    messages.  Edge features are encoded as:
      (h_i, h_j, pos_i, pos_j, rel_pos, feat1, feat2, is_spatial)
    Spatial:  pos=x, feat1=du0, feat2=|du0|, is_spatial=1
    Temporal: pos=t, feat1=x/t, feat2=0,     is_spatial=0
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        k_x: int,
        k_t: int,
        activation: str,
        weno_eps: float = 1e-6,
        weno_p: float = 2.0,
        radius_x: float | None = None,
        radius_t: float | None = None,
        causal_temporal: bool = True,
        unified_mp: bool = False,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_t = k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.weno_eps = weno_eps
        self.weno_p = weno_p
        self.causal = causal_temporal
        self.unified_mp = unified_mp
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()

        if unified_mp:
            # unified message: (h_i, h_j, pos_i, pos_j, rel_pos, feat1, feat2, is_spatial)
            uni_in = 2 * d_latent + 6
            self.uni_msg = _make_mlp(uni_in, d_hidden, d_latent, 3, activation)
        else:
            # spatial message: (h_i, h_j, x_i, x_j, dx, du0, |du0|) -> d_latent
            sp_in = 2 * d_latent + 5
            self.sp_msg = _make_mlp(sp_in, d_hidden, d_latent, 3, activation)

            # temporal message: (h_i, h_j, t_i, t_j, dt, x_i/t_i) -> d_latent
            tp_in = 2 * d_latent + 4
            self.tp_msg = _make_mlp(tp_in, d_hidden, d_latent, 3, activation)

        # update + local linear (same σ(K(v) + W·v) structure)
        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    @staticmethod
    def _spatial_beta(h: torch.Tensor) -> torch.Tensor:
        """Smoothness indicator along spatial dim.  h: [B, nt, nx, d] -> [B, nt, nx]."""
        # forward and backward differences, replicate at boundaries
        diff_fwd = F.pad(h[:, :, 1:, :] - h[:, :, :-1, :], (0, 0, 0, 1))   # [B,nt,nx,d]
        diff_bwd = F.pad(h[:, :, :-1, :] - h[:, :, 1:, :], (0, 0, 1, 0))   # [B,nt,nx,d]
        beta = (diff_fwd ** 2 + diff_bwd ** 2).sum(dim=-1)                   # [B,nt,nx]
        return beta

    @staticmethod
    def _temporal_beta(h: torch.Tensor) -> torch.Tensor:
        """Smoothness indicator along temporal dim.  h: [B, nt, nx, d] -> [B, nt, nx]."""
        diff_fwd = F.pad(h[:, 1:, :, :] - h[:, :-1, :, :], (0, 0, 0, 0, 0, 1))
        diff_bwd = F.pad(h[:, :-1, :, :] - h[:, 1:, :, :], (0, 0, 0, 0, 1, 0))
        beta = (diff_fwd ** 2 + diff_bwd ** 2).sum(dim=-1)
        return beta

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        u0: torch.Tensor,
        shock_indicator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        h:                [B, nt, nx, d]
        x:                [B, nx]
        t:                [nt]
        u0:               [B, nx]
        shock_indicator:  [B, nt, nx] optional — from external detector.
                          When provided, messages are weighted by (1 - indicator)
                          instead of WENO smoothness.
        """
        B, nt, nx, d = h.shape
        use_external = shock_indicator is not None

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

        use_weno = self.weno_p > 0 and not use_external

        # external detector: precompute smooth-region weight (1 - indicator)
        if use_external:
            # shock_indicator: [B, nt, nx] in [0, 1], 1 = shock, 0 = smooth
            ext_weight = (1.0 - shock_indicator).unsqueeze(-1)  # [B, nt, nx, 1]

        # ---- spatial message passing (along x, for each t) ----
        h_flat = h.reshape(B * nt, nx, d).permute(0, 2, 1)
        h_xp = F.pad(h_flat, (k_x, k_x), mode="replicate")
        h_xp = h_xp.permute(0, 2, 1).reshape(B, nt, nx + 2 * k_x, d)

        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        u0_pad = F.pad(u0.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)

        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)

        if use_weno:
            beta_x = self._spatial_beta(h)
            beta_x_pad = F.pad(beta_x, (k_x, k_x), mode="replicate")
            sp_omega_raw = []
            for j in range(-k_x, k_x + 1):
                beta_j = beta_x_pad[:, :, k_x + j : k_x + j + nx]
                omega_j = 1.0 / (self.weno_eps + beta_j).pow(self.weno_p)
                if self.radius_x is not None:
                    x_j_val = x_pad[:, k_x + j : k_x + j + nx]
                    rel_x_j = (x_j_val - x[:, :nx]).abs()
                    r_mask = (rel_x_j <= self.radius_x).unsqueeze(1).expand_as(omega_j)
                    omega_j = omega_j * r_mask
                sp_omega_raw.append(omega_j)
            sp_omega_sum = torch.stack(sp_omega_raw, dim=0).sum(dim=0).clamp(min=1e-8)

        sp_agg = h.new_zeros(B, nt, nx, d)
        for idx, j in enumerate(range(-k_x, k_x + 1)):
            h_j = h_xp[:, :, k_x + j : k_x + j + nx, :]
            x_j_val = x_pad[:, k_x + j : k_x + j + nx].unsqueeze(1).unsqueeze(-1)
            rel_x = (x_j_val - x_i)
            du0 = (u0 - u0_pad[:, k_x + j : k_x + j + nx]).unsqueeze(1).unsqueeze(-1)
            abs_du0 = du0.abs()
            x_j_val = x_j_val.expand_as(h[:, :, :, :1])
            x_i_exp = x_i.expand_as(x_j_val)
            rel_x = rel_x.expand_as(x_j_val)
            du0 = du0.expand_as(rel_x)
            abs_du0 = abs_du0.expand_as(rel_x)

            if self.unified_mp:
                is_sp = h.new_ones(B, nt, nx, 1)
                msg_in = torch.cat([h, h_j, x_i_exp, x_j_val, rel_x, du0, abs_du0, is_sp], dim=-1)
                msg = self.uni_msg(msg_in)
            else:
                msg_in = torch.cat([h, h_j, x_i_exp, x_j_val, rel_x, du0, abs_du0], dim=-1)
                msg = self.sp_msg(msg_in)

            if use_external:
                contrib = msg * ext_weight
                if self.radius_x is not None:
                    r_mask = (rel_x.abs() <= self.radius_x)
                    contrib = contrib * r_mask
            elif use_weno:
                omega_norm = (sp_omega_raw[idx] / sp_omega_sum).unsqueeze(-1)
                contrib = msg * omega_norm
            else:
                contrib = msg
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

        t_range = range(-k_t, 1) if self.causal else range(-k_t, k_t + 1)

        t_i_abs = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)
        x_over_t = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1) / t_i_abs.clamp(min=1e-6)

        if use_weno:
            beta_t = self._temporal_beta(h)
            beta_t_perm = beta_t.permute(0, 2, 1)
            beta_t_pad = F.pad(beta_t_perm, (k_t, k_t), mode="replicate")
            beta_t_pad = beta_t_pad.permute(0, 2, 1)
            tp_omega_raw = []
            for j in t_range:
                beta_j = beta_t_pad[:, k_t + j : k_t + j + nt, :]
                omega_j = 1.0 / (self.weno_eps + beta_j).pow(self.weno_p)
                if self.radius_t is not None:
                    t_j = t_pad[k_t + j : k_t + j + nt]
                    rel_t_j = (t_j - t).abs()
                    t_mask = (rel_t_j <= self.radius_t).view(1, nt, 1).expand_as(omega_j)
                    omega_j = omega_j * t_mask
                tp_omega_raw.append(omega_j)
            tp_omega_sum = torch.stack(tp_omega_raw, dim=0).sum(dim=0).clamp(min=1e-8)

        tp_agg = h.new_zeros(B, nt, nx, d)
        for idx, j in enumerate(t_range):
            h_j = h_tp[:, k_t + j : k_t + j + nt, :, :]
            t_j = t_pad[k_t + j : k_t + j + nt]
            t_j_abs = t_j.view(1, nt, 1, 1).expand(B, nt, nx, 1)
            rel_t = (t_j - t).view(1, nt, 1, 1).expand(B, nt, nx, 1)

            if self.unified_mp:
                zeros = h.new_zeros(B, nt, nx, 1)
                is_sp = zeros  # is_spatial = 0
                msg_in = torch.cat([h, h_j, t_i_abs, t_j_abs, rel_t, x_over_t, zeros, is_sp], dim=-1)
                msg = self.uni_msg(msg_in)
            else:
                msg_in = torch.cat([h, h_j, t_i_abs, t_j_abs, rel_t, x_over_t], dim=-1)
                msg = self.tp_msg(msg_in)

            if use_external:
                contrib = msg * ext_weight
                if self.radius_t is not None:
                    t_mask = (rel_t.abs() <= self.radius_t)
                    contrib = contrib * t_mask
            elif use_weno:
                omega_norm = (tp_omega_raw[idx] / tp_omega_sum).unsqueeze(-1)
                contrib = msg * omega_norm
            else:
                contrib = msg
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
    """Hyperbolic Neural Operator with configurable shock handling.

    shock_mode controls how discontinuities are handled in message passing:
      - "pinn": PINN coarse decoder → PDE residual → binary shock mask →
                clamp messages to [0, delta] in shock regions.
                Computed once before MP (static).
      - "weno": WENO smoothness indicator on latent field →
                weight messages by 1/(ε + β)^p.
                Recomputed at each MP layer (adaptive).

    Common architecture: P -> [σ(K(v) + W·v)]^T -> Q
    Physics-enriched edge features:
      - Spatial: (h_i, h_j, x_i, x_j, dx, du0, |du0|)
      - Temporal: (h_i, h_j, t_i, t_j, dt, x/t)
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
        shock_mode: str = "pinn",
        weno_eps: float = 1e-6,
        weno_p: float = 2.0,
        unified_mp: bool = False,
        detector_path: str | None = None,
        detector_cfg: dict | None = None,
        readout: str = "gelu",
    ) -> None:
        super().__init__()
        self.stencil_k_x = stencil_k_x
        self.stencil_k_t = stencil_k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.shock_mode = shock_mode
        self.unified_mp = unified_mp
        self.has_external_detector = False

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

        # PINN shock detector (only created for pinn mode, but kept as optional
        # so old checkpoints can still load)
        if shock_mode == "pinn":
            self.shock_detector = _ShockDetectorPINN(d_latent, d_hidden, activation)
        else:
            self.shock_detector = None

        # space-time MP layers
        if shock_mode == "weno":
            self.mp_layers = nn.ModuleList([
                _WENOSpaceTimeMPLayer(
                    d_latent, d_hidden, stencil_k_x, stencil_k_t, activation,
                    weno_eps=weno_eps,
                    weno_p=weno_p,
                    radius_x=radius_x,
                    radius_t=radius_t,
                    causal_temporal=causal_temporal,
                    unified_mp=unified_mp,
                )
                for _ in range(n_layers)
            ])
        else:
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

        # Q: decoder (readout activation can differ from internal activation)
        self.decoder = _make_mlp(d_latent, d_hidden, 1, 3, readout)

        # external pre-trained PINN shock detector (frozen)
        if detector_path is not None:
            from hyperbolic_pde.models.shock_detector import ShockDetector
            det_cfg = detector_cfg or {}
            self.external_detector = ShockDetector(
                d_latent=int(det_cfg.get("d_latent", 64)),
                d_hidden=int(det_cfg.get("d_hidden", 128)),
                n_layers=int(det_cfg.get("n_layers", 6)),
                activation=str(det_cfg.get("activation", "tanh")),
                ic_points=int(det_cfg.get("ic_points", 128)),
            )
            ckpt = torch.load(detector_path, map_location="cpu", weights_only=True)
            self.external_detector.load_state_dict(ckpt)
            self.external_detector.requires_grad_(False)
            self.external_detector.eval()
            self.has_external_detector = True

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
            u_coarse:        [B, nt, nx]  PINN coarse prediction (zeros if weno)
            shock_indicator: [B, nt, nx]  shock indicator (WENO spatial beta if weno)
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

        # --- external detector (frozen, if available) ---
        ext_indicator = None
        if self.has_external_detector:
            self.external_detector.eval()
            # enable_grad needed because compute_shock_indicator_grid uses autograd
            # internally, but detector params are frozen so no gradients flow back
            with torch.enable_grad():
                ext_indicator, _ = self.external_detector(u0, x[0] if x.dim() > 1 else x, t)
            ext_indicator = ext_indicator.detach()

        if self.shock_mode == "pinn":
            # --- PINN shock detection ---
            dx_val = (x[0, 1] - x[0, 0]).abs().item()
            dt_val = (t[1] - t[0]).abs().item()
            shock_indicator, u_coarse = self.shock_detector(h, dx_val, dt_val)
            shock_indicator_detached = shock_indicator.detach()
            # prefer external detector if available
            si_for_mp = ext_indicator if ext_indicator is not None else shock_indicator_detached

            for layer in self.mp_layers:
                h = layer(h, x, t, u0, si_for_mp)
        else:
            # --- WENO mode ---
            u_coarse = torch.zeros(B, nt, nx, device=h.device)

            for layer in self.mp_layers:
                h = layer(h, x, t, u0, shock_indicator=ext_indicator)

            # return shock indicator for visualisation
            if ext_indicator is not None:
                shock_indicator = ext_indicator
            else:
                shock_indicator = _WENOSpaceTimeMPLayer._spatial_beta(h)
                si_max = shock_indicator.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                shock_indicator = shock_indicator / si_max

        # --- Q: decoder with skip from u0 ---
        correction = self.decoder(h).squeeze(-1)                         # [B, nt, nx]
        u0_exp = u0.unsqueeze(1).expand(B, nt, nx)
        u_pred = u0_exp + correction

        return u_pred, u_coarse, shock_indicator
