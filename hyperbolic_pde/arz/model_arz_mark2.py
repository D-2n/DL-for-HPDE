"""HypNO-ARZ Mark 2: two-wave-family space-time GNN operator (homogeneous ARZ).

Design (see writeup/hypno_arz_mark2.tex and the design memo):

* HOMOGENEOUS ARZ ONLY (tau = inf). No relaxation source, no Veq/diseq node
  features, no balance loss. `w` is a true 1-Riemann invariant for all time.

* Two characteristic wave families, separated AT THE MESSAGE:
    - 1-wave (GNL): forms shocks/rarefactions, preserves w. Carries the
      Rankine-Hugoniot SECANT speed s_1 = [rho*v]/[rho] (mirrors LWR a_ij),
      strength drho, and an entropy gate (Oleinik/Lax) on its branch only.
    - 2-contact (LD): advects at v, preserves v. Carries s_2 = v_ij and
      strength dw. No entropy, no secant (LD => secant == eigenvalue).

* Soft, parameter-free PHYSICS ROUTER theta = |dw|/(|dw|+|dv|) blends the two
  family messages per adjacent edge, re-decided every layer from the current
  decoded state. theta~0 -> GNL, theta~1 -> contact.

* Decoder outputs PRIMITIVE (rho, v), BOTH LINEAR (unconstrained, matching the
  best LWR-ST3 readout); recover w = v + p(rho) by exact algebra. The sigmoid on
  rho was removed -- it smeared shocks (squashing softens interior jumps into
  ramps). Global output skip default OFF.

* Upwind gate is a TOGGLE, default OFF (FVM upwinding is flux-splitting, not
  edge-pruning; the causal stencil already encodes the domain of dependence).

* Configurable width/depth: adj_depth / nonadj_depth / update_depth (depths),
  update_hidden (update width, null->d_hidden), d_hidden (adj/family + decoder),
  d_hidden_nonadj (nonadj width, null->d_hidden), decoder_depth.

Inputs:  rho0 [B,nx], w0 [B,nx], x [nx] or [B,nx], t [nt]
Outputs: rho [B,nt,nx], w [B,nt,nx], u_hats list (deep supervision, (rho,w))
  -- forward returns the SAME (rho_pred, w_pred, u_hats) signature as HypNO_ARZ
     so existing eval code works unchanged (w recovered internally).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from hyperbolic_pde.models.hypno_st3 import (
    _enumerate_ball_offsets,
    _make_mlp,
    _pad_space_time,
    _spatial_pad_width,
)
from hyperbolic_pde.arz import physics_arz as _P_MOD

_SECANT_EPS = 1e-6   # |drho| guard for the RH secant denominator
_THETA_EPS = 1e-8    # router denominator guard


def _p(rho):
    if _P_MOD._P_FORM == "rho":
        return rho
    return rho + rho * rho


def _dp(rho):
    if _P_MOD._P_FORM == "rho":
        return torch.ones_like(rho)
    return 1.0 + 2.0 * rho


# --------------------------------------------------------------------------- #
# shared edge-physics helper
# --------------------------------------------------------------------------- #
def _adj_interface_physics(rho_i, v_i, w_i, rho_j, v_j, w_j, rel_x):
    """Compute the per-family adjacent-edge physics scalars.

    Returns dict with: s_1 (RH secant), s_2 (= v_ij), drho, dw, theta,
    chi_1bad (entropy-violation flag), sign_rel_x.
    """
    drho = rho_j - rho_i
    dv = v_j - v_i
    dw = w_j - w_i
    # RH secant speed of the mass flux F1 = rho*v: s_1 = [rho*v]/[rho].
    # Falls back to the differential interface eigenvalue when |drho| is tiny.
    rho_ij = 0.5 * (rho_i + rho_j)
    v_ij = 0.5 * (v_i + v_j)
    lam1_ij = v_ij - rho_ij * _dp(rho_ij)
    num = rho_j * v_j - rho_i * v_i
    drho_safe = torch.where(drho.abs() < _SECANT_EPS, torch.ones_like(drho), drho)
    s_1 = torch.where(drho.abs() < _SECANT_EPS, lam1_ij, num / drho_safe)
    s_2 = v_ij  # contact speed (LD): secant == eigenvalue, v constant across it
    # Soft physics router.
    theta = dw.abs() / (dw.abs() + dv.abs() + _THETA_EPS)
    # Entropy (Lax) flag for the 1-wave: an admissible 1-shock satisfies the
    # bracket lam1(U_R) < s_1 < lam1(U_L). It is violated only by an expansion
    # shock, where the secant speed s_1 falls OUTSIDE that bracket. A rarefaction
    # (lam1_L < lam1_R, with s_1 between them) is admissible and is NOT flagged.
    # NOTE: the old test `lam1_L < lam1_R` mislabeled every rarefaction as bad.
    lam1_i = v_i - rho_i * _dp(rho_i)
    lam1_j = v_j - rho_j * _dp(rho_j)
    lam1_L = torch.where(rel_x > 0, lam1_i, lam1_j)
    lam1_R = torch.where(rel_x > 0, lam1_j, lam1_i)
    chi_1bad = ((s_1 > lam1_L) | (s_1 < lam1_R)).float()
    return {
        "s_1": s_1, "s_2": s_2, "drho": drho, "dw": dw,
        "theta": theta, "chi_1bad": chi_1bad,
        "sign_rel_x": torch.sign(rel_x),
    }


# --------------------------------------------------------------------------- #
# Lifting
# --------------------------------------------------------------------------- #
class _ArzLiftingM2(nn.Module):
    """Mark2 lifting. Node input (7): rho0, w0, v0, lam1_0, x, t, xi.

    Adjacent edges use the same family-split GNL/LD construction as the
    processor, theta-blended, with entropy on the GNL branch.
    Non-adjacent edges carry geometry only.
    """

    def __init__(
        self,
        d_latent: int, d_hidden: int,
        stencil_k_x: int, stencil_k_t: int,
        activation: str = "gelu",
        causal_temporal: bool = True,
        normalize_edge_offsets: bool = True,
        d_hidden_nonadj: Optional[int] = None,
        adj_depth: int = 3,
        nonadj_depth: int = 3,
        use_upwind_gate: bool = False,
        double_batch: bool = False,
        neighborhood_spacing: int = 1,
    ) -> None:
        super().__init__()
        self.k_x = stencil_k_x
        self.k_t = stencil_k_t
        self.causal = causal_temporal
        self.normalize_edge_offsets = normalize_edge_offsets
        self.use_upwind_gate = use_upwind_gate
        self.double_batch = double_batch
        self.neighborhood_spacing = neighborhood_spacing
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        self.node_mlp = _make_mlp(7, d_hidden, d_latent, 2, activation)
        # family edge MLPs: [h_i, h_j, speed, strength, sign] -> 2d+3
        self.msg_gnl = _make_mlp(2 * d_latent + 3, d_hidden, d_latent, adj_depth, activation)
        self.msg_ld  = _make_mlp(2 * d_latent + 3, d_hidden, d_latent, adj_depth, activation)
        self.nonadj_msg = _make_mlp(2 * d_latent + 3, dh_na, d_latent, nonadj_depth, activation)

        self.phys_temp = nn.Parameter(torch.tensor(0.0))    # upwind temperature (if on)
        self.phys_gamma = nn.Parameter(torch.tensor(-2.0))  # entropy gate strength

        self.combine = _make_mlp(2 * d_latent, d_hidden, d_latent, 2, activation)

    def _g_ent(self, chi_1bad):
        gamma = torch.sigmoid(self.phys_gamma)
        return 1.0 - (1.0 - gamma) * chi_1bad

    def _g_up(self, speed, sign_rel_x):
        tau = F.softplus(self.phys_temp).clamp(min=1e-6)
        return torch.sigmoid(-speed * sign_rel_x / tau)

    def forward(self, rho0, w0, x, t):
        B, nx = rho0.shape
        nt = t.shape[0]
        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)
        dx_grid = (x[0, 1] - x[0, 0]).abs().item()
        dt_grid = (t[1] - t[0]).abs().item()

        rho0_bc = rho0.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        w0_bc = w0.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        x_bc = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        t_bc = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)

        v0_bc = w0_bc - _p(rho0_bc)
        lam1_0 = v0_bc - rho0_bc * _dp(rho0_bc)
        eps_t = max(dt_grid, 1e-6)
        xi_bc = x_bc / t_bc.clamp(min=eps_t)

        node_in = torch.cat([rho0_bc, w0_bc, v0_bc, lam1_0, x_bc, t_bc, xi_bc], dim=-1)  # 7
        h_node = self.node_mlp(node_in)

        pad_x = _spatial_pad_width(
            self.k_x, dilated_spatial=False,
            double_batch=self.double_batch,
            neighborhood_spacing=self.neighborhood_spacing,
        )
        rho0_pad = F.pad(rho0.unsqueeze(1), (pad_x, pad_x), mode="replicate").squeeze(1)
        w0_pad   = F.pad(w0.unsqueeze(1),   (pad_x, pad_x), mode="replicate").squeeze(1)
        x_pad    = F.pad(x.unsqueeze(1),    (pad_x, pad_x), mode="replicate").squeeze(1)
        t_pad    = F.pad(t.view(1, 1, -1),  (self.k_t, self.k_t), mode="replicate").view(-1)

        offsets = _enumerate_ball_offsets(
            self.k_x, self.k_t, self.causal,
            double_batch=self.double_batch,
            neighborhood_spacing=self.neighborhood_spacing,
        )

        msgs: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []

        for di, dm in offsets:
            rho_j = rho0_pad[:, pad_x + di : pad_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            w_j   = w0_pad[:, pad_x + di : pad_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            x_j   = x_pad[:, pad_x + di : pad_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            t_j   = t_pad[self.k_t + dm : self.k_t + dm + nt].view(1, nt, 1, 1).expand(B, nt, nx, 1)
            rel_x = x_j - x_bc
            rel_t = t_j - t_bc
            v_j = w_j - _p(rho_j)

            if self.normalize_edge_offsets:
                rel_x_feat = rel_x / dx_grid
                rel_t_feat = rel_t / dt_grid
            else:
                rel_x_feat = rel_x
                rel_t_feat = rel_t

            is_adj_sp = (dm == 0) and (abs(di) == 1)
            sign_rel_x = torch.sign(rel_x)

            if is_adj_sp:
                ph = _adj_interface_physics(rho0_bc, v0_bc, w0_bc, rho_j, v_j, w_j, rel_x)
                gnl_in = torch.cat([h_node, h_node, ph["s_1"], ph["drho"], ph["sign_rel_x"]], dim=-1)
                ld_in  = torch.cat([h_node, h_node, ph["s_2"], ph["dw"],  ph["sign_rel_x"]], dim=-1)
                m_gnl = self.msg_gnl(gnl_in)
                m_ld  = self.msg_ld(ld_in)
                theta = ph["theta"]
                g_ent = self._g_ent(ph["chi_1bad"])
                if self.use_upwind_gate:
                    g_ent = g_ent * self._g_up(ph["s_1"], ph["sign_rel_x"])
                msg = (1.0 - theta) * g_ent * m_gnl + theta * m_ld
                gate = torch.ones_like(rel_x)
            else:
                geo_in = torch.cat([h_node, h_node, rel_x_feat, rel_t_feat, sign_rel_x], dim=-1)
                msg = self.nonadj_msg(geo_in)
                gate = torch.ones_like(rel_x)

            msgs.append(msg)
            gates.append(gate)

        gate_sum = torch.stack(gates, dim=-2).sum(dim=-2) + 1e-3
        all_msgs = torch.stack(msgs, dim=3)
        all_gates = torch.stack(gates, dim=3)
        agg = (all_gates / gate_sum.unsqueeze(3) * all_msgs).sum(dim=3)
        return self.combine(torch.cat([h_node, agg], dim=-1))


# --------------------------------------------------------------------------- #
# Message-passing layer
# --------------------------------------------------------------------------- #
class _ArzMPLayerM2(nn.Module):
    """Mark2 processor layer: family-split GNL/LD adjacent messages + geometric
    non-adjacent message, theta-blended, gated, learned residual update."""

    def __init__(
        self,
        d_latent: int, d_hidden: int,
        k_x: int, k_t: int,
        activation: str = "gelu",
        causal_temporal: bool = True,
        d_hidden_nonadj: Optional[int] = None,
        shared_decoder: Optional[nn.Module] = None,
        normalize_edge_offsets: bool = True,
        adj_depth: int = 3,
        nonadj_depth: int = 3,
        update_depth: int = 3,
        update_hidden: Optional[int] = None,
        use_upwind_gate: bool = False,
        double_batch: bool = False,
        neighborhood_spacing: int = 1,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_t = k_t
        self.causal = causal_temporal
        self.normalize_edge_offsets = normalize_edge_offsets
        self.use_upwind_gate = use_upwind_gate
        self.double_batch = double_batch
        self.neighborhood_spacing = neighborhood_spacing
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj
        upd_h = d_hidden if update_hidden is None else update_hidden

        if shared_decoder is None:
            raise ValueError("_ArzMPLayerM2 requires a shared_decoder")
        object.__setattr__(self, "_shared_decoder", shared_decoder)

        self.phys_temp = nn.Parameter(torch.tensor(0.0))
        self.phys_gamma = nn.Parameter(torch.tensor(-2.0))
        self.phys_cfl_scale = nn.Parameter(torch.tensor(0.0))

        self.msg_gnl = _make_mlp(2 * d_latent + 3, d_hidden, d_latent, adj_depth, activation)
        self.msg_ld  = _make_mlp(2 * d_latent + 3, d_hidden, d_latent, adj_depth, activation)
        self.nonadj_msg = _make_mlp(2 * d_latent + 3, dh_na, d_latent, nonadj_depth, activation)

        self.update_net = _make_mlp(2 * d_latent, upd_h, d_latent, update_depth, activation)
        self.W = nn.Linear(d_latent, d_latent)

    def _g_ent(self, chi_1bad):
        gamma = torch.sigmoid(self.phys_gamma)
        return 1.0 - (1.0 - gamma) * chi_1bad

    def _g_up(self, speed, sign_rel_x):
        tau = F.softplus(self.phys_temp).clamp(min=1e-6)
        return torch.sigmoid(-speed * sign_rel_x / tau)

    def _g_cfl(self, dm, rel_t, rel_x, spec_i):
        if dm == 0:
            return torch.ones_like(rel_t)
        cfl_scale = F.softplus(self.phys_cfl_scale).clamp(min=1e-6)
        dx_edge = rel_x.abs()
        cfl = torch.where(
            dx_edge > 0,
            spec_i * rel_t.abs() / dx_edge.clamp(min=1e-12),
            torch.zeros_like(rel_t),
        )
        return torch.exp(-cfl_scale * F.relu(cfl - 1.0) ** 2)

    def _decode_primitive(self, h):
        """Shared decoder -> (rho, v, w). rho and v both LINEAR (matching the
        best LWR-ST3 readout: GELU hidden activations, unconstrained output);
        w = v + p(rho). The sigmoid on rho was removed -- it squashed interior
        jumps into ramps (slope sigma*(1-sigma) is small away from 0.5), smearing
        shocks. The forward pass is division-safe for any rho (no 1/rho, no
        sqrt/log), so an unconstrained rho cannot blow up the physics features."""
        out = self._shared_decoder(h)            # [..., 2] = (d_rho, d_v)
        rho = out[..., 0:1]
        v = out[..., 1:2]
        w = v + _p(rho)
        return rho, v, w

    def forward(self, h, x, t, rho0, w0):
        B, nt, nx, d = h.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)
        dx_val = (x[0, 1] - x[0, 0]).abs().item()
        dt_val = (t[1] - t[0]).abs().item()

        rho_hat, v_hat, w_hat = self._decode_primitive(h)
        lam1_i = v_hat - rho_hat * _dp(rho_hat)
        lam2_i = v_hat
        spec_i = torch.maximum(lam1_i.abs(), lam2_i.abs())

        pad_x = _spatial_pad_width(
            self.k_x, dilated_spatial=False,
            double_batch=self.double_batch,
            neighborhood_spacing=self.neighborhood_spacing,
        )
        h_pad = _pad_space_time(h, pad_x, self.k_t)
        rho_pad = _pad_space_time(rho_hat, pad_x, self.k_t)
        v_pad = _pad_space_time(v_hat, pad_x, self.k_t)
        w_pad = _pad_space_time(w_hat, pad_x, self.k_t)
        x_pad = F.pad(x.unsqueeze(1), (pad_x, pad_x), mode="replicate").squeeze(1)
        t_pad = F.pad(t.view(1, 1, -1), (self.k_t, self.k_t), mode="replicate").view(-1)

        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        t_i = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)

        offsets = _enumerate_ball_offsets(
            self.k_x, self.k_t, self.causal,
            double_batch=self.double_batch,
            neighborhood_spacing=self.neighborhood_spacing,
        )

        msgs: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []

        for di, dm in offsets:
            h_j = h_pad[:, self.k_t + dm : self.k_t + dm + nt,
                            pad_x + di : pad_x + di + nx, :]
            rho_j = rho_pad[:, self.k_t + dm : self.k_t + dm + nt,
                                pad_x + di : pad_x + di + nx, :]
            v_j = v_pad[:, self.k_t + dm : self.k_t + dm + nt,
                            pad_x + di : pad_x + di + nx, :]
            w_j = w_pad[:, self.k_t + dm : self.k_t + dm + nt,
                            pad_x + di : pad_x + di + nx, :]
            x_j = x_pad[:, pad_x + di : pad_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            t_j = t_pad[self.k_t + dm : self.k_t + dm + nt].view(1, nt, 1, 1).expand(B, nt, nx, 1)
            rel_x = x_j - x_i
            rel_t = t_j - t_i

            if self.normalize_edge_offsets:
                rel_x_feat = rel_x / dx_val
                rel_t_feat = rel_t / dt_val
            else:
                rel_x_feat = rel_x
                rel_t_feat = rel_t

            is_adj_sp = (dm == 0) and (abs(di) == 1)
            sign_rel_x = torch.sign(rel_x)

            if is_adj_sp:
                ph = _adj_interface_physics(rho_hat, v_hat, w_hat, rho_j, v_j, w_j, rel_x)
                gnl_in = torch.cat([h, h_j, ph["s_1"], ph["drho"], ph["sign_rel_x"]], dim=-1)
                ld_in  = torch.cat([h, h_j, ph["s_2"], ph["dw"],  ph["sign_rel_x"]], dim=-1)
                m_gnl = self.msg_gnl(gnl_in)
                m_ld  = self.msg_ld(ld_in)
                theta = ph["theta"]
                g_ent = self._g_ent(ph["chi_1bad"])
                if self.use_upwind_gate:
                    g_ent = g_ent * self._g_up(ph["s_1"], ph["sign_rel_x"])
                msg = (1.0 - theta) * g_ent * m_gnl + theta * m_ld
                gate = torch.ones_like(rel_x)
            else:
                geo_in = torch.cat([h, h_j, rel_x_feat, rel_t_feat, sign_rel_x], dim=-1)
                msg = self.nonadj_msg(geo_in)
                gate = self._g_cfl(dm, rel_t, rel_x, spec_i)

            msgs.append(msg)
            gates.append(gate)

        gate_sum = torch.stack(gates, dim=-2).sum(dim=-2) + 1e-3
        all_msgs = torch.stack(msgs, dim=3)
        all_gates = torch.stack(gates, dim=3)
        agg = (all_gates / gate_sum.unsqueeze(3) * all_msgs).sum(dim=3)

        upd_in = torch.cat([h, agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)
        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class HypNO_ARZ_Mark2(nn.Module):
    """HypNO-ARZ Mark 2 operator (homogeneous ARZ, two-family).

    forward returns (rho_pred, w_pred, u_hats) -- same signature as HypNO_ARZ,
    so eval code is unchanged. The decoder is (rho, v); w is recovered.
    u_hats are (rho, w) per-layer readouts for deep supervision.
    """

    def __init__(
        self,
        stencil_k_x: int = 2,
        stencil_k_t: int = 2,
        d_latent: int = 96,
        d_hidden: int = 96,
        n_layers: int = 7,
        activation: str = "gelu",
        causal_temporal: bool = True,
        d_hidden_nonadj: Optional[int] = None,
        decoder_depth: int = 3,
        adj_depth: int = 3,
        nonadj_depth: int = 3,
        update_depth: int = 3,
        update_hidden: Optional[int] = None,
        skip: bool = False,
        use_upwind_gate: bool = False,
        use_checkpoint: bool = False,
        normalize_edge_offsets: bool = True,
        double_batch: bool = False,
        neighborhood_spacing: int = 1,
        **_ignored,
    ) -> None:
        super().__init__()
        self.skip = skip
        self.use_checkpoint = use_checkpoint
        if _ignored:
            print(f"[HypNO_ARZ_Mark2] IGNORED kwargs = {sorted(_ignored.keys())}")
        print(
            f"[HypNO_ARZ_Mark2] kx={stencil_k_x} kt={stencil_k_t} d_latent={d_latent} "
            f"d_hidden={d_hidden} layers={n_layers} adj_depth={adj_depth} "
            f"nonadj_depth={nonadj_depth} update_depth={update_depth} "
            f"update_hidden={update_hidden} skip={skip} "
            f"use_upwind_gate={use_upwind_gate}"
        )

        self.lifting = _ArzLiftingM2(
            d_latent, d_hidden,
            stencil_k_x=stencil_k_x, stencil_k_t=stencil_k_t,
            activation=activation, causal_temporal=causal_temporal,
            normalize_edge_offsets=normalize_edge_offsets,
            d_hidden_nonadj=d_hidden_nonadj,
            adj_depth=adj_depth, nonadj_depth=nonadj_depth,
            use_upwind_gate=use_upwind_gate,
            double_batch=double_batch, neighborhood_spacing=neighborhood_spacing,
        )

        # Decoder outputs (d_rho, d_v) -- primitive frame.
        self.decoder = _make_mlp(d_latent, d_hidden, 2, decoder_depth, activation)

        self.mp_layers = nn.ModuleList([
            _ArzMPLayerM2(
                d_latent, d_hidden,
                k_x=stencil_k_x, k_t=stencil_k_t,
                activation=activation, causal_temporal=causal_temporal,
                d_hidden_nonadj=d_hidden_nonadj,
                shared_decoder=self.decoder,
                normalize_edge_offsets=normalize_edge_offsets,
                adj_depth=adj_depth, nonadj_depth=nonadj_depth,
                update_depth=update_depth, update_hidden=update_hidden,
                use_upwind_gate=use_upwind_gate,
                double_batch=double_batch, neighborhood_spacing=neighborhood_spacing,
            )
            for _ in range(n_layers)
        ])

    def _decode(self, h, rho0, w0):
        out = self.decoder(h)                    # (d_rho, d_v)
        rho = out[..., 0:1]                      # LINEAR (sigmoid removed -- see _decode_primitive)
        v = out[..., 1:2]
        if self.skip:
            B, nt, nx, _ = out.shape
            v0 = (w0 - _p(rho0)).unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            v = v + v0
        w = v + _p(rho)
        return torch.cat([rho, v, w], dim=-1)    # [..., 3]

    def forward(self, rho0, w0, x, t):
        h = self.lifting(rho0, w0, x, t)
        u_hats: list[torch.Tensor] = []
        for layer in self.mp_layers:
            if self.use_checkpoint:
                h = torch_checkpoint(layer, h, x, t, rho0, w0, use_reentrant=False)
            else:
                h = layer(h, x, t, rho0, w0)
            dec = self._decode(h, rho0, w0)      # [..., 3] = (rho, v, w)
            # u_hats are (rho, w) for deep supervision compatibility.
            u_hats.append(torch.cat([dec[..., 0:1], dec[..., 2:3]], dim=-1))

        final = self._decode(h, rho0, w0)
        rho_pred = final[..., 0]
        w_pred = final[..., 2]
        return rho_pred, w_pred, u_hats

    def forward_primitive(self, rho0, w0, x, t):
        """Same as forward but also returns v_pred -- used by the (rho,v) loss."""
        h = self.lifting(rho0, w0, x, t)
        u_hats_rv: list[torch.Tensor] = []       # (rho, v) per layer
        for layer in self.mp_layers:
            if self.use_checkpoint:
                h = torch_checkpoint(layer, h, x, t, rho0, w0, use_reentrant=False)
            else:
                h = layer(h, x, t, rho0, w0)
            dec = self._decode(h, rho0, w0)
            u_hats_rv.append(dec[..., 0:2].clone())
        final = self._decode(h, rho0, w0)
        return final[..., 0], final[..., 1], final[..., 2], u_hats_rv


# --------------------------------------------------------------------------- #
# Constructor kwargs from a config block
# --------------------------------------------------------------------------- #
def cfg_to_kwargs_m2(model_cfg: dict) -> dict:
    _dhn = model_cfg.get("d_hidden_nonadj", None)
    _uh = model_cfg.get("update_hidden", None)
    return dict(
        stencil_k_x=int(model_cfg.get("stencil_k_x", 2)),
        stencil_k_t=int(model_cfg.get("stencil_k_t", 2)),
        d_latent=int(model_cfg.get("d_latent", 96)),
        d_hidden=int(model_cfg.get("d_hidden", 96)),
        n_layers=int(model_cfg.get("n_layers", 7)),
        activation=str(model_cfg.get("activation", "gelu")),
        causal_temporal=bool(model_cfg.get("causal_temporal", True)),
        d_hidden_nonadj=int(_dhn) if _dhn is not None else None,
        decoder_depth=int(model_cfg.get("decoder_depth", 3)),
        adj_depth=int(model_cfg.get("adj_depth", 3)),
        nonadj_depth=int(model_cfg.get("nonadj_depth", 3)),
        update_depth=int(model_cfg.get("update_depth", 3)),
        update_hidden=int(_uh) if _uh is not None else None,
        skip=bool(model_cfg.get("skip", False)),
        use_upwind_gate=bool(model_cfg.get("use_upwind_gate", False)),
        use_checkpoint=False,
        normalize_edge_offsets=bool(model_cfg.get("normalize_edge_offsets", True)),
        double_batch=bool(model_cfg.get("double_batch", False)),
        neighborhood_spacing=int(model_cfg.get("neighborhood_spacing", 1)),
    )


def load_hypno_arz_mark2_from_checkpoint(
    ckpt_path, device: str = "cpu", config_path=None, model_section: str = "hypno_arz_mark2",
):
    """Reconstruct a HypNO_ARZ_Mark2 from a bare state_dict + run config.yaml."""
    import yaml
    from pathlib import Path as _Path

    ckpt_path = _Path(ckpt_path)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)

    def _strip(sd):
        if isinstance(sd, dict) and any(k.startswith("_orig_mod.") for k in sd):
            return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        return sd

    state_dict = _strip(raw)

    if config_path is None:
        for cand in (ckpt_path.parent / "config.yaml",
                     ckpt_path.parent.parent / "config.yaml"):
            if cand.exists():
                config_path = cand
                break
        if config_path is None:
            raise FileNotFoundError(
                f"Could not locate config.yaml alongside {ckpt_path}; pass config_path=."
            )
    with _Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if model_section not in cfg:
        raise KeyError(
            f"Section {model_section!r} not in {config_path}; available: "
            f"{sorted(k for k in cfg if isinstance(cfg.get(k), dict))}"
        )
    model = HypNO_ARZ_Mark2(**cfg_to_kwargs_m2(cfg[model_section])).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, float("inf")  # tau = inf (homogeneous)
