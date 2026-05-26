"""HypNO-ARZ: space-time GNN operator for 1D ARZ with relaxation.

Mirrors hyperbolic_pde.models.hypno_st3 (HypNO-ST3 backbone) but swaps the
boundary computations for the ARZ system (plan §5).

Design notes (post-tuning, 2026-05-26):

* **Pure-pairwise edges are the only edge design.** Adjacent edges carry
  *interface* quantities only (lambda1_ij, lambda2_ij, drho, dv, dw, theta,
  upwind/lax indicators); non-adjacent edges carry only geometric features
  (rel_x, rel_t, sign(rel_x)) plus spectral radius. The two streams now have
  separate MLPs in both the lifting and the MP layers (no zero-padded
  unified-MLP path). i/j primitive levels are NOT replicated into edges --
  they live in h via the lifting's node MLP.
* **`normalize_edge_offsets` defaults to True.** Raw rel_x / rel_t are
  divided by dx_grid / dt_grid before going into the edge MLPs, so edge
  features are resolution-invariant integer-ish offsets. The physics gate
  and CFL still use raw rel_x / rel_t (those are physical comparisons).

Decoder outputs (rho, w) -- 2 channels.
Lifting node input (9 channels): rho0, w0, v0, y0, V(rho0), v0-V(rho0), x, t, xi.
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

# Frozen ARZ closures, kept in pure torch here (we re-implement them in-module
# rather than importing physics_arz to avoid the numpy/torch dispatch overhead
# inside the inner loop).
def _p(rho): return rho + rho * rho
def _dp(rho): return 1.0 + 2.0 * rho
def _Veq(rho): return 1.0 - rho
def _w_eq(rho): return 1.0 + rho * rho


# --------------------------------------------------------------------------- #
# Lifting layer
# --------------------------------------------------------------------------- #
class _ArzLifting(nn.Module):
    """Space-time lifting for ARZ (pure-pairwise edges).

    Node input (9 channels):
        rho0, w0, v0, y0, V(rho0), v0-V(rho0), x, t, xi=x/max(t,eps)

    Adjacent edges (8 dims):
        [sign(rel_x), rel_t_feat, drho, dv, dw, theta, lam1_ij, lam2_ij]
    Non-adjacent edges (3 dims):
        [rel_x_feat, rel_t_feat, sign(rel_x)]
    """

    def __init__(
        self,
        d_latent: int, d_hidden: int,
        stencil_k_x: int, stencil_k_t: int,
        activation: str = "gelu",
        causal_temporal: bool = True,
        normalize_edge_offsets: bool = True,
        d_hidden_nonadj: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.k_x = stencil_k_x
        self.k_t = stencil_k_t
        self.causal = causal_temporal
        self.normalize_edge_offsets = normalize_edge_offsets
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        self.node_mlp = _make_mlp(9, d_hidden, d_latent, 2, activation)
        self.adj_edge_mlp    = _make_mlp(8, d_hidden, d_latent, 2, activation)
        self.nonadj_edge_mlp = _make_mlp(3, dh_na,    d_latent, 2, activation)

        # Per-field gate temperatures.
        self.phys_temp1 = nn.Parameter(torch.tensor(0.0))   # softplus -> tau1
        self.phys_temp2 = nn.Parameter(torch.tensor(0.0))   # softplus -> tau2
        self.phys_gamma = nn.Parameter(torch.tensor(-2.0))  # sigmoid  -> gamma

        self.combine = _make_mlp(2 * d_latent, d_hidden, d_latent, 2, activation)

    def _gate_adj(
        self,
        rel_x: torch.Tensor,
        lam1_ij: torch.Tensor, lam2_ij: torch.Tensor,
        chi_1bad: torch.Tensor, theta: torch.Tensor,
    ) -> torch.Tensor:
        r = torch.sign(rel_x)
        tau1 = F.softplus(self.phys_temp1).clamp(min=1e-6)
        tau2 = F.softplus(self.phys_temp2).clamp(min=1e-6)
        g_up1 = torch.sigmoid(lam1_ij * r / tau1)
        g_up2 = torch.sigmoid(lam2_ij * r / tau2)
        g_up = 1.0 - (1.0 - g_up1) * (1.0 - g_up2)
        gamma = torch.sigmoid(self.phys_gamma)
        g_ent = 1.0 - (1.0 - gamma) * chi_1bad * (1.0 - theta)
        return g_up * g_ent

    def forward(
        self,
        rho0: torch.Tensor,   # [B, nx]
        w0: torch.Tensor,     # [B, nx]
        x: torch.Tensor,      # [B, nx] or [nx]
        t: torch.Tensor,      # [nt]
    ) -> torch.Tensor:
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
        y0_bc = rho0_bc * w0_bc
        Veq_bc = _Veq(rho0_bc)
        diseq_bc = v0_bc - Veq_bc
        eps_t = max(dt_grid, 1e-6)
        xi_bc = x_bc / t_bc.clamp(min=eps_t)

        node_in = torch.cat([
            rho0_bc, w0_bc, v0_bc, y0_bc, Veq_bc, diseq_bc,
            x_bc, t_bc, xi_bc,
        ], dim=-1)  # [B, nt, nx, 9]
        h_node = self.node_mlp(node_in)

        # Pad rho0, w0 for neighbour lookups.
        pad_x = _spatial_pad_width(self.k_x, dilated_spatial=False)
        rho0_pad = F.pad(rho0.unsqueeze(1), (pad_x, pad_x), mode="replicate").squeeze(1)
        w0_pad   = F.pad(w0.unsqueeze(1),   (pad_x, pad_x), mode="replicate").squeeze(1)
        x_pad    = F.pad(x.unsqueeze(1),    (pad_x, pad_x), mode="replicate").squeeze(1)
        t_pad    = F.pad(t.view(1, 1, -1),  (self.k_t, self.k_t), mode="replicate").view(-1)

        offsets = _enumerate_ball_offsets(self.k_x, self.k_t, self.causal)

        adj_feats:    list[torch.Tensor] = []
        nonadj_feats: list[torch.Tensor] = []
        adj_gates:    list[torch.Tensor] = []
        nonadj_gates: list[torch.Tensor] = []

        for di, dm in offsets:
            rho_j = rho0_pad[:, pad_x + di : pad_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            w_j   = w0_pad[:, pad_x + di : pad_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            x_j   = x_pad[:, pad_x + di : pad_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            t_j   = t_pad[self.k_t + dm : self.k_t + dm + nt].view(1, nt, 1, 1).expand(B, nt, nx, 1)
            rel_x = x_j - x_bc
            rel_t = t_j - t_bc

            if self.normalize_edge_offsets:
                rel_x_feat = rel_x / dx_grid
                rel_t_feat = rel_t / dt_grid
            else:
                rel_x_feat = rel_x
                rel_t_feat = rel_t

            r = torch.sign(rel_x)
            is_adj_sp = (dm == 0) and (abs(di) == 1)

            if is_adj_sp:
                v_j = w_j - _p(rho_j)
                drho = rho_j - rho0_bc
                dv = v_j - v0_bc
                dw = w_j - w0_bc
                theta = dw.abs() / (dw.abs() + dv.abs() + 1e-8)
                rho_ij = 0.5 * (rho0_bc + rho_j)
                v_ij = 0.5 * (v0_bc + v_j)
                lam1_ij = v_ij - rho_ij * _dp(rho_ij)
                lam2_ij = v_ij
                lam1_i = v0_bc - rho0_bc * _dp(rho0_bc)
                lam1_jn = v_j - rho_j * _dp(rho_j)
                lam1_L = torch.where(rel_x > 0, lam1_i, lam1_jn)
                lam1_R = torch.where(rel_x > 0, lam1_jn, lam1_i)
                chi_1bad = (lam1_L < lam1_R).float()

                edge_in = torch.cat([
                    r, rel_t_feat,
                    drho, dv, dw, theta,
                    lam1_ij, lam2_ij,
                ], dim=-1)  # 8
                gate = self._gate_adj(rel_x, lam1_ij, lam2_ij, chi_1bad, theta)
                adj_feats.append(edge_in)
                adj_gates.append(gate)
            else:
                edge_in = torch.cat([
                    rel_x_feat, rel_t_feat, r,
                ], dim=-1)  # 3
                gate = torch.ones_like(rel_x)
                nonadj_feats.append(edge_in)
                nonadj_gates.append(gate)

        all_gates = adj_gates + nonadj_gates
        gate_sum = torch.stack(all_gates, dim=-2).sum(dim=-2) + 1e-3

        n_adj = len(adj_feats); n_nonadj = len(nonadj_feats)
        adj_in = torch.stack(adj_feats, dim=3)
        nonadj_in = torch.stack(nonadj_feats, dim=3)
        d_out = h_node.shape[-1]
        adj_out = self.adj_edge_mlp(adj_in.reshape(-1, adj_in.shape[-1])).reshape(B, nt, nx, n_adj, d_out)
        nonadj_out = self.nonadj_edge_mlp(nonadj_in.reshape(-1, nonadj_in.shape[-1])).reshape(B, nt, nx, n_nonadj, d_out)
        all_msgs = torch.cat([adj_out, nonadj_out], dim=3)
        all_gates_t = torch.stack(all_gates, dim=3)
        agg = (all_gates_t / gate_sum.unsqueeze(3) * all_msgs).sum(dim=3)

        return self.combine(torch.cat([h_node, agg], dim=-1))


# --------------------------------------------------------------------------- #
# Physics-gated MP layer
# --------------------------------------------------------------------------- #
class _ArzMPLayer(nn.Module):
    """Two-edge-MLP space-time MP for ARZ (pure-pairwise).

    Adjacent edges (2d + 12):
        [h_i, h_j, r, lam1_ij, lam2_ij, lam1_ij*r, lam2_ij*r,
         chi_up1, chi_up2, drho, dv, dw, theta, chi_1bad]
    Non-adjacent edges (2d + 4):
        [h_i, h_j, rel_x_feat, rel_t_feat, sign(rel_x), max(|lam1_j|,|lam2_j|)]
    """

    def __init__(
        self,
        d_latent: int, d_hidden: int,
        k_x: int, k_t: int,
        activation: str = "gelu",
        causal_temporal: bool = True,
        d_hidden_nonadj: Optional[int] = None,
        shared_decoder: Optional[nn.Module] = None,
        normalize_edge_offsets: bool = True,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_t = k_t
        self.causal = causal_temporal
        self.normalize_edge_offsets = normalize_edge_offsets
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        if shared_decoder is None:
            raise ValueError("_ArzMPLayer requires a shared_decoder")
        object.__setattr__(self, "_shared_decoder", shared_decoder)

        self.phys_temp1 = nn.Parameter(torch.tensor(0.0))
        self.phys_temp2 = nn.Parameter(torch.tensor(0.0))
        self.phys_gamma = nn.Parameter(torch.tensor(-2.0))
        self.phys_cfl_scale = nn.Parameter(torch.tensor(0.0))

        self.adj_msg = _make_mlp(2 * d_latent + 12, d_hidden, d_latent, 3, activation)
        self.nonadj_msg = _make_mlp(2 * d_latent + 4, dh_na, d_latent, 3, activation)

        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    def _gate_adj(
        self, rel_x: torch.Tensor,
        lam1_ij: torch.Tensor, lam2_ij: torch.Tensor,
        chi_1bad: torch.Tensor, theta: torch.Tensor,
    ) -> torch.Tensor:
        r = torch.sign(rel_x)
        tau1 = F.softplus(self.phys_temp1).clamp(min=1e-6)
        tau2 = F.softplus(self.phys_temp2).clamp(min=1e-6)
        g_up1 = torch.sigmoid(lam1_ij * r / tau1)
        g_up2 = torch.sigmoid(lam2_ij * r / tau2)
        g_up = 1.0 - (1.0 - g_up1) * (1.0 - g_up2)
        gamma = torch.sigmoid(self.phys_gamma)
        g_ent = 1.0 - (1.0 - gamma) * chi_1bad * (1.0 - theta)
        return g_up * g_ent

    def _gate_nonadj(
        self, dm: int,
        rel_t: torch.Tensor,
        spec_radius_i: torch.Tensor,
        dx_grid: float,
    ) -> torch.Tensor:
        if dm == 0:
            return torch.ones_like(rel_t)
        cfl_scale = F.softplus(self.phys_cfl_scale).clamp(min=1e-6)
        cfl = spec_radius_i * rel_t.abs() / dx_grid
        return torch.exp(-cfl_scale * F.relu(cfl - 1.0) ** 2)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        rho0: torch.Tensor,
        w0: torch.Tensor,
    ) -> torch.Tensor:
        B, nt, nx, d = h.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)
        dx_val = (x[0, 1] - x[0, 0]).abs().item()
        dt_val = (t[1] - t[0]).abs().item()

        u_hat = self._shared_decoder(h)                       # [B, nt, nx, 2]
        rho_hat = u_hat[..., 0:1].clamp(1e-6, 1.0)
        w_hat = u_hat[..., 1:2]
        v_hat = w_hat - _p(rho_hat)
        lam1_i = v_hat - rho_hat * _dp(rho_hat)
        lam2_i = v_hat
        spec_i = torch.maximum(lam1_i.abs(), lam2_i.abs())

        pad_x = _spatial_pad_width(self.k_x, dilated_spatial=False)
        h_pad = _pad_space_time(h, pad_x, self.k_t)
        rho_hat_pad = _pad_space_time(rho_hat, pad_x, self.k_t)
        w_hat_pad = _pad_space_time(w_hat, pad_x, self.k_t)
        x_pad = F.pad(x.unsqueeze(1), (pad_x, pad_x), mode="replicate").squeeze(1)
        t_pad = F.pad(t.view(1, 1, -1), (self.k_t, self.k_t), mode="replicate").view(-1)

        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        t_i = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)

        offsets = _enumerate_ball_offsets(self.k_x, self.k_t, self.causal)

        adj_feats:    list[torch.Tensor] = []
        nonadj_feats: list[torch.Tensor] = []
        adj_gates:    list[torch.Tensor] = []
        nonadj_gates: list[torch.Tensor] = []

        for di, dm in offsets:
            h_j = h_pad[:, self.k_t + dm : self.k_t + dm + nt,
                            pad_x + di : pad_x + di + nx, :]
            rho_j = rho_hat_pad[:, self.k_t + dm : self.k_t + dm + nt,
                                    pad_x + di : pad_x + di + nx, :]
            w_j = w_hat_pad[:, self.k_t + dm : self.k_t + dm + nt,
                                pad_x + di : pad_x + di + nx, :]
            x_j = x_pad[:, pad_x + di : pad_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            t_j = t_pad[self.k_t + dm : self.k_t + dm + nt].view(1, nt, 1, 1).expand(B, nt, nx, 1)
            rel_x = x_j - x_i
            rel_t = t_j - t_i
            v_j = w_j - _p(rho_j)
            lam1_j = v_j - rho_j * _dp(rho_j)
            lam2_j = v_j
            spec_j = torch.maximum(lam1_j.abs(), lam2_j.abs())

            if self.normalize_edge_offsets:
                rel_x_feat = rel_x / dx_val
                rel_t_feat = rel_t / dt_val
            else:
                rel_x_feat = rel_x
                rel_t_feat = rel_t

            is_adj_sp = (dm == 0) and (abs(di) == 1)
            r = torch.sign(rel_x)

            if is_adj_sp:
                rho_ij = 0.5 * (rho_hat + rho_j)
                v_ij = 0.5 * (v_hat + v_j)
                lam1_ij = v_ij - rho_ij * _dp(rho_ij)
                lam2_ij = v_ij
                drho = rho_j - rho_hat
                dv = v_j - v_hat
                dw = w_j - w_hat
                theta = dw.abs() / (dw.abs() + dv.abs() + 1e-8)
                chi_up1 = (lam1_ij * r < 0).float()
                chi_up2 = (lam2_ij * r < 0).float()
                lam1_L = torch.where(rel_x > 0, lam1_i, lam1_j)
                lam1_R = torch.where(rel_x > 0, lam1_j, lam1_i)
                chi_1bad = (lam1_L < lam1_R).float()

                msg_in = torch.cat([
                    h, h_j,
                    r, lam1_ij, lam2_ij, lam1_ij * r, lam2_ij * r,
                    chi_up1, chi_up2,
                    drho, dv, dw, theta, chi_1bad,
                ], dim=-1)  # 2d + 12
                gate = self._gate_adj(rel_x, lam1_ij, lam2_ij, chi_1bad, theta)
                adj_feats.append(msg_in)
                adj_gates.append(gate)
            else:
                msg_in = torch.cat([
                    h, h_j,
                    rel_x_feat, rel_t_feat, r, spec_j,
                ], dim=-1)  # 2d + 4
                gate = self._gate_nonadj(dm, rel_t, spec_i, dx_val)
                nonadj_feats.append(msg_in)
                nonadj_gates.append(gate)

        all_gates = adj_gates + nonadj_gates
        gate_sum = torch.stack(all_gates, dim=-2).sum(dim=-2) + 1e-3

        n_adj = len(adj_feats)
        n_nonadj = len(nonadj_feats)
        adj_in = torch.stack(adj_feats, dim=3)
        nonadj_in = torch.stack(nonadj_feats, dim=3)
        adj_out = self.adj_msg(adj_in.reshape(-1, adj_in.shape[-1])).reshape(B, nt, nx, n_adj, d)
        nonadj_out = self.nonadj_msg(nonadj_in.reshape(-1, nonadj_in.shape[-1])).reshape(B, nt, nx, n_nonadj, d)
        all_msgs = torch.cat([adj_out, nonadj_out], dim=3)
        all_gates_t = torch.stack(all_gates, dim=3)
        agg = (all_gates_t / gate_sum.unsqueeze(3) * all_msgs).sum(dim=3)

        upd_in = torch.cat([h, agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)
        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class HypNO_ARZ(nn.Module):
    """HypNO operator for 1D ARZ with relaxation.

    Inputs:  rho0 [B,nx], w0 [B,nx], x [nx] or [B,nx], t [nt]
    Outputs: rho [B,nt,nx], w [B,nt,nx], u_hats list (deep supervision)
    """

    def __init__(
        self,
        stencil_k_x: int = 2,
        stencil_k_t: int = 2,
        d_latent: int = 64,
        d_hidden: int = 96,
        n_layers: int = 4,
        activation: str = "gelu",
        causal_temporal: bool = True,
        d_hidden_nonadj: Optional[int] = None,
        decoder_depth: int = 3,
        skip: bool = True,
        use_checkpoint: bool = False,
        normalize_edge_offsets: bool = True,
        **_ignored,
    ) -> None:
        super().__init__()
        self.skip = skip
        self.use_checkpoint = use_checkpoint
        self.normalize_edge_offsets = normalize_edge_offsets
        if _ignored:
            print(f"[HypNO_ARZ] IGNORED kwargs = {sorted(_ignored.keys())}")
        print(
            f"[HypNO_ARZ] kx={stencil_k_x} kt={stencil_k_t} "
            f"d_latent={d_latent} d_hidden={d_hidden} layers={n_layers} "
            f"skip={skip} normalize_edge_offsets={normalize_edge_offsets}"
        )

        self.lifting = _ArzLifting(
            d_latent, d_hidden,
            stencil_k_x=stencil_k_x, stencil_k_t=stencil_k_t,
            activation=activation, causal_temporal=causal_temporal,
            normalize_edge_offsets=normalize_edge_offsets,
            d_hidden_nonadj=d_hidden_nonadj,
        )

        # Decoder outputs (rho, w) -- 2 channels.
        self.decoder = _make_mlp(d_latent, d_hidden, 2, decoder_depth, activation)

        self.mp_layers = nn.ModuleList([
            _ArzMPLayer(
                d_latent, d_hidden,
                k_x=stencil_k_x, k_t=stencil_k_t,
                activation=activation, causal_temporal=causal_temporal,
                d_hidden_nonadj=d_hidden_nonadj,
                shared_decoder=self.decoder,
                normalize_edge_offsets=normalize_edge_offsets,
            )
            for _ in range(n_layers)
        ])

    def _decode(
        self, h: torch.Tensor, rho0: torch.Tensor, w0: torch.Tensor,
    ) -> torch.Tensor:
        out = self.decoder(h)  # [B, nt, nx, 2]
        if self.skip:
            B, nt, nx, _ = out.shape
            u0 = torch.stack([rho0, w0], dim=-1).unsqueeze(1).expand(B, nt, nx, 2)
            out = out + u0
        return out

    def forward(
        self,
        rho0: torch.Tensor, w0: torch.Tensor,
        x: torch.Tensor, t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        h = self.lifting(rho0, w0, x, t)

        u_hats: list[torch.Tensor] = []
        for layer in self.mp_layers:
            if self.use_checkpoint:
                h = torch_checkpoint(
                    layer, h, x, t, rho0, w0, use_reentrant=False,
                )
            else:
                h = layer(h, x, t, rho0, w0)
            u_hats.append(self._decode(h, rho0, w0))

        u_pred = u_hats[-1]                              # [B, nt, nx, 2]
        rho_pred = u_pred[..., 0]
        w_pred = u_pred[..., 1]
        return rho_pred, w_pred, u_hats
