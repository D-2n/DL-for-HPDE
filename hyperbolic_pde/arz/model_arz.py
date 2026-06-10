"""HypNO-ARZ: space-time GNN operator for 1D ARZ with relaxation.

Mirrors hyperbolic_pde.models.hypno_st3 (HypNO-ST3 backbone) but swaps the
boundary computations for the ARZ system (plan section 5).

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

This module also exposes `load_hypno_arz_from_checkpoint` -- a single helper
that handles both checkpoint formats:

  (A) legacy: dict with keys {model, opt, epoch, args, tau} from the old
      CLI-flag trainer (`hyperbolic_pde.arz.train_arz`, removed).
  (B) new (matches LWR): bare state_dict saved by
      `hyperbolic_pde/scripts/train_hypno_arz.py`. The trainer writes the
      architecture into `run_dir/config.yaml`, so reconstruction needs that
      file too (pass via `config_path=` or let the helper auto-locate it
      alongside the checkpoint).
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

# Frozen ARZ closures. We branch on physics_arz._P_FORM at *call* time -- the
# branch resolves to a single torch op either way, so the cost is negligible
# (Python int compare) and we avoid the import-time freeze that would make the
# pressure_form switch ineffective for already-instantiated models.
from hyperbolic_pde.arz import physics_arz as _P_MOD


def _p(rho):
    if _P_MOD._P_FORM == "rho":
        return rho
    return rho + rho * rho


def _dp(rho):
    if _P_MOD._P_FORM == "rho":
        return torch.ones_like(rho)
    return 1.0 + 2.0 * rho


def _entropy_bad_1shock(
    rho_i, v_i, lam1_i, rho_j, v_j, lam1_j, rel_x, lam1_ij,
    delta: float = 0.01, eps: float = 1e-6,
):
    """Lax-inadmissibility flag for the genuinely-nonlinear 1-field.

    Mirrors the LWR Oleinik gate (hypno_st3._physics_gate_ball): a discontinuity
    is an *admissible* 1-shock iff the 1-characteristics converge into it,

        lambda1(U_L) >= s1 >= lambda1(U_R),

    where s1 is the Rankine-Hugoniot speed. Across a genuine 1-wave the
    1-Riemann invariant w is preserved, so s1 follows from RH on the rho
    component, s1 = (rho_R*v_R - rho_L*v_L)/(rho_R - rho_L); it falls back to the
    interface eigenvalue lam1_ij when |rho_R - rho_L| is below eps (smooth
    limit). The flag is 1 only when the interface *looks like* a 1-shock
    (lambda1_L > lambda1_R, converging) yet FAILS the speed bracketing -- i.e.
    a speed-inadmissible shock. Rarefaction-shaped interfaces
    (lambda1_L < lambda1_R) are NOT flagged here; they are simply not 1-shocks
    and the upwind gate handles their information flow.

    L/R are assigned by the edge orientation: for rel_x>0 the source cell i is
    the left state, else the neighbour j is.
    """
    rho_L = torch.where(rel_x > 0, rho_i, rho_j)
    rho_R = torch.where(rel_x > 0, rho_j, rho_i)
    v_L = torch.where(rel_x > 0, v_i, v_j)
    v_R = torch.where(rel_x > 0, v_j, v_i)
    lam1_L = torch.where(rel_x > 0, lam1_i, lam1_j)
    lam1_R = torch.where(rel_x > 0, lam1_j, lam1_i)

    drho = rho_R - rho_L
    drho_safe = torch.where(drho.abs() < eps, torch.ones_like(drho), drho)
    s1_rh = (rho_R * v_R - rho_L * v_L) / drho_safe
    s1 = torch.where(drho.abs() < eps, lam1_ij, s1_rh)

    is_1shock = (lam1_L > lam1_R).float()                       # converging chars
    entropy_ok = ((lam1_L >= s1 - delta) & (s1 >= lam1_R - delta)).float()
    return is_1shock * (1.0 - entropy_ok)


def _Veq(rho):
    return 1.0 - rho


def _w_eq(rho):
    if _P_MOD._P_FORM == "rho":
        return torch.ones_like(rho)
    return 1.0 + rho * rho


# --------------------------------------------------------------------------- #
# Lifting layer
# --------------------------------------------------------------------------- #
class _ArzLifting(nn.Module):
    """Space-time lifting for ARZ (pure-pairwise edges).

    Node input:
      * 9 channels if use_relaxation_features=True (default, ARZ-with-relaxation):
            rho0, w0, v0, y0, V(rho0), v0-V(rho0), x, t, xi=x/max(t,eps)
      * 7 channels if use_relaxation_features=False (homogeneous / Riemann):
            rho0, w0, v0, y0, x, t, xi
        (Veq and diseq are dropped; the network gets no equilibrium hint.)

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
        use_relaxation_features: bool = True,
        double_batch: bool = False,
        neighborhood_spacing: int = 1,
    ) -> None:
        super().__init__()
        if double_batch and stencil_k_x % 2 != 0:
            raise ValueError(
                f"double_batch requires stencil_k_x even; got {stencil_k_x}."
            )
        self.k_x = stencil_k_x
        self.k_t = stencil_k_t
        self.causal = causal_temporal
        self.normalize_edge_offsets = normalize_edge_offsets
        self.use_relaxation_features = use_relaxation_features
        self.double_batch = double_batch
        self.neighborhood_spacing = neighborhood_spacing
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        n_node_in = 9 if use_relaxation_features else 7
        self.node_mlp = _make_mlp(n_node_in, d_hidden, d_latent, 2, activation)
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
        # Upwind convention: edge `i <- j` should pass information when the
        # wave at the interface flows from j toward i, i.e. when
        # lambda_m * sign(x_j - x_i) < 0. The sigmoid argument therefore
        # carries a leading minus sign so g_up_m is large (gate open) on
        # upstream edges and small (gate closed) on downstream edges.
        g_up1 = torch.sigmoid(-lam1_ij * r / tau1)
        g_up2 = torch.sigmoid(-lam2_ij * r / tau2)
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
        eps_t = max(dt_grid, 1e-6)
        xi_bc = x_bc / t_bc.clamp(min=eps_t)

        if self.use_relaxation_features:
            Veq_bc = _Veq(rho0_bc)
            diseq_bc = v0_bc - Veq_bc
            node_in = torch.cat([
                rho0_bc, w0_bc, v0_bc, y0_bc, Veq_bc, diseq_bc,
                x_bc, t_bc, xi_bc,
            ], dim=-1)  # [B, nt, nx, 9]
        else:
            node_in = torch.cat([
                rho0_bc, w0_bc, v0_bc, y0_bc,
                x_bc, t_bc, xi_bc,
            ], dim=-1)  # [B, nt, nx, 7]
        h_node = self.node_mlp(node_in)

        # Pad rho0, w0 for neighbour lookups.
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
                chi_1bad = _entropy_bad_1shock(
                    rho0_bc, v0_bc, lam1_i, rho_j, v_j, lam1_jn, rel_x, lam1_ij,
                )

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

    Adjacent edges (2d + 3):
        [h_i, h_j, lam1_ij, lam2_ij, sign(rel_x)]
    Non-adjacent edges (2d + 3):
        [h_i, h_j, rel_x_feat, rel_t_feat, sign(rel_x)]

    Pure-pairwise: the message carries only the endpoint latents plus minimal
    pair-intrinsic interface scalars (the two interface eigenvalues and the edge
    orientation). Quantities the gate already consumes (theta, chi_1bad, the
    per-family upwind flags) and node-native scalars (state jumps, neighbour
    spectral radius) are kept out of the message. Mirrors the LWR pure-pairwise
    convention, extended to ARZ's two characteristic families.
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
        double_batch: bool = False,
        neighborhood_spacing: int = 1,
    ) -> None:
        super().__init__()
        if double_batch and k_x % 2 != 0:
            raise ValueError(
                f"double_batch requires k_x even; got {k_x}."
            )
        self.k_x = k_x
        self.k_t = k_t
        self.causal = causal_temporal
        self.normalize_edge_offsets = normalize_edge_offsets
        self.double_batch = double_batch
        self.neighborhood_spacing = neighborhood_spacing
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        if shared_decoder is None:
            raise ValueError("_ArzMPLayer requires a shared_decoder")
        object.__setattr__(self, "_shared_decoder", shared_decoder)

        self.phys_temp1 = nn.Parameter(torch.tensor(0.0))
        self.phys_temp2 = nn.Parameter(torch.tensor(0.0))
        self.phys_gamma = nn.Parameter(torch.tensor(-2.0))
        self.phys_cfl_scale = nn.Parameter(torch.tensor(0.0))

        self.adj_msg = _make_mlp(2 * d_latent + 3, d_hidden, d_latent, 3, activation)
        self.nonadj_msg = _make_mlp(2 * d_latent + 3, dh_na, d_latent, 3, activation)

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
        # Upwind convention: edge `i <- j` should pass information when the
        # wave at the interface flows from j toward i, i.e. when
        # lambda_m * sign(x_j - x_i) < 0. The sigmoid argument therefore
        # carries a leading minus sign so g_up_m is large (gate open) on
        # upstream edges and small (gate closed) on downstream edges.
        g_up1 = torch.sigmoid(-lam1_ij * r / tau1)
        g_up2 = torch.sigmoid(-lam2_ij * r / tau2)
        g_up = 1.0 - (1.0 - g_up1) * (1.0 - g_up2)
        gamma = torch.sigmoid(self.phys_gamma)
        g_ent = 1.0 - (1.0 - gamma) * chi_1bad * (1.0 - theta)
        return g_up * g_ent

    def _gate_nonadj(
        self, dm: int,
        rel_t: torch.Tensor,
        rel_x: torch.Tensor,
        spec_radius_i: torch.Tensor,
        dx_grid: float,
    ) -> torch.Tensor:
        if dm == 0:
            return torch.ones_like(rel_t)
        cfl_scale = F.softplus(self.phys_cfl_scale).clamp(min=1e-6)
        # CFL number uses the edge's true spatial span |di|*dx (= |rel_x|),
        # not the scalar grid dx, so long-reach diagonal edges are judged
        # against the distance they actually cover. Pure-temporal edges
        # (di==0, rel_x==0) span no space and are always inside the
        # characteristic cone -> gate stays open.
        dx_edge = rel_x.abs()
        cfl = torch.where(
            dx_edge > 0,
            spec_radius_i * rel_t.abs() / dx_edge.clamp(min=1e-12),
            torch.zeros_like(rel_t),
        )
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

        pad_x = _spatial_pad_width(
            self.k_x, dilated_spatial=False,
            double_batch=self.double_batch,
            neighborhood_spacing=self.neighborhood_spacing,
        )
        h_pad = _pad_space_time(h, pad_x, self.k_t)
        rho_hat_pad = _pad_space_time(rho_hat, pad_x, self.k_t)
        w_hat_pad = _pad_space_time(w_hat, pad_x, self.k_t)
        x_pad = F.pad(x.unsqueeze(1), (pad_x, pad_x), mode="replicate").squeeze(1)
        t_pad = F.pad(t.view(1, 1, -1), (self.k_t, self.k_t), mode="replicate").view(-1)

        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        t_i = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)

        offsets = _enumerate_ball_offsets(
            self.k_x, self.k_t, self.causal,
            double_batch=self.double_batch,
            neighborhood_spacing=self.neighborhood_spacing,
        )

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
            lam1_j = v_j - rho_j * _dp(rho_j)  # needed by the entropy flag below

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
                dv = v_j - v_hat
                dw = w_j - w_hat
                theta = dw.abs() / (dw.abs() + dv.abs() + 1e-8)
                chi_1bad = _entropy_bad_1shock(
                    rho_hat, v_hat, lam1_i, rho_j, v_j, lam1_j, rel_x, lam1_ij,
                )

                # Pure-pairwise: only the interface eigenvalues. The per-family
                # upwind flags (chi_up1/chi_up2) were removed -- they duplicate
                # what the upwind gate already encodes from lam*_ij and r, and
                # carried no interpretable signal in the message. theta/chi_1bad
                # feed the gate only, not the message.
                msg_in = torch.cat([
                    h, h_j,
                    lam1_ij, lam2_ij,
                    r,
                ], dim=-1)  # 2d + 3
                gate = self._gate_adj(rel_x, lam1_ij, lam2_ij, chi_1bad, theta)
                adj_feats.append(msg_in)
                adj_gates.append(gate)
            else:
                msg_in = torch.cat([
                    h, h_j,
                    rel_x_feat, rel_t_feat, r,
                ], dim=-1)  # 2d + 3
                gate = self._gate_nonadj(dm, rel_t, rel_x, spec_i, dx_val)
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
        use_relaxation_features: bool = True,
        double_batch: bool = False,
        neighborhood_spacing: int = 1,
        **_ignored,
    ) -> None:
        super().__init__()
        self.skip = skip
        self.use_checkpoint = use_checkpoint
        self.normalize_edge_offsets = normalize_edge_offsets
        self.use_relaxation_features = use_relaxation_features
        self.double_batch = double_batch
        self.neighborhood_spacing = neighborhood_spacing
        if _ignored:
            print(f"[HypNO_ARZ] IGNORED kwargs = {sorted(_ignored.keys())}")
        print(
            f"[HypNO_ARZ] kx={stencil_k_x} kt={stencil_k_t} "
            f"d_latent={d_latent} d_hidden={d_hidden} layers={n_layers} "
            f"skip={skip} normalize_edge_offsets={normalize_edge_offsets} "
            f"use_relaxation_features={use_relaxation_features} "
            f"double_batch={double_batch}"
            + (f" neighborhood_spacing={neighborhood_spacing}" if double_batch else "")
        )

        self.lifting = _ArzLifting(
            d_latent, d_hidden,
            stencil_k_x=stencil_k_x, stencil_k_t=stencil_k_t,
            activation=activation, causal_temporal=causal_temporal,
            normalize_edge_offsets=normalize_edge_offsets,
            d_hidden_nonadj=d_hidden_nonadj,
            use_relaxation_features=use_relaxation_features,
            double_batch=double_batch,
            neighborhood_spacing=neighborhood_spacing,
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
                double_batch=double_batch,
                neighborhood_spacing=neighborhood_spacing,
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


# --------------------------------------------------------------------------- #
# Checkpoint loading (handles both legacy dict and new state_dict formats)
# --------------------------------------------------------------------------- #
def _cfg_to_kwargs(model_cfg: dict) -> dict:
    """Translate a `hypno_arz` config block into HypNO_ARZ constructor kwargs."""
    _dhn = model_cfg.get("d_hidden_nonadj", None)
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
        skip=bool(model_cfg.get("skip", True)),
        use_checkpoint=False,
        normalize_edge_offsets=bool(model_cfg.get("normalize_edge_offsets", True)),
        use_relaxation_features=bool(model_cfg.get("use_relaxation_features", True)),
        double_batch=bool(model_cfg.get("double_batch", False)),
        neighborhood_spacing=int(model_cfg.get("neighborhood_spacing", 1)),
    )


def load_hypno_arz_from_checkpoint(
    ckpt_path,
    device: str = "cpu",
    config_path=None,
    model_section: Optional[str] = None,
):
    """Reconstruct a HypNO_ARZ from a checkpoint file.

    Auto-detects the format:
      * **Legacy dict** (`{"model": ..., "args": {...}, "tau": ...}`): kwargs
        come from `ck["args"]`.
      * **Bare state_dict** (new trainer's `checkpoint_epoch*.pt`,
        `model_final.pt`, or `<save_path>`): the architecture lives in
        `<run_dir>/config.yaml`. The helper auto-locates that file by walking
        upwards from `ckpt_path` (the new trainer writes it next to the
        checkpoints). Override with explicit `config_path=`.

    Returns
    -------
    model : HypNO_ARZ (on `device`, in eval mode)
    tau   : float | None  -- pulled from the legacy dict if present,
                              else from `arz_data.tau` / `arz_trial.tau` in
                              the YAML, else None.
    """
    import yaml
    from pathlib import Path as _Path

    ckpt_path = _Path(ckpt_path)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Strip any torch.compile prefix.
    def _strip(sd):
        if isinstance(sd, dict) and any(k.startswith("_orig_mod.") for k in sd):
            return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        return sd

    # --- Format detection ------------------------------------------------- #
    is_legacy = isinstance(raw, dict) and "model" in raw and "args" in raw

    if is_legacy:
        a = raw["args"]
        kwargs = dict(
            stencil_k_x=a["kx"], stencil_k_t=a["kt"],
            d_latent=a["d_latent"], d_hidden=a["d_hidden"],
            n_layers=a["depth"], decoder_depth=a["decoder_depth"],
            skip=a["skip"], use_checkpoint=False,
            normalize_edge_offsets=a.get("normalize_edge_offsets", True),
        )
        model = HypNO_ARZ(**kwargs).to(device)
        model.load_state_dict(_strip(raw["model"]))
        model.eval()
        tau = float(raw.get("tau")) if raw.get("tau") is not None else None
        return model, tau

    # New format: bare state_dict (plus optional torch.compile prefix).
    state_dict = _strip(raw)

    if config_path is None:
        # The trainer writes config.yaml at run_dir/. Try sibling, then
        # parent (when ckpt is at run_dir/checkpoint_epochN.pt) and grandparent.
        for cand in (
            ckpt_path.parent / "config.yaml",
            ckpt_path.parent.parent / "config.yaml",
        ):
            if cand.exists():
                config_path = cand
                break
        if config_path is None:
            raise FileNotFoundError(
                f"Could not locate config.yaml alongside {ckpt_path}. "
                f"Pass config_path= explicitly."
            )
    with _Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Pick model section: explicit arg > auto-detect from state_dict > "hypno_arz" default.
    # The Riemann pretraining run uses hypno_arz_riemann with a 7-channel node MLP
    # (use_relaxation_features=false); the relaxation run uses hypno_arz with 9 channels.
    candidates = ("hypno_arz", "hypno_arz_riemann")
    if model_section is None:
        node_w = state_dict.get("lifting.node_mlp.0.weight")
        if node_w is not None and node_w.shape[1] == 7:
            # 7-channel lifting => relaxation features off => Riemann section
            model_section = "hypno_arz_riemann" if "hypno_arz_riemann" in cfg else "hypno_arz"
        else:
            model_section = "hypno_arz" if "hypno_arz" in cfg else next(
                (s for s in candidates if s in cfg), "hypno_arz"
            )
    if model_section not in cfg:
        raise KeyError(
            f"Section {model_section!r} not found in {config_path}; available: "
            f"{sorted(k for k in cfg if isinstance(cfg.get(k), dict))}"
        )
    model_cfg = cfg.get(model_section, {})
    kwargs = _cfg_to_kwargs(model_cfg)
    model = HypNO_ARZ(**kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # tau preference: Riemann section (if Riemann model) > arz_data > arz_trial > None.
    tau = None
    tau_order = (
        ("arz_riemann_trial", "arz_data", "arz_trial")
        if model_section == "hypno_arz_riemann"
        else ("arz_data", "arz_trial", "arz_riemann_trial")
    )
    for sec in tau_order:
        if sec in cfg and isinstance(cfg[sec], dict) and "tau" in cfg[sec]:
            tau = float(cfg[sec]["tau"])
            break
    return model, tau
