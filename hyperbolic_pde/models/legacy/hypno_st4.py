"""HypNO-ST v4 — time-marching spatial operator.

Predicts one step at a time: u^{n+1} = u^n + dMLP(h^{(L),n}).

Changes from hypno_st3.py
-------------------------
* Purely spatial. No temporal message passing, no joint space-time graph.
* One-shot input is a single frame u^n [B, nx] (Markovian). The full
  trajectory is obtained by iterating the model nt - 1 times externally.
* Decoder = Option A step-increment: u^{n+1} = u^n + dMLP(h^{(L)}).
  No global u0 skip (that was the st2/st3 `skip: true` behavior).
* `shock_mode` toggle preserved: "physics" | "classic" | "weno".
  "pinn" mode dropped (would need a separate time-residual pipeline).
* Adjacent / non-adjacent split kept from st3: separate edge MLPs,
  optional smaller `d_hidden_nonadj` for the non-adj MLP.
* Physics gate (upwind + Oleinik + optional char-cone using dt) preserved
  on adjacent edges; non-adj falls back to 1 (or char-cone w/ dt).
* Per-layer `state_probe` retained for deep supervision.
* No temporal_cone, no causal_temporal, no radius_t, no stencil_k_t.
"""
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


# Feature vector widths (no time slot, unlike st3):
# Adjacent (13): u_i, u_j, du, |du|, u_avg, rel_x, f_i, f_j, a_i, a_j, a_ij, sign(a_ij), upwind
# Non-adj  (8):  u_i, u_j, f_i, f_j, a_i, a_j, x_i, x_j
N_EDGE_FEATS_ADJ = 13
N_EDGE_FEATS_NONADJ = 8


def _compute_adj_edge_feats(
    u_i: torch.Tensor, u_j: torch.Tensor, rel_x: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """(du, u_avg, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind) for adjacent edges."""
    f_i = u_i * (1.0 - u_i)
    f_j = u_j * (1.0 - u_j)
    a_i = 1.0 - 2.0 * u_i
    a_j = 1.0 - 2.0 * u_j
    du = u_j - u_i
    u_avg = 0.5 * (u_i + u_j)
    du_safe = torch.where(du.abs() < 1e-6, torch.ones_like(du), du)
    a_ij = torch.where(du.abs() < 1e-6, 1.0 - 2.0 * u_avg, (f_j - f_i) / du_safe)
    sign_a = torch.sign(a_ij)
    upwind = (a_ij * rel_x < 0).float()
    return du, u_avg, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind


def _compute_nonadj_pair_feats(
    u_i: torch.Tensor, u_j: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """(f_i, f_j, a_i, a_j) pointwise for non-adjacent edges."""
    f_i = u_i * (1.0 - u_i)
    f_j = u_j * (1.0 - u_j)
    a_i = 1.0 - 2.0 * u_i
    a_j = 1.0 - 2.0 * u_j
    return f_i, f_j, a_i, a_j


# --------------------------------------------------------------------------- #
# Lifting layer — purely spatial
# --------------------------------------------------------------------------- #
class _SpatialLifting(nn.Module):
    """u^n + x -> h^(0) [B, nx, d_latent].

    Node MLP input = (u_n, x) = 2-dim.
    Adjacent edge MLP input  = 13-dim static features (no time).
    Non-adj  edge MLP input  =  8-dim static features (no time).
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        stencil_k: int,
        activation: str,
        radius_x: float | None = None,
        encoder_scaling: str = "physics",
        encoder_type: str = "gnn",
        use_char_cone: bool = False,
        d_hidden_nonadj: int | None = None,
    ) -> None:
        super().__init__()
        self.k = stencil_k
        self.radius_x = radius_x
        self.encoder_scaling = encoder_scaling
        self.encoder_type = encoder_type
        self.use_char_cone = use_char_cone
        self.node_mlp = _make_mlp(2, d_hidden, d_latent, 2, activation)
        if encoder_type == "mlp":
            return

        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj
        self.edge_mlp_adj = _make_mlp(N_EDGE_FEATS_ADJ, d_hidden, d_latent, 2, activation)
        self.edge_mlp_nonadj = _make_mlp(N_EDGE_FEATS_NONADJ, dh_na, d_latent, 2, activation)

        if encoder_scaling == "gate_net":
            self.gate_net_adj = _make_mlp(N_EDGE_FEATS_ADJ, d_hidden, 1, 2, activation)
            self.gate_net_nonadj = _make_mlp(N_EDGE_FEATS_NONADJ, dh_na, 1, 2, activation)
        elif encoder_scaling == "upwind":
            self.upwind_alpha = nn.Parameter(torch.tensor(0.1))
        elif encoder_scaling == "physics":
            self.phys_temperature = nn.Parameter(torch.tensor(0.0))
            self.phys_gamma_entropy = nn.Parameter(torch.tensor(-2.0))
            if use_char_cone:
                self.phys_char_width = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(f"Unknown encoder_scaling: {encoder_scaling!r}")

        self.combine = _make_mlp(2 * d_latent, d_hidden, d_latent, 2, activation)

    def _physics_gate_adj(
        self, u_i: torch.Tensor, u_j: torch.Tensor, rel_x: torch.Tensor,
        a_ij: torch.Tensor, a_j: torch.Tensor, dt: float, dx_grid: float,
    ) -> torch.Tensor:
        temperature = F.softplus(self.phys_temperature).clamp(min=1e-6)
        gamma_ent = torch.sigmoid(self.phys_gamma_entropy)

        g_upwind = torch.sigmoid(-a_ij * rel_x / temperature)

        u_L = torch.where(rel_x > 0, u_i, u_j)
        u_R = torch.where(rel_x > 0, u_j, u_i)
        a_L = 1.0 - 2.0 * u_L
        a_R = 1.0 - 2.0 * u_R
        is_shock = (u_L > u_R).float()
        entropy_ok = ((a_L >= a_ij - 0.01) & (a_ij >= a_R - 0.01)).float()
        g_entropy = 1.0 - is_shock * (1.0 - entropy_ok) * (1.0 - gamma_ent)

        gate = g_upwind * g_entropy

        if self.use_char_cone:
            char_w = F.softplus(self.phys_char_width).clamp(min=1e-6)
            char_miss = (rel_x + a_j * dt).abs()
            sigma = char_w * (dx_grid + a_j.abs() * abs(dt))
            gate = gate * torch.exp(-0.5 * (char_miss / sigma.clamp(min=1e-6)) ** 2)

        return gate

    def _physics_gate_nonadj(
        self, rel_x: torch.Tensor, a_j: torch.Tensor,
        dt: float, dx_grid: float,
    ) -> torch.Tensor:
        if not self.use_char_cone:
            return torch.ones_like(rel_x)
        char_w = F.softplus(self.phys_char_width).clamp(min=1e-6)
        char_miss = (rel_x + a_j * dt).abs()
        sigma = char_w * (dx_grid + a_j.abs() * abs(dt))
        return torch.exp(-0.5 * (char_miss / sigma.clamp(min=1e-6)) ** 2)

    def _get_max_k(self, x: torch.Tensor) -> int:
        dx = (x[0, 1] - x[0, 0]).abs().item()
        return max(1, int(self.radius_x / dx + 0.5))

    def forward(
        self, u_n: torch.Tensor, x: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        B, nx = u_n.shape

        u_bc = u_n.unsqueeze(-1)
        x_bc = x.unsqueeze(-1)

        node_in = torch.cat([u_bc, x_bc], dim=-1)
        h_node = self.node_mlp(node_in)

        if self.encoder_type == "mlp":
            return h_node

        dx_grid = (x[0, 1] - x[0, 0]).abs().item()
        k = self._get_max_k(x) if self.radius_x is not None else self.k

        u_pad = F.pad(u_n.unsqueeze(1), (k, k), mode="replicate").squeeze(1)
        x_pad = F.pad(x.unsqueeze(1), (k, k), mode="replicate").squeeze(1)

        agg = torch.zeros_like(h_node)
        for j in range(-k, k + 1):
            is_adjacent = abs(j) <= 1
            u_k = u_pad[:, k + j : k + j + nx]
            x_k = x_pad[:, k + j : k + j + nx]

            u_i_bc = u_bc
            u_k_bc = u_k.unsqueeze(-1)
            x_i_bc = x_bc
            x_k_bc = x_k.unsqueeze(-1)

            f_i = u_i_bc * (1.0 - u_i_bc)
            f_k = u_k_bc * (1.0 - u_k_bc)
            a_i = 1.0 - 2.0 * u_i_bc
            a_k = 1.0 - 2.0 * u_k_bc
            rel_x = x_k_bc - x_i_bc

            if is_adjacent:
                du = u_k_bc - u_i_bc
                u_avg = 0.5 * (u_i_bc + u_k_bc)
                du_safe = torch.where(du.abs() < 1e-6, torch.ones_like(du), du)
                a_ik = torch.where(du.abs() < 1e-6, 1.0 - 2.0 * u_avg, (f_k - f_i) / du_safe)
                sign_a = torch.sign(a_ik)
                upwind = (a_ik * rel_x < 0).float()

                edge_in = torch.cat([
                    u_i_bc, u_k_bc,
                    du, du.abs(), u_avg,
                    rel_x,
                    f_i, f_k,
                    a_i, a_k, a_ik,
                    sign_a, upwind,
                ], dim=-1)                                              # [..., 13]
                msg = self.edge_mlp_adj(edge_in)

                if self.encoder_scaling == "gate_net":
                    gate = torch.sigmoid(self.gate_net_adj(edge_in))
                elif self.encoder_scaling == "upwind":
                    alpha = torch.sigmoid(self.upwind_alpha)
                    gate = upwind + alpha * (1.0 - upwind)
                else:  # physics
                    gate = self._physics_gate_adj(
                        u_i=u_i_bc, u_j=u_k_bc, rel_x=rel_x,
                        a_ij=a_ik, a_j=a_k, dt=dt, dx_grid=dx_grid,
                    )
                contrib = gate * msg
            else:
                edge_in = torch.cat([
                    u_i_bc, u_k_bc,
                    f_i, f_k,
                    a_i, a_k,
                    x_i_bc, x_k_bc,
                ], dim=-1)                                              # [..., 8]
                msg = self.edge_mlp_nonadj(edge_in)

                if self.encoder_scaling == "gate_net":
                    gate = torch.sigmoid(self.gate_net_nonadj(edge_in))
                elif self.encoder_scaling == "upwind":
                    alpha = torch.sigmoid(self.upwind_alpha)
                    gate = alpha * torch.ones_like(u_i_bc)
                else:  # physics
                    gate = self._physics_gate_nonadj(
                        rel_x=rel_x, a_j=a_k, dt=dt, dx_grid=dx_grid,
                    )
                contrib = gate * msg

            if self.radius_x is not None:
                contrib = contrib * (rel_x.abs() <= self.radius_x)
            agg = agg + contrib

        return self.combine(torch.cat([h_node, agg], dim=-1))


# --------------------------------------------------------------------------- #
# Spatial MP layers
# --------------------------------------------------------------------------- #
class _ClassicSpatialMP(nn.Module):
    """Spatial MP with no physics weighing, adj/non-adj split."""

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        k_x: int,
        activation: str,
        radius_x: float | None = None,
        d_hidden_nonadj: int | None = None,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.radius_x = radius_x
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        self.state_probe = nn.Linear(d_latent, 1)

        # adj spatial: (h_i, h_j, x_i, x_j, rel_x, du, |du|) = 2d + 5
        self.sp_msg_adj = _make_mlp(2 * d_latent + 5, d_hidden, d_latent, 3, activation)
        # non-adj spatial: (h_i, h_j, u_i, u_j, f_i, f_j, a_i, a_j, x_i, x_j) = 2d + 8
        self.sp_msg_nonadj = _make_mlp(2 * d_latent + 8, dh_na, d_latent, 3, activation)

        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    def forward(
        self, h: torch.Tensor, x: torch.Tensor, u_n: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        B, nx, d = h.shape
        dx_val = (x[0, 1] - x[0, 0]).abs().item()
        k_x = max(1, int(self.radius_x / dx_val + 0.5)) if self.radius_x is not None else self.k_x

        h_pad = F.pad(h.permute(0, 2, 1), (k_x, k_x), mode="replicate").permute(0, 2, 1)
        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        u_pad = F.pad(u_n.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)

        x_i = x.unsqueeze(-1)

        sp_agg = h.new_zeros(B, nx, d)
        for j in range(-k_x, k_x + 1):
            is_adjacent = abs(j) <= 1
            h_j = h_pad[:, k_x + j : k_x + j + nx, :]
            x_j = x_pad[:, k_x + j : k_x + j + nx].unsqueeze(-1)
            rel_x = x_j - x_i
            x_i_exp = x_i.expand_as(x_j)

            if is_adjacent:
                du = (u_n - u_pad[:, k_x + j : k_x + j + nx]).unsqueeze(-1)
                msg_in = torch.cat([h, h_j, x_i_exp, x_j, rel_x, du, du.abs()], dim=-1)
                msg = self.sp_msg_adj(msg_in)
            else:
                u_j = u_pad[:, k_x + j : k_x + j + nx].unsqueeze(-1)
                u_i_exp = u_n.unsqueeze(-1)
                f_i = u_i_exp * (1.0 - u_i_exp)
                f_j = u_j * (1.0 - u_j)
                a_i = 1.0 - 2.0 * u_i_exp
                a_j = 1.0 - 2.0 * u_j
                msg_in = torch.cat([
                    h, h_j, u_i_exp, u_j, f_i, f_j, a_i, a_j, x_i_exp, x_j,
                ], dim=-1)
                msg = self.sp_msg_nonadj(msg_in)

            contrib = msg
            if self.radius_x is not None:
                contrib = contrib * (rel_x.abs() <= self.radius_x)
            sp_agg = sp_agg + contrib

        h_nonlocal = self.update_net(torch.cat([h, sp_agg], dim=-1))
        h_local = self.W(h)
        return self.act(h_nonlocal + h_local)


class _PhysicsSpatialMP(nn.Module):
    """Spatial MP with analytical physics gate (upwind + Oleinik + optional char-cone)."""

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        k_x: int,
        activation: str,
        radius_x: float | None = None,
        use_char_cone: bool = False,
        d_hidden_nonadj: int | None = None,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.radius_x = radius_x
        self.use_char_cone = use_char_cone
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        self.state_probe = nn.Linear(d_latent, 1)
        self.phys_temperature = nn.Parameter(torch.tensor(0.0))
        self.phys_gamma_entropy = nn.Parameter(torch.tensor(-2.0))
        if use_char_cone:
            self.phys_char_width = nn.Parameter(torch.tensor(0.0))

        # adj spatial: 2d + 12 (h_i, h_j, rel_x, u_i, u_j, du, u_avg, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind)
        self.sp_msg_adj = _make_mlp(2 * d_latent + 12, d_hidden, d_latent, 3, activation)
        # non-adj spatial: 2d + 8
        self.sp_msg_nonadj = _make_mlp(2 * d_latent + 8, dh_na, d_latent, 3, activation)

        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    def _gate_adj(
        self, u_i: torch.Tensor, u_j: torch.Tensor, rel_x: torch.Tensor,
        a_ij: torch.Tensor, a_j: torch.Tensor, dt: float, dx_grid: float,
    ) -> torch.Tensor:
        temperature = F.softplus(self.phys_temperature).clamp(min=1e-6)
        gamma_ent = torch.sigmoid(self.phys_gamma_entropy)

        g_upwind = torch.sigmoid(-a_ij * rel_x / temperature)

        u_L = torch.where(rel_x > 0, u_i, u_j)
        u_R = torch.where(rel_x > 0, u_j, u_i)
        a_L = 1.0 - 2.0 * u_L
        a_R = 1.0 - 2.0 * u_R
        is_shock = (u_L > u_R).float()
        entropy_ok = ((a_L >= a_ij - 0.01) & (a_ij >= a_R - 0.01)).float()
        g_entropy = 1.0 - is_shock * (1.0 - entropy_ok) * (1.0 - gamma_ent)

        gate = g_upwind * g_entropy

        if self.use_char_cone:
            char_w = F.softplus(self.phys_char_width).clamp(min=1e-6)
            char_miss = (rel_x + a_j * dt).abs()
            sigma = char_w * (dx_grid + a_j.abs() * abs(dt))
            gate = gate * torch.exp(-0.5 * (char_miss / sigma.clamp(min=1e-6)) ** 2)
        return gate

    def _gate_nonadj(
        self, rel_x: torch.Tensor, a_j: torch.Tensor, dt: float, dx_grid: float,
    ) -> torch.Tensor:
        if not self.use_char_cone:
            return torch.ones_like(rel_x)
        char_w = F.softplus(self.phys_char_width).clamp(min=1e-6)
        char_miss = (rel_x + a_j * dt).abs()
        sigma = char_w * (dx_grid + a_j.abs() * abs(dt))
        return torch.exp(-0.5 * (char_miss / sigma.clamp(min=1e-6)) ** 2)

    def forward(
        self, h: torch.Tensor, x: torch.Tensor, u_n: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        B, nx, d = h.shape
        dx_val = (x[0, 1] - x[0, 0]).abs().item()
        k_x = max(1, int(self.radius_x / dx_val + 0.5)) if self.radius_x is not None else self.k_x

        u_hat = torch.sigmoid(self.state_probe(h)).squeeze(-1)
        u_hat_i = u_hat.unsqueeze(-1)

        h_pad = F.pad(h.permute(0, 2, 1), (k_x, k_x), mode="replicate").permute(0, 2, 1)
        u_hat_pad = F.pad(u_hat.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        x_i = x.unsqueeze(-1)

        sp_agg = h.new_zeros(B, nx, d)
        for j in range(-k_x, k_x + 1):
            is_adjacent = abs(j) <= 1
            h_j = h_pad[:, k_x + j : k_x + j + nx, :]
            x_j = x_pad[:, k_x + j : k_x + j + nx].unsqueeze(-1)
            u_hat_j = u_hat_pad[:, k_x + j : k_x + j + nx].unsqueeze(-1)
            rel_x = x_j - x_i

            if is_adjacent:
                du, u_avg, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind = \
                    _compute_adj_edge_feats(u_hat_i, u_hat_j, rel_x)
                msg_in = torch.cat([
                    h, h_j, rel_x,
                    u_hat_i.expand_as(rel_x), u_hat_j,
                    du, u_avg, f_i, f_j, a_i, a_j, a_ij,
                    sign_a, upwind,
                ], dim=-1)
                msg = self.sp_msg_adj(msg_in)
                gate = self._gate_adj(
                    u_i=u_hat_i, u_j=u_hat_j, rel_x=rel_x,
                    a_ij=a_ij, a_j=a_j, dt=dt, dx_grid=dx_val,
                )
            else:
                f_i, f_j, a_i, a_j = _compute_nonadj_pair_feats(u_hat_i, u_hat_j)
                msg_in = torch.cat([
                    h, h_j, u_hat_i.expand_as(rel_x), u_hat_j,
                    f_i, f_j, a_i, a_j, x_i.expand_as(x_j), x_j,
                ], dim=-1)
                msg = self.sp_msg_nonadj(msg_in)
                gate = self._gate_nonadj(
                    rel_x=rel_x, a_j=a_j, dt=dt, dx_grid=dx_val,
                )

            contrib = msg * gate
            if self.radius_x is not None:
                contrib = contrib * (rel_x.abs() <= self.radius_x)
            sp_agg = sp_agg + contrib

        h_nonlocal = self.update_net(torch.cat([h, sp_agg], dim=-1))
        h_local = self.W(h)
        return self.act(h_nonlocal + h_local)


class _WENOSpatialMP(nn.Module):
    """Spatial MP with WENO smoothness weighting over u_hat."""

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        k_x: int,
        activation: str,
        weno_eps: float = 1e-6,
        weno_p: float = 2.0,
        radius_x: float | None = None,
        d_hidden_nonadj: int | None = None,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.radius_x = radius_x
        self.weno_eps = weno_eps
        self.weno_p = weno_p
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        self.state_probe = nn.Linear(d_latent, 1)
        self.sp_msg_adj = _make_mlp(2 * d_latent + 12, d_hidden, d_latent, 3, activation)
        self.sp_msg_nonadj = _make_mlp(2 * d_latent + 8, dh_na, d_latent, 3, activation)

        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    @staticmethod
    def _scalar_beta(u: torch.Tensor) -> torch.Tensor:
        diff_fwd = F.pad(u[:, 1:] - u[:, :-1], (0, 1))
        diff_bwd = F.pad(u[:, :-1] - u[:, 1:], (1, 0))
        return diff_fwd ** 2 + diff_bwd ** 2

    def forward(
        self, h: torch.Tensor, x: torch.Tensor, u_n: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        B, nx, d = h.shape
        dx_val = (x[0, 1] - x[0, 0]).abs().item()
        k_x = max(1, int(self.radius_x / dx_val + 0.5)) if self.radius_x is not None else self.k_x
        use_weno = self.weno_p > 0

        u_hat = torch.sigmoid(self.state_probe(h)).squeeze(-1)
        u_hat_i = u_hat.unsqueeze(-1)

        h_pad = F.pad(h.permute(0, 2, 1), (k_x, k_x), mode="replicate").permute(0, 2, 1)
        u_hat_pad = F.pad(u_hat.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        x_i = x.unsqueeze(-1)

        if use_weno:
            beta = self._scalar_beta(u_hat)
            beta_pad = F.pad(beta, (k_x, k_x), mode="replicate")
            omega_raw: list[torch.Tensor] = []
            for j in range(-k_x, k_x + 1):
                beta_j = beta_pad[:, k_x + j : k_x + j + nx]
                omega_j = 1.0 / (self.weno_eps + beta_j).pow(self.weno_p)
                omega_raw.append(omega_j)
            omega_sum = torch.stack(omega_raw, dim=0).sum(dim=0).clamp(min=1e-8)

        sp_agg = h.new_zeros(B, nx, d)
        for idx, j in enumerate(range(-k_x, k_x + 1)):
            is_adjacent = abs(j) <= 1
            h_j = h_pad[:, k_x + j : k_x + j + nx, :]
            x_j = x_pad[:, k_x + j : k_x + j + nx].unsqueeze(-1)
            u_hat_j = u_hat_pad[:, k_x + j : k_x + j + nx].unsqueeze(-1)
            rel_x = x_j - x_i

            if is_adjacent:
                du, u_avg, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind = \
                    _compute_adj_edge_feats(u_hat_i, u_hat_j, rel_x)
                msg_in = torch.cat([
                    h, h_j, rel_x,
                    u_hat_i.expand_as(rel_x), u_hat_j,
                    du, u_avg, f_i, f_j, a_i, a_j, a_ij,
                    sign_a, upwind,
                ], dim=-1)
                msg = self.sp_msg_adj(msg_in)
            else:
                f_i, f_j, a_i, a_j = _compute_nonadj_pair_feats(u_hat_i, u_hat_j)
                msg_in = torch.cat([
                    h, h_j, u_hat_i.expand_as(rel_x), u_hat_j,
                    f_i, f_j, a_i, a_j, x_i.expand_as(x_j), x_j,
                ], dim=-1)
                msg = self.sp_msg_nonadj(msg_in)

            if use_weno:
                omega_norm = (omega_raw[idx] / omega_sum).unsqueeze(-1)
                contrib = msg * omega_norm
            else:
                contrib = msg

            if self.radius_x is not None:
                contrib = contrib * (rel_x.abs() <= self.radius_x)
            sp_agg = sp_agg + contrib

        h_nonlocal = self.update_net(torch.cat([h, sp_agg], dim=-1))
        h_local = self.W(h)
        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class HypNO_ST4(nn.Module):
    """Time-marching spatial operator: u^n -> u^{n+1}.

    Forward signature takes a single frame u^n. Full-trajectory rollout is
    driven externally by the training / eval scripts via iterated calls.
    """

    def __init__(
        self,
        stencil_k_x: int = 3,
        d_latent: int = 64,
        d_hidden: int = 96,
        n_layers: int = 6,
        activation: str = "gelu",
        shock_mode: str = "physics",            # "physics" | "classic" | "weno"
        use_char_cone: bool = False,
        weno_eps: float = 1e-6,
        weno_p: float = 2.0,
        encoder_type: str = "gnn",              # "gnn" | "mlp"
        encoder_scaling: str = "physics",       # "gate_net" | "upwind" | "physics"
        readout: str = "gelu",
        radius_x: float | None = None,
        d_hidden_nonadj: int | None = None,
    ) -> None:
        super().__init__()
        self.stencil_k_x = stencil_k_x
        self.radius_x = radius_x
        self.shock_mode = shock_mode
        self.use_char_cone = use_char_cone

        self.lifting = _SpatialLifting(
            d_latent=d_latent, d_hidden=d_hidden, stencil_k=stencil_k_x,
            activation=activation, radius_x=radius_x,
            encoder_scaling=encoder_scaling, encoder_type=encoder_type,
            use_char_cone=use_char_cone, d_hidden_nonadj=d_hidden_nonadj,
        )

        if shock_mode == "physics":
            self.mp_layers = nn.ModuleList([
                _PhysicsSpatialMP(
                    d_latent, d_hidden, stencil_k_x, activation,
                    radius_x=radius_x, use_char_cone=use_char_cone,
                    d_hidden_nonadj=d_hidden_nonadj,
                )
                for _ in range(n_layers)
            ])
        elif shock_mode == "classic":
            self.mp_layers = nn.ModuleList([
                _ClassicSpatialMP(
                    d_latent, d_hidden, stencil_k_x, activation,
                    radius_x=radius_x, d_hidden_nonadj=d_hidden_nonadj,
                )
                for _ in range(n_layers)
            ])
        elif shock_mode == "weno":
            self.mp_layers = nn.ModuleList([
                _WENOSpatialMP(
                    d_latent, d_hidden, stencil_k_x, activation,
                    weno_eps=weno_eps, weno_p=weno_p,
                    radius_x=radius_x, d_hidden_nonadj=d_hidden_nonadj,
                )
                for _ in range(n_layers)
            ])
        else:
            raise ValueError(f"Unknown shock_mode: {shock_mode!r}")

        self.decoder = _make_mlp(d_latent, d_hidden, 1, 3, readout)

    def forward(
        self, u_n: torch.Tensor, x: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        u_n : [B, nx] current state
        x   : [nx] or [B, nx] spatial grid
        dt  : scalar time step used by the char-cone gate (only read when use_char_cone=True)

        Returns
        -------
        u_next : [B, nx] predicted next state
        u_hats : list of [B, nx] per-layer state probes (deep supervision)
        """
        B, nx = u_n.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)

        h = self.lifting(u_n, x, dt)

        u_hats: list[torch.Tensor] = []
        for layer in self.mp_layers:
            h = layer(h, x, u_n, dt=dt)
            if hasattr(layer, "state_probe"):
                u_hats.append(torch.sigmoid(layer.state_probe(h)).squeeze(-1))

        delta = self.decoder(h).squeeze(-1)
        u_next = u_n + delta
        return u_next, u_hats
