"""HypNO-ST 2D v0 — constant-coefficient linear advection.

Solves the linear scalar transport equation

    u_t + a * u_x + b * u_y = 0

with constant `(a, b)` (default a = 1, b = 0.7). The exact entropy solution is
`u(x, y, t) = u_0(x - a*t, y - b*t)` — no shocks form, characteristics never
cross. v0 exists as a pre-LWR validation step for the 2D graph construction.

Design (per `hypno_2d_v0_plan.md`, locked with user decisions):

* **Three separate message MLPs** routed by adjacency class — `adj_x_msg`,
  `adj_y_msg`, `nonadj_msg`. Matches spec-2 §1.
* **Per-class gate-normalised aggregation** (spec-2 §2-4): each class sums to
  one independently, then the three per-class aggregates are concatenated
  side-by-side before update_net (override of spec-2 §5: no fixed `α_x, α_y, η`
  multipliers — `update_net` learns the relative weighting).
* **Gate values are NOT features** (override of spec-2 putting them in the
  edge vector). Gates act only as multiplicative aggregation weights.
* **Entropy gate ≡ 1** for v0 linear (no nonlinear shocks). Structural slot
  preserved for v1 nonlinear LWR, where the sign convention matches
  `hypno_st3.py:_ball_physics_gate` (`is_shock = (u_L < u_R)` for concave
  LWR; `g_entropy = 1 - is_shock * (1 - entropy_ok) * (1 - γ_ent)` —
  suppresses inadmissible expansion shocks).
* **Temporal gate**: characteristic-alignment cone-style
  `g_temp = exp(-κ ‖(Δr - (a,b)Δt) / h_s‖²)`, applied only on non-adjacent
  edges (spec-2 §4 chose this over the broken edge-length-additive CFL of
  spec-1 §3).
* **Boundary**: ghost-cell replicate everywhere (model padding + data gen).

Node MLP input (12 dims, spec-1 §"Lifting"):

    u_0_{i,j},  x_i, y_j, t_n,  a, b, λ_x, λ_y,
    F̂^x_{i-1/2,j}(u^0),  F̂^x_{i+1/2,j}(u^0),
    F̂^y_{i,j-1/2}(u^0),  F̂^y_{i,j+1/2}(u^0)

with `λ_x = a, λ_y = b` (kept as separate features for forward-compat with
variable-coefficient extensions).

Edge feature vectors (gates dropped):

    adj_x:  [du_0, a, sgn(x_k - x_i), χ^x_up]                (4 dims)
    adj_y:  [du_0, b, sgn(y_p - y_j), χ^y_up]                (4 dims)
    nonadj: [du_0, Δx, Δy, Δt, a, b, χ^non_up]               (7 dims)

MP-layer edge feature vectors:

    adj_x:  [h_i, h_j, du, a, sgn(x_k - x_i), χ^x_up]        (2d + 4)
    adj_y:  [h_i, h_j, du, b, sgn(y_p - y_j), χ^y_up]        (2d + 4)
    nonadj: [h_i, h_j, du, Δx, Δy, Δt, a, b, χ^non_up]       (2d + 7)

Aggregation per class:

    g^x_k        = σ((a · (x_i - x_k)) / (τ_x · |x_i - x_k| + ε))
    M^x_{i,j,n}  = Σ_{k} (g^x_k / (Σ_j g^x_j + 1e-3)) · m^x_k

(analogous for y; for non-adj `g^non = g^non_up · g_temp`.)

Update:

    h^{ℓ+1} = act( update_net([h, M^x, M^y, M^non]) + W·h )
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# --------------------------------------------------------------------------- #
# Helpers (mostly forked from hypno_st3_2d.py — kept local so the v0 module
# is self-contained against future drift in that file).
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


def _enumerate_ball_offsets_2d(
    k_x: int, k_y: int, k_t: int, causal: bool,
) -> list[tuple[int, int, int]]:
    """Enumerate every (di_x, di_y, dm) in the 3D Chebyshev ball, excluding centre."""
    m_range = range(-k_t, 1) if causal else range(-k_t, k_t + 1)
    out: list[tuple[int, int, int]] = []
    for dm in m_range:
        for dy in range(-k_y, k_y + 1):
            for dx in range(-k_x, k_x + 1):
                if dx == 0 and dy == 0 and dm == 0:
                    continue
                out.append((dx, dy, dm))
    return out


def _pad_space_time_2d(h: torch.Tensor, k_x: int, k_y: int, k_t: int) -> torch.Tensor:
    """Replicate-pad h [B, nt, ny, nx, d] on the three grid axes."""
    h_pad = F.pad(
        h.permute(0, 4, 1, 2, 3),                      # [B, d, nt, ny, nx]
        (k_x, k_x, k_y, k_y, k_t, k_t),
        mode="replicate",
    ).permute(0, 2, 3, 4, 1)                           # [B, nt+2k_t, ny+2k_y, nx+2k_x, d]
    return h_pad


def _adjacency_class(di_x: int, di_y: int, dm: int) -> str:
    """Classify a ball offset. 'adj_x' / 'adj_y' = same-time face neighbours,
    'nonadj' = everything else (temporal, diagonal, long-range)."""
    if dm == 0 and abs(di_x) == 1 and di_y == 0:
        return "adj_x"
    if dm == 0 and di_x == 0 and abs(di_y) == 1:
        return "adj_y"
    return "nonadj"


# --------------------------------------------------------------------------- #
# Numerical fluxes used by the lifting node features (mirrors
# `hyperbolic_pde.data.fvm_2d.upwind_flux_x/y` but works on torch tensors).
# --------------------------------------------------------------------------- #
def _upwind_flux_x_torch(u: torch.Tensor, a: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (F^x_{i-1/2,j}, F^x_{i+1/2,j}) using first-order upwind on u [B, ny, nx].

    Replicate boundary on the upwind side. Used for the lifting node features.
    """
    if a >= 0:
        u_left = torch.cat([u[..., :, :1], u[..., :, :-1]], dim=-1)
        return a * u_left, a * u
    u_right = torch.cat([u[..., :, 1:], u[..., :, -1:]], dim=-1)
    return a * u, a * u_right


def _upwind_flux_y_torch(u: torch.Tensor, b: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (F^y_{i,j-1/2}, F^y_{i,j+1/2}) using first-order upwind on u [B, ny, nx]."""
    if b >= 0:
        u_below = torch.cat([u[..., :1, :], u[..., :-1, :]], dim=-2)
        return b * u_below, b * u
    u_above = torch.cat([u[..., 1:, :], u[..., -1:, :]], dim=-2)
    return b * u, b * u_above


# --------------------------------------------------------------------------- #
# Gate primitives (linear advection, v0)
# --------------------------------------------------------------------------- #
def _soft_upwind_axis(
    speed: float, rel: torch.Tensor, tau: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """Soft upwind gate on an axial edge.

    For x-adjacent edges, speed = a and rel = x_i - x_k = -rel_x_from_target.
    Note the spec writes the gate as σ(a·(x_i - x_k) / (τ·|x_i - x_k| + ε)).
    Since |x_i - x_k| = dx (positive scalar), this reduces to σ(a·sgn(x_i-x_k)/τ),
    which is high when the source is upwind of the target (information flow).
    """
    return torch.sigmoid(speed * rel / (tau * rel.abs() + eps))


def _soft_upwind_2d(
    a: float, b: float,
    rel_x_target_minus_source: torch.Tensor,
    rel_y_target_minus_source: torch.Tensor,
    tau: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft upwind gate for non-adjacent edges: σ((aΔx + bΔy) / (τ·|Δr| + ε)).

    `rel_*_target_minus_source = x_i - x_k`; gate is high when the
    source-to-target direction aligns with the wave velocity (a, b), i.e.
    the source lies upstream of the target.
    """
    dot = a * rel_x_target_minus_source + b * rel_y_target_minus_source
    norm = torch.sqrt(
        rel_x_target_minus_source ** 2 + rel_y_target_minus_source ** 2 + eps ** 2
    )
    return torch.sigmoid(dot / (tau * norm + eps))


def _char_alignment_gate(
    a: float, b: float,
    rel_x_target_minus_source: torch.Tensor,
    rel_y_target_minus_source: torch.Tensor,
    rel_t_target_minus_source: torch.Tensor,
    kappa: torch.Tensor,
    h_s: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Characteristic-alignment temporal/CFL gate (spec-2 §4):

        g_temp = exp(-κ ‖(Δr - (a, b) Δt) / h_s‖²)

    Peaks when the edge displacement matches what a characteristic of the
    PDE could traverse in time Δt — i.e. the source lies on the backward
    characteristic from the target. Δr = (x_i - x_k, y_j - y_p),
    Δt = t_n - t_m (target-minus-source convention).
    """
    miss_x = rel_x_target_minus_source - a * rel_t_target_minus_source
    miss_y = rel_y_target_minus_source - b * rel_t_target_minus_source
    sq = (miss_x ** 2 + miss_y ** 2) / (h_s ** 2 + eps)
    return torch.exp(-kappa * sq)


# --------------------------------------------------------------------------- #
# Lifting layer
# --------------------------------------------------------------------------- #
class _LinearAdvectionLiftingLayer2D(nn.Module):
    """Lifting for v0 linear advection.

    Node MLP consumes the 12-dim per-node feature vector (u, x, y, t, a, b,
    λ_x, λ_y, F̂^x_{i±1/2}, F̂^y_{j±1/2}). Then three per-class edge MLPs
    consume pair-intrinsic edge features and the per-class gate-normalised
    aggregates are concatenated side-by-side and combined with the node
    feature via a small combine MLP.
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        stencil_k_x: int,
        stencil_k_y: int,
        stencil_k_t: int,
        a: float,
        b: float,
        h_s: float,
        activation: str = "gelu",
        encoder_type: str = "gnn",
        causal_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.h_s = float(h_s)
        self.k_x = stencil_k_x
        self.k_y = stencil_k_y
        self.k_t = stencil_k_t
        self.encoder_type = encoder_type
        self.causal = causal_temporal

        self.node_mlp = _make_mlp(12, d_hidden, d_latent, 2, activation)
        print(
            "[Lifting-Lin2D] node_mlp in=12 "
            "(u, x, y, t, a, b, lam_x, lam_y, Fx_left, Fx_right, Fy_below, Fy_above)"
        )
        if encoder_type == "mlp":
            return

        # Three per-class edge MLPs. Pair-intrinsic features only (no
        # broadcast of u_i / u_j separately — du0 carries the pair signal).
        self.adj_x_msg = _make_mlp(4, d_hidden, d_latent, 2, activation)
        self.adj_y_msg = _make_mlp(4, d_hidden, d_latent, 2, activation)
        self.nonadj_msg = _make_mlp(7, d_hidden, d_latent, 2, activation)
        print(
            "[Lifting-Lin2D] edge MLPs (pure-pairwise): "
            "adj_x in=4, adj_y in=4, nonadj in=7"
        )

        # Learnable gate temperatures (softplus + small floor).
        self.tau_x = nn.Parameter(torch.tensor(0.0))
        self.tau_y = nn.Parameter(torch.tensor(0.0))
        self.tau_non = nn.Parameter(torch.tensor(0.0))
        self.kappa_temp = nn.Parameter(torch.tensor(0.0))

        # Combine [h_node, M^x, M^y, M^non] -> d_latent.
        self.combine = _make_mlp(4 * d_latent, d_hidden, d_latent, 2, activation)

    def forward(
        self,
        u0: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        u0 : [B, ny, nx]
        x  : [nx]
        y  : [ny]
        t  : [nt]
        """
        B, ny, nx = u0.shape
        nt = t.shape[0]

        # --------- Node features (12-dim) --------- #
        Fx_left, Fx_right = _upwind_flux_x_torch(u0, self.a)          # [B, ny, nx]
        Fy_below, Fy_above = _upwind_flux_y_torch(u0, self.b)         # [B, ny, nx]

        u0_bc = u0.view(B, 1, ny, nx, 1).expand(B, nt, ny, nx, 1)
        x_bc = x.view(1, 1, 1, nx, 1).expand(B, nt, ny, nx, 1)
        y_bc = y.view(1, 1, ny, 1, 1).expand(B, nt, ny, nx, 1)
        t_bc = t.view(1, nt, 1, 1, 1).expand(B, nt, ny, nx, 1)
        a_bc = u0_bc.new_full(u0_bc.shape, self.a)
        b_bc = u0_bc.new_full(u0_bc.shape, self.b)
        lam_x_bc = a_bc  # λ_x = a for linear advection
        lam_y_bc = b_bc  # λ_y = b
        Fxl_bc = Fx_left.view(B, 1, ny, nx, 1).expand(B, nt, ny, nx, 1)
        Fxr_bc = Fx_right.view(B, 1, ny, nx, 1).expand(B, nt, ny, nx, 1)
        Fyb_bc = Fy_below.view(B, 1, ny, nx, 1).expand(B, nt, ny, nx, 1)
        Fya_bc = Fy_above.view(B, 1, ny, nx, 1).expand(B, nt, ny, nx, 1)

        node_in = torch.cat([
            u0_bc, x_bc, y_bc, t_bc,
            a_bc, b_bc, lam_x_bc, lam_y_bc,
            Fxl_bc, Fxr_bc, Fyb_bc, Fya_bc,
        ], dim=-1)                                                     # [..., 12]
        h_node = self.node_mlp(node_in)

        if self.encoder_type == "mlp":
            if return_intermediates:
                return h_node, h_node, h_node
            return h_node

        k_x, k_y, k_t = self.k_x, self.k_y, self.k_t

        # Pad u0 spatially; coordinate vectors get the matching pad.
        u0_pad = F.pad(u0, (k_x, k_x, k_y, k_y), mode="replicate")     # [B, ny+2k_y, nx+2k_x]
        x_pad = F.pad(x.view(1, 1, nx), (k_x, k_x), mode="replicate").view(-1)
        y_pad = F.pad(y.view(1, 1, ny), (k_y, k_y), mode="replicate").view(-1)
        t_pad = F.pad(t.view(1, 1, nt), (k_t, k_t), mode="replicate").view(-1)

        offsets = _enumerate_ball_offsets_2d(k_x, k_y, k_t, self.causal)

        adj_x_feats: list[torch.Tensor] = []
        adj_y_feats: list[torch.Tensor] = []
        nonadj_feats: list[torch.Tensor] = []
        adj_x_gates: list[torch.Tensor] = []
        adj_y_gates: list[torch.Tensor] = []
        nonadj_gates: list[torch.Tensor] = []

        tau_x = F.softplus(self.tau_x).clamp(min=1e-6)
        tau_y = F.softplus(self.tau_y).clamp(min=1e-6)
        tau_non = F.softplus(self.tau_non).clamp(min=1e-6)
        kappa = F.softplus(self.kappa_temp).clamp(min=1e-6)

        for di_x, di_y, dm in offsets:
            cls = _adjacency_class(di_x, di_y, dm)

            u0_j = u0_pad[
                :, k_y + di_y : k_y + di_y + ny, k_x + di_x : k_x + di_x + nx,
            ].view(B, 1, ny, nx, 1).expand(B, nt, ny, nx, 1)
            x_j = x_pad[k_x + di_x : k_x + di_x + nx].view(1, 1, 1, nx, 1).expand(B, nt, ny, nx, 1)
            y_j = y_pad[k_y + di_y : k_y + di_y + ny].view(1, 1, ny, 1, 1).expand(B, nt, ny, nx, 1)
            t_j = t_pad[k_t + dm : k_t + dm + nt].view(1, nt, 1, 1, 1).expand(B, nt, ny, nx, 1)

            rel_x_neighbour_minus_target = x_j - x_bc  # x_k - x_i; sign feature
            rel_y_neighbour_minus_target = y_j - y_bc
            rel_t = t_j - t_bc                          # Δt = t_n - t_m? careful:
            # convention: target=i,j,n; source=k,p,m. Δt = t_n - t_m = t_i - t_j.
            # rel_t as defined above = t_j - t_bc = t_m - t_n = -Δt. So we
            # negate to keep the spec's convention.
            rel_t_target_minus_source = -rel_t
            rel_x_target_minus_source = -rel_x_neighbour_minus_target
            rel_y_target_minus_source = -rel_y_neighbour_minus_target

            du0 = u0_j - u0_bc                          # pair-intrinsic state diff

            if cls == "adj_x":
                # Upwind flag: χ^x_up = 1[a (x_i - x_k) > 0].
                chi = (self.a * rel_x_target_minus_source > 0).float()
                feat = torch.cat([
                    du0,
                    a_bc,
                    torch.sign(rel_x_neighbour_minus_target),  # sgn(x_k - x_i)
                    chi,
                ], dim=-1)                                                # 4-dim
                gate = _soft_upwind_axis(self.a, rel_x_target_minus_source, tau_x)
                adj_x_feats.append(feat)
                adj_x_gates.append(gate)

            elif cls == "adj_y":
                chi = (self.b * rel_y_target_minus_source > 0).float()
                feat = torch.cat([
                    du0,
                    b_bc,
                    torch.sign(rel_y_neighbour_minus_target),
                    chi,
                ], dim=-1)                                                # 4-dim
                gate = _soft_upwind_axis(self.b, rel_y_target_minus_source, tau_y)
                adj_y_feats.append(feat)
                adj_y_gates.append(gate)

            else:  # nonadj
                chi = (
                    self.a * rel_x_target_minus_source
                    + self.b * rel_y_target_minus_source
                    > 0
                ).float()
                feat = torch.cat([
                    du0,
                    rel_x_target_minus_source, rel_y_target_minus_source,
                    rel_t_target_minus_source,
                    a_bc, b_bc,
                    chi,
                ], dim=-1)                                                # 7-dim
                g_up = _soft_upwind_2d(
                    self.a, self.b,
                    rel_x_target_minus_source, rel_y_target_minus_source,
                    tau_non,
                )
                g_tmp = _char_alignment_gate(
                    self.a, self.b,
                    rel_x_target_minus_source, rel_y_target_minus_source,
                    rel_t_target_minus_source,
                    kappa, self.h_s,
                )
                gate = g_up * g_tmp
                nonadj_feats.append(feat)
                nonadj_gates.append(gate)

        # --- per-class gate-normalised aggregation ---
        def _agg_class(
            feats: list[torch.Tensor], gates: list[torch.Tensor],
            mlp: nn.Module,
        ) -> torch.Tensor | None:
            if not feats:
                return None
            stacked_feat = torch.stack(feats, dim=4)                       # [B,nt,ny,nx,n,F]
            n = stacked_feat.shape[4]
            stacked_gate = torch.stack(gates, dim=4)                       # [B,nt,ny,nx,n,1]
            msgs = mlp(
                stacked_feat.reshape(-1, stacked_feat.shape[-1])
            ).reshape(B, nt, ny, nx, n, -1)                                # [...,n,d]
            gate_sum = stacked_gate.sum(dim=4) + 1e-3                      # [...,1]
            return ((stacked_gate / gate_sum.unsqueeze(4)) * msgs).sum(dim=4)

        d_latent = h_node.shape[-1]
        zero_agg = h_node.new_zeros((B, nt, ny, nx, d_latent))
        M_x = _agg_class(adj_x_feats, adj_x_gates, self.adj_x_msg)
        M_y = _agg_class(adj_y_feats, adj_y_gates, self.adj_y_msg)
        M_non = _agg_class(nonadj_feats, nonadj_gates, self.nonadj_msg)
        M_x = zero_agg if M_x is None else M_x
        M_y = zero_agg if M_y is None else M_y
        M_non = zero_agg if M_non is None else M_non

        agg = torch.cat([M_x, M_y, M_non], dim=-1)                         # [...,3d]
        h_full = self.combine(torch.cat([h_node, agg], dim=-1))            # combine: 4d -> d
        if return_intermediates:
            return h_full, h_node, h_full
        return h_full


# --------------------------------------------------------------------------- #
# Message-passing layer
# --------------------------------------------------------------------------- #
class _LinearAdvectionMPLayer2D(nn.Module):
    """One 2D message-passing layer for v0 linear advection.

    Three per-class edge MLPs (adj_x, adj_y, nonadj). Each takes the per-edge
    feature vector concatenated with the source/target latents h_i, h_j.
    Gate-normalised aggregation per class. Per-class aggregates are
    concatenated side-by-side before update_net (no α_x, α_y, η scaling —
    deliberate departure from spec-2 §5; update_net learns the weighting).
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        k_x: int,
        k_y: int,
        k_t: int,
        a: float,
        b: float,
        h_s: float,
        activation: str = "gelu",
        causal_temporal: bool = True,
        d_hidden_nonadj: int | None = None,
    ) -> None:
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.h_s = float(h_s)
        self.k_x = k_x
        self.k_y = k_y
        self.k_t = k_t
        self.causal = causal_temporal
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        # 2*d_latent for (h_i, h_j) + per-class feature dim.
        self.adj_x_msg = _make_mlp(2 * d_latent + 4, d_hidden, d_latent, 3, activation)
        self.adj_y_msg = _make_mlp(2 * d_latent + 4, d_hidden, d_latent, 3, activation)
        self.nonadj_msg = _make_mlp(2 * d_latent + 7, dh_na, d_latent, 3, activation)
        print(
            f"[MP-Lin2D] adj_x in=2*{d_latent}+4={2 * d_latent + 4} | "
            f"adj_y in=2*{d_latent}+4={2 * d_latent + 4} | "
            f"nonadj in=2*{d_latent}+7={2 * d_latent + 7}"
        )

        # Learnable gate temperatures.
        self.tau_x = nn.Parameter(torch.tensor(0.0))
        self.tau_y = nn.Parameter(torch.tensor(0.0))
        self.tau_non = nn.Parameter(torch.tensor(0.0))
        self.kappa_temp = nn.Parameter(torch.tensor(0.0))

        # update_net consumes [h, M^x, M^y, M^non] = 4d_latent.
        self.update_net = _make_mlp(4 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        u0: torch.Tensor,
    ) -> torch.Tensor:
        B, nt, ny, nx, d = h.shape
        k_x, k_y, k_t = self.k_x, self.k_y, self.k_t

        h_pad = _pad_space_time_2d(h, k_x, k_y, k_t)
        # u0 is t-invariant so we only need spatial padding (hoisted once).
        u0_pad_b = F.pad(u0, (k_x, k_x, k_y, k_y), mode="replicate")
        u0_i_5d = u0.view(B, 1, ny, nx, 1).expand(B, nt, ny, nx, 1)
        x_pad = F.pad(x.view(1, 1, nx), (k_x, k_x), mode="replicate").view(-1)
        y_pad = F.pad(y.view(1, 1, ny), (k_y, k_y), mode="replicate").view(-1)
        t_pad = F.pad(t.view(1, 1, nt), (k_t, k_t), mode="replicate").view(-1)

        x_i = x.view(1, 1, 1, nx, 1).expand(B, nt, ny, nx, 1)
        y_i = y.view(1, 1, ny, 1, 1).expand(B, nt, ny, nx, 1)
        t_i = t.view(1, nt, 1, 1, 1).expand(B, nt, ny, nx, 1)
        a_bc = h.new_full((B, nt, ny, nx, 1), self.a)
        b_bc = h.new_full((B, nt, ny, nx, 1), self.b)

        offsets = _enumerate_ball_offsets_2d(k_x, k_y, k_t, self.causal)

        adj_x_feats: list[torch.Tensor] = []
        adj_y_feats: list[torch.Tensor] = []
        nonadj_feats: list[torch.Tensor] = []
        adj_x_gates: list[torch.Tensor] = []
        adj_y_gates: list[torch.Tensor] = []
        nonadj_gates: list[torch.Tensor] = []

        tau_x = F.softplus(self.tau_x).clamp(min=1e-6)
        tau_y = F.softplus(self.tau_y).clamp(min=1e-6)
        tau_non = F.softplus(self.tau_non).clamp(min=1e-6)
        kappa = F.softplus(self.kappa_temp).clamp(min=1e-6)

        for di_x, di_y, dm in offsets:
            cls = _adjacency_class(di_x, di_y, dm)

            h_j = h_pad[
                :, k_t + dm : k_t + dm + nt,
                k_y + di_y : k_y + di_y + ny,
                k_x + di_x : k_x + di_x + nx, :,
            ]
            x_j = x_pad[k_x + di_x : k_x + di_x + nx].view(1, 1, 1, nx, 1).expand(B, nt, ny, nx, 1)
            y_j = y_pad[k_y + di_y : k_y + di_y + ny].view(1, 1, ny, 1, 1).expand(B, nt, ny, nx, 1)
            t_j = t_pad[k_t + dm : k_t + dm + nt].view(1, nt, 1, 1, 1).expand(B, nt, ny, nx, 1)

            # Target-minus-source displacement (spec convention).
            dx_t = x_i - x_j
            dy_t = y_i - y_j
            dt_t = t_i - t_j

            # `du` here is on latent-state proxy: use u0 difference (no decoded
            # state needed in v0 — the linear flux makes that unnecessary).
            # Source u0 is the spatial neighbour at the same x_j, y_j (u0
            # doesn't depend on t).
            u0_j = u0[
                torch.arange(B, device=u0.device)[:, None, None, None],
                # placeholder broadcasting — replaced below
            ]  # not used; we slice u0 separately for du below
            # Simpler: rebuild u0_j from u0 via the same spatial offset.
            u0_pad_b = F.pad(u0, (k_x, k_x, k_y, k_y), mode="replicate")
            u0_j_2d = u0_pad_b[
                :, k_y + di_y : k_y + di_y + ny, k_x + di_x : k_x + di_x + nx,
            ]                                                              # [B, ny, nx]
            u0_j_5d = u0_j_2d.view(B, 1, ny, nx, 1).expand(B, nt, ny, nx, 1)
            u0_i_5d = u0.view(B, 1, ny, nx, 1).expand(B, nt, ny, nx, 1)
            du = u0_j_5d - u0_i_5d

            if cls == "adj_x":
                chi = (self.a * dx_t > 0).float()
                feat = torch.cat([
                    h, h_j,
                    du, a_bc,
                    torch.sign(x_j - x_i),
                    chi,
                ], dim=-1)                                                  # 2d + 4
                gate = _soft_upwind_axis(self.a, dx_t, tau_x)
                adj_x_feats.append(feat)
                adj_x_gates.append(gate)

            elif cls == "adj_y":
                chi = (self.b * dy_t > 0).float()
                feat = torch.cat([
                    h, h_j,
                    du, b_bc,
                    torch.sign(y_j - y_i),
                    chi,
                ], dim=-1)                                                  # 2d + 4
                gate = _soft_upwind_axis(self.b, dy_t, tau_y)
                adj_y_feats.append(feat)
                adj_y_gates.append(gate)

            else:  # nonadj
                chi = (self.a * dx_t + self.b * dy_t > 0).float()
                feat = torch.cat([
                    h, h_j,
                    du, dx_t, dy_t, dt_t,
                    a_bc, b_bc,
                    chi,
                ], dim=-1)                                                  # 2d + 7
                g_up = _soft_upwind_2d(self.a, self.b, dx_t, dy_t, tau_non)
                g_tmp = _char_alignment_gate(
                    self.a, self.b, dx_t, dy_t, dt_t, kappa, self.h_s,
                )
                gate = g_up * g_tmp
                nonadj_feats.append(feat)
                nonadj_gates.append(gate)

        def _agg(
            feats: list[torch.Tensor], gates: list[torch.Tensor],
            mlp: nn.Module,
        ) -> torch.Tensor:
            if not feats:
                return h.new_zeros((B, nt, ny, nx, d))
            stacked_feat = torch.stack(feats, dim=4)
            n = stacked_feat.shape[4]
            stacked_gate = torch.stack(gates, dim=4)
            msgs = mlp(
                stacked_feat.reshape(-1, stacked_feat.shape[-1])
            ).reshape(B, nt, ny, nx, n, d)
            gate_sum = stacked_gate.sum(dim=4) + 1e-3
            return ((stacked_gate / gate_sum.unsqueeze(4)) * msgs).sum(dim=4)

        M_x = _agg(adj_x_feats, adj_x_gates, self.adj_x_msg)
        M_y = _agg(adj_y_feats, adj_y_gates, self.adj_y_msg)
        M_non = _agg(nonadj_feats, nonadj_gates, self.nonadj_msg)

        # No α_x, α_y, η scaling — see module docstring. update_net learns it.
        agg = torch.cat([M_x, M_y, M_non], dim=-1)                          # [...,3d]
        h_nonlocal = self.update_net(torch.cat([h, agg], dim=-1))           # 4d -> d
        h_local = self.W(h)
        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #
class HypNO_Linear_2D_v0(nn.Module):
    """HypNO-ST 2D v0 — constant-coefficient linear advection.

    Lifting layer -> n_layers message-passing layers -> shared decoder.
    Deep supervision: every MP layer produces an intermediate `u_hat` via
    the shared decoder, returned in `u_hats`.
    """

    def __init__(
        self,
        stencil_k_x: int = 3,
        stencil_k_y: int = 3,
        stencil_k_t: int = 3,
        d_latent: int = 64,
        d_hidden: int = 64,
        d_hidden_nonadj: int | None = None,
        n_layers: int = 6,
        activation: str = "gelu",
        causal_temporal: bool = True,
        readout: str = "gelu",
        encoder_type: str = "gnn",
        skip: bool = True,
        use_checkpoint: bool = True,
        a: float = 1.0,
        b: float = 0.7,
        h_s: float | None = None,
        **_ignored,
    ) -> None:
        super().__init__()
        # Default h_s = sqrt(dx * dy) computed at forward time if not set.
        # Stored as a Python float here for simplicity (model is grid-aware
        # via h_s only at construction; if the grid spacing changes, set h_s
        # via the config).
        if h_s is None:
            h_s = 1.0  # caller should set explicitly when grid is known
        print("=" * 60)
        print("[HypNO_Linear_2D_v0] constructing:")
        print(f"  stencil_k_x/y/t  = {stencil_k_x}/{stencil_k_y}/{stencil_k_t}")
        print(f"  d_latent         = {d_latent}")
        print(f"  d_hidden         = {d_hidden}")
        print(f"  d_hidden_nonadj  = {d_hidden_nonadj}")
        print(f"  n_layers         = {n_layers}")
        print(f"  encoder_type     = {encoder_type}")
        print(f"  readout          = {readout}")
        print(f"  skip             = {skip}")
        print(f"  use_checkpoint   = {use_checkpoint}")
        print(f"  (a, b)           = ({a}, {b})")
        print(f"  h_s              = {h_s}")
        if _ignored:
            print(f"  IGNORED kwargs   = {sorted(_ignored.keys())}")
        print("=" * 60)

        self.skip = skip
        self.use_checkpoint = use_checkpoint
        self.a = float(a)
        self.b = float(b)
        self.h_s = float(h_s)

        self.lifting = _LinearAdvectionLiftingLayer2D(
            d_latent, d_hidden,
            stencil_k_x=stencil_k_x, stencil_k_y=stencil_k_y, stencil_k_t=stencil_k_t,
            a=self.a, b=self.b, h_s=self.h_s,
            activation=activation, encoder_type=encoder_type,
            causal_temporal=causal_temporal,
        )

        self.decoder = _make_mlp(d_latent, d_hidden, 1, 3, readout)

        self.mp_layers = nn.ModuleList([
            _LinearAdvectionMPLayer2D(
                d_latent, d_hidden, stencil_k_x, stencil_k_y, stencil_k_t,
                a=self.a, b=self.b, h_s=self.h_s,
                activation=activation, causal_temporal=causal_temporal,
                d_hidden_nonadj=d_hidden_nonadj,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        u0: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        u0 : [B, ny, nx]
        x  : [nx]
        y  : [ny]
        t  : [nt]
        Returns (u_pred, u_hats) where u_pred = u_hats[-1] is [B, nt, ny, nx].
        """
        B, ny, nx = u0.shape
        nt = t.shape[0]

        h = self.lifting(u0, x, y, t)

        def _decode(h_in: torch.Tensor) -> torch.Tensor:
            d_out = self.decoder(h_in).squeeze(-1)                          # [B, nt, ny, nx]
            if self.skip:
                d_out = d_out + u0.view(B, 1, ny, nx).expand(B, nt, ny, nx)
            return d_out

        u_hats: list[torch.Tensor] = []
        for layer in self.mp_layers:
            if self.use_checkpoint and self.training:
                h = torch_checkpoint(layer, h, x, y, t, u0, use_reentrant=False)
            else:
                h = layer(h, x, y, t, u0)
            u_hats.append(_decode(h))

        u_pred = u_hats[-1]
        return u_pred, u_hats
