"""HypNO-ST v3 — structurally separate adjacent / non-adjacent edge MLPs.

Changes from hypno_st2.py
-------------------------
* No adjacency masking.  For non-adjacent edges (|offset| > 1), interface
  quantities (du, |du|, u_avg, rel_x, s=a_ij, sign(s), upwind) are not
  present in the feature vector at all — not zeroed, absent.
* Adjacent feature vector is unchanged (13-dim static features + time).
* Non-adjacent feature vector is 8-dim:
      (u_i, u_k, f_i, f_k, a_i, a_k, x_i, x_k)
  All entries are pointwise-per-node (f = u(1-u), a = 1-2u) or absolute
  coordinates, so none depend on interface geometry.
* Physics MP layer owns TWO edge MLPs — adj_msg and nonadj_msg — routed
  by whether the edge is adjacent-spatial (dm==0, |di|==1).
* The non-adjacent MLP can use a smaller hidden width via
  ``d_hidden_nonadj`` (defaults to ``d_hidden``).
* Physics gate: upwind and Oleinik entropy require a_ij and are adj-only.
  On non-adjacent edges the gate is identically 1 (char-cone component
  still applies when ``use_char_cone`` is True — it is node-local).
* Gate-normalised aggregation: weights = gate_k / sum(gate_j), summing to 1.
* Only physics shock_mode is supported. encoder_type (gnn/mlp),
  encoder_scaling (classic/physics), use_char_cone, causal_temporal,
  loss_type, detector_path all preserved.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import sys
from datetime import datetime
from pathlib import Path
from hyperbolic_pde.utils.runtime import apply_runtime_overrides, resolve_config_path
import yaml


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> dict:
    base_path = ROOT / "configs" / "hyperbolic_pde.yaml"
    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path.resolve() == base_path.resolve():
        return cfg
    with path.open("r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    return _deep_update(cfg, override or {})


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent))
cfg = load_config(resolve_config_path(ROOT / "configs"))
cfg = apply_runtime_overrides(cfg)


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


# ---- index constants ------------------------------------------------------ #
# Adjacent static feature vector (12 slots):
# 0: u_i,  1: u_k,  2: du,  3: u_avg,  4: sign(rel_x),
# 5: f_i,  6: f_k,  7: a_i,  8: a_k,  9: a_ij (RH speed),  10: sign(a_ij),  11: upwind
# NOTE: |du| and the signed rel_x magnitude were removed (uninformative /
# replaced with sign(rel_x)). a_ij now falls back to a_i (= lambda_i) when
# |du| is below the safety threshold, instead of the u_avg-based estimate.
N_EDGE_FEATS_ADJ = 12

# Non-adjacent static feature vector (8 slots):
# 0: u_i,  1: u_k,  2: f_i,  3: f_k,  4: a_i,  5: a_k,  6: x_i,  7: x_k
N_EDGE_FEATS_NONADJ = 8


def precompute_lwr_edge_features_v3(
    u0: torch.Tensor,
    x: torch.Tensor,
    stencil_k: int,
    radius_x: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Precompute static LWR edge features for v3.

    Returns two separate tensors — one per adjacency class — so the
    non-adjacent tensor really does carry fewer columns (no masked zeros).

    Parameters
    ----------
    u0 : Tensor [N, nx]
    x  : Tensor [nx]
    stencil_k : int
    radius_x  : float | None

    Returns
    -------
    feats_adj    : Tensor [N, nx, 3, 12]        — offsets j in {-1, 0, 1}
    feats_nonadj : Tensor [N, nx, 2(k-1), 8]    — offsets j in {-k..-2} ∪ {2..k}
                   ``None`` when k <= 1 (no non-adjacent edges exist).
    """
    if radius_x is not None:
        dx = (x[1] - x[0]).abs().item()
        k = max(1, int(radius_x / dx + 0.5))
    else:
        k = stencil_k

    N, nx = u0.shape

    u_pad = F.pad(u0.unsqueeze(1), (k, k), mode="replicate").squeeze(1)
    x_exp = x.unsqueeze(0).expand(N, -1)
    x_pad = F.pad(x_exp.unsqueeze(1), (k, k), mode="replicate").squeeze(1)

    adj_list: list[torch.Tensor] = []
    nonadj_list: list[torch.Tensor] = []
    for j in range(-k, k + 1):
        u_k = u_pad[:, k + j : k + j + nx]
        x_k = x_pad[:, k + j : k + j + nx]

        f_i = u0 * (1.0 - u0)
        f_k = u_k * (1.0 - u_k)
        a_i = 1.0 - 2.0 * u0
        a_k = 1.0 - 2.0 * u_k

        if abs(j) <= 1:
            du     = u_k - u0
            u_avg  = 0.5 * (u0 + u_k)
            rel_x  = x_k - x_exp
            sign_dx = torch.sign(rel_x)

            du_safe = torch.where(du.abs() < 1e-6, torch.ones_like(du), du)
            a_ik = torch.where(
                du.abs() < 1e-6,
                a_i,
                (f_k - f_i) / du_safe,
            )
            sign_a = torch.sign(a_ik)
            upwind = (a_ik * rel_x < 0).float()

            feat = torch.stack([
                u0, u_k,
                du, u_avg,
                sign_dx,
                f_i, f_k,
                a_i, a_k, a_ik,
                sign_a, upwind,
            ], dim=-1)                                          # [N, nx, 12]
            adj_list.append(feat)
        else:
            feat = torch.stack([
                u0, u_k,
                f_i, f_k,
                a_i, a_k,
                x_exp, x_k,
            ], dim=-1)                                          # [N, nx, 8]
            nonadj_list.append(feat)

    feats_adj = torch.stack(adj_list, dim=2)                    # [N, nx, 3, 13]
    feats_nonadj = (
        torch.stack(nonadj_list, dim=2) if nonadj_list else None
    )                                                           # [N, nx, 2(k-1), 8] or None
    return feats_adj, feats_nonadj


# --------------------------------------------------------------------------- #
# Helper: spatial edge features from decoded state (for MP layers)
# --------------------------------------------------------------------------- #
def _compute_adj_spatial_edge_feats(
    u_hat_i: torch.Tensor, u_hat_j: torch.Tensor,
    rel_x: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Return the 9-tuple of adjacent-edge features from decoded state.

    Outputs: (du, u_avg, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind).
    """
    f_i = u_hat_i * (1.0 - u_hat_i)
    f_j = u_hat_j * (1.0 - u_hat_j)
    a_i = 1.0 - 2.0 * u_hat_i
    a_j = 1.0 - 2.0 * u_hat_j

    du    = u_hat_j - u_hat_i
    u_avg = 0.5 * (u_hat_i + u_hat_j)

    du_safe = torch.where(du.abs() < 1e-6, torch.ones_like(du), du)
    a_ij = torch.where(
        du.abs() < 1e-6,
        a_i,
        (f_j - f_i) / du_safe,
    )
    sign_a = torch.sign(a_ij)
    upwind = (a_ij * rel_x < 0).float()

    return du, u_avg, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind


def _compute_nonadj_pair_feats(
    u_hat_i: torch.Tensor, u_hat_j: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Return per-node pointwise features used on non-adjacent edges.

    Outputs: (f_i, f_j, a_i, a_j).  Absolute positions and u values are
    supplied separately by the caller.
    """
    f_i = u_hat_i * (1.0 - u_hat_i)
    f_j = u_hat_j * (1.0 - u_hat_j)
    a_i = 1.0 - 2.0 * u_hat_i
    a_j = 1.0 - 2.0 * u_hat_j
    return f_i, f_j, a_i, a_j


def _enumerate_ball_offsets(k_x: int, k_t: int, causal: bool) -> list[tuple[int, int]]:
    """Product-box space-time stencil (Chebyshev ball).

    Returns all `(di, dm)` with `|di| <= k_x` and `|dm| <= k_t`, excluding
    the centre `(0, 0)`. When `causal=True`, only edges with `dm <= 0`
    (past or present in time) are returned -- diagonals respect the same
    causality as pure-temporal edges.
    """
    m_range = range(-k_t, 1) if causal else range(-k_t, k_t + 1)
    out: list[tuple[int, int]] = []
    for dm in m_range:
        for di in range(-k_x, k_x + 1):
            if di == 0 and dm == 0:
                continue
            out.append((di, dm))
    return out


def _pad_space_time(h: torch.Tensor, k_x: int, k_t: int) -> torch.Tensor:
    """Replicate-pad `h [B, nt, nx, d]` to `[B, nt+2k_t, nx+2k_x, d]`.

    Ghost-cell (zero-order extrapolation) padding on both axes, done in
    one call.
    """
    # conv2d-style padding expects [B, C, H, W] with pad = (left, right, top, bottom)
    h_pad = F.pad(
        h.permute(0, 3, 1, 2),                 # [B, d, nt, nx]
        (k_x, k_x, k_t, k_t),
        mode="replicate",
    ).permute(0, 2, 3, 1)                      # [B, nt+2k_t, nx+2k_x, d]
    return h_pad


# --------------------------------------------------------------------------- #
# Lifting layer
# --------------------------------------------------------------------------- #
class _SpaceTimeLiftingLayer(nn.Module):
    """Space-time lifting over a product-box ball neighbourhood.

    A single unified edge MLP consumes edges `(di, dm)` inside the
    Chebyshev ball `|di| <= k_x, |dm| <= k_t` (excluding the centre),
    including diagonals. Two gating modes are supported:

    * ``encoder_scaling = "classic"`` -- no gate (gate == 1).
    * ``encoder_scaling = "physics"`` -- analytical physics gate.
      Upwind and Oleinik components fire only on adjacent-spatial edges
      (``dm == 0, |di| == 1``); all other edges pass through untouched.
      If ``use_char_cone`` is true, the char-cone component applies on
      every edge, using ``rel_t`` when it's non-zero and the absolute
      ``t_i`` on pure-spatial edges (preserving the v3 semantic).

    Edge feature vector (unified, 14 static dims; t absorbed via
    ``t_i, t_j``):
        u0_i, u0_j, f0_i, f0_j, a0_i, a0_j, du0,
        sign(rel_x), rel_t, t_i, t_j, a0_ij, sign(a0_ij), is_adj_sp
    """

    def __init__(
        self, d_latent: int, d_hidden: int, stencil_k_x: int, stencil_k_t: int,
        activation: str,
        radius_x: float | None = None,
        radius_t: float | None = None,
        encoder_scaling: str = "physics",
        encoder_type: str = "gnn",
        use_char_cone: bool = False,
        causal_temporal: bool = True,
        include_flux: bool = True,
        pure_pairwise_edges: bool = False,
    ) -> None:
        super().__init__()
        self.k_x = stencil_k_x
        self.k_t = stencil_k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.encoder_scaling = encoder_scaling
        self.encoder_type = encoder_type
        self.use_char_cone = use_char_cone
        self.causal = causal_temporal
        self.include_flux = include_flux
        self.pure_pairwise_edges = pure_pairwise_edges
        node_in = 5 if include_flux else 4
        self.node_mlp = _make_mlp(node_in, d_hidden, d_latent, 2, activation)
        print(
            f"[Lifting] include_flux={include_flux} pure_pairwise_edges={pure_pairwise_edges} | "
            f"node_mlp in={node_in} (5 with flux, 4 without)"
        )
        if encoder_type == "mlp":
            return

        # Edge MLP input dim:
        #   pure_pairwise_edges=True:  8  -> [du0, sign(rel_x), rel_t, t_bc, t_j, a0_ij, sign(a0_ij), is_adj_flag]
        #   include_flux=True:        14  -> + u0_i, u0_j, f0_i, f0_j, a0_i, a0_j
        #   include_flux=False:       12  -> + u0_i, u0_j, a0_i, a0_j  (drops f0_i, f0_j)
        if pure_pairwise_edges:
            edge_in = 8
        elif include_flux:
            edge_in = 14
        else:
            edge_in = 12
        self.edge_mlp = _make_mlp(edge_in, d_hidden, d_latent, 2, activation)
        print(
            f"[Lifting] edge_mlp in={edge_in} "
            f"(8 pure-pairwise, 14 with flux, 12 without flux)"
        )

        if encoder_scaling == "classic":
            pass
        elif encoder_scaling == "physics":
            self.phys_temperature = nn.Parameter(torch.tensor(0.0))
            self.phys_gamma_entropy = nn.Parameter(torch.tensor(-2.0))
            if use_char_cone:
                self.phys_char_width = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(
                f"Unknown encoder_scaling: {encoder_scaling!r}. "
                "Expected 'classic' or 'physics'."
            )
        self.combine = _make_mlp(2 * d_latent, d_hidden, d_latent, 2, activation)

    def _physics_gate_ball(
        self,
        di: int, dm: int,
        u_i: torch.Tensor, u_j: torch.Tensor,
        rel_x: torch.Tensor, rel_t: torch.Tensor,
        a_ij: torch.Tensor, a_j: torch.Tensor,
        t_i: torch.Tensor, dx_grid: float,
    ) -> torch.Tensor:
        """Physics gate on a single ball edge `(di, dm)`.

        Upwind x Oleinik fire only on adjacent-spatial edges
        (``dm == 0, |di| == 1``). On all other edges those components
        reduce to 1. The optional char-cone factor is node-local and is
        applied whenever it's enabled; it uses ``rel_t`` for diagonal /
        pure-temporal edges and falls back to ``t_i`` on pure-spatial
        edges (v3-compatible).
        """
        is_adj_sp = (dm == 0) and (abs(di) == 1)
        is_pure_sp = (dm == 0)

        if is_adj_sp:
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
        else:
            gate = torch.ones_like(rel_x)

        if self.use_char_cone:
            char_w = F.softplus(self.phys_char_width).clamp(min=1e-6)
            if is_pure_sp:
                t_ref = t_i
                char_miss = (rel_x + a_j * t_ref).abs()
            else:
                char_miss = (rel_x - a_j * rel_t).abs()
                t_ref = rel_t
            sigma = char_w * (dx_grid + a_j.abs() * t_ref.abs().clamp(min=1e-6))
            g_char = torch.exp(-0.5 * (char_miss / sigma.clamp(min=1e-6)) ** 2)
            gate = gate * g_char

        return gate

    def forward(
        self,
        u0: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        edge_feats_adj_pre: torch.Tensor | None = None,
        edge_feats_nonadj_pre: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        # Precomputed edge tensors are ignored -- they only cover pure-
        # spatial offsets at t=0 and don't apply to the space-time ball.
        del edge_feats_adj_pre, edge_feats_nonadj_pre

        B, nx = u0.shape
        nt = t.shape[0]
        dx_grid = (x[0, 1] - x[0, 0]).abs().item()
        dt_grid = (t[1] - t[0]).abs().item()

        u0_bc = u0.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        x_bc  = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        t_bc  = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)

        f0_i_node = u0_bc * (1.0 - u0_bc)
        a0_i_node = 1.0 - 2.0 * u0_bc
        if self.include_flux:
            node_in = torch.cat([u0_bc, x_bc, t_bc, f0_i_node, a0_i_node], dim=-1)
        else:
            node_in = torch.cat([u0_bc, x_bc, t_bc, a0_i_node], dim=-1)
        h_node  = self.node_mlp(node_in)

        if self.encoder_type == "mlp":
            if return_intermediates:
                # In MLP mode "post-node" and "post-full" coincide.
                return h_node, h_node, h_node
            return h_node

        # Resolve neighbourhood extents.
        k_x = (
            max(1, int(self.radius_x / dx_grid + 0.5))
            if self.radius_x is not None else self.k_x
        )
        k_t = (
            max(1, int(self.radius_t / dt_grid + 0.5))
            if self.radius_t is not None else self.k_t
        )

        # Pad u0 in x, and (u0 broadcast along t) needs no temporal pad
        # since u0 itself is t-invariant -- only h_node at neighbour
        # times matters for the node stream, which is already computed
        # above.  The edge features depend on u0_j (function of x_j
        # only) and on absolute x_j, t_j, so we only need to shift t_j.
        u0_pad = F.pad(u0.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        t_pad = F.pad(
            t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate",
        ).squeeze(0).squeeze(0)

        f0_i = f0_i_node
        a0_i = a0_i_node

        offsets = _enumerate_ball_offsets(k_x, k_t, self.causal)

        def _build_edge(di: int, dm: int):
            """Return ``(edge_in, gate, rel_x, rel_t)`` for one ball edge.

            Cheap: no ``edge_mlp`` call here.
            """
            u0_j = u0_pad[:, k_x + di : k_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            x_j  = x_pad[:, k_x + di : k_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            t_j  = t_pad[k_t + dm : k_t + dm + nt].view(1, nt, 1, 1).expand(B, nt, nx, 1)
            rel_x = x_j - x_bc
            rel_t = t_j - t_bc
            f0_j = u0_j * (1.0 - u0_j)
            a0_j = 1.0 - 2.0 * u0_j
            du0 = u0_j - u0_bc
            is_adj_sp = (dm == 0) and (abs(di) == 1)
            if is_adj_sp:
                du_safe = torch.where(du0.abs() < 1e-6, torch.ones_like(du0), du0)
                a0_ij = torch.where(
                    du0.abs() < 1e-6,
                    a0_i.expand_as(du0),
                    (f0_j - f0_i) / du_safe,
                )
                sign_a0 = torch.sign(a0_ij)
            else:
                a0_ij = torch.zeros_like(du0)
                sign_a0 = torch.zeros_like(du0)
            is_adj_flag = u0_bc.new_full(u0_bc.shape, 1.0 if is_adj_sp else 0.0)
            sign_rel_x = torch.sign(rel_x)
            if self.pure_pairwise_edges:
                edge_in = torch.cat([
                    du0,
                    sign_rel_x, rel_t,
                    t_bc, t_j,
                    a0_ij, sign_a0,
                    is_adj_flag,
                ], dim=-1)                                                      # [..., 8]
            elif self.include_flux:
                edge_in = torch.cat([
                    u0_bc, u0_j,
                    f0_i, f0_j,
                    a0_i, a0_j,
                    du0,
                    sign_rel_x, rel_t,
                    t_bc, t_j,
                    a0_ij, sign_a0,
                    is_adj_flag,
                ], dim=-1)                                                      # [..., 14]
            else:
                edge_in = torch.cat([
                    u0_bc, u0_j,
                    a0_i, a0_j,
                    du0,
                    sign_rel_x, rel_t,
                    t_bc, t_j,
                    a0_ij, sign_a0,
                    is_adj_flag,
                ], dim=-1)                                                      # [..., 12]
            if self.encoder_scaling == "classic":
                gate = torch.ones_like(rel_x)
            else:  # physics
                gate = self._physics_gate_ball(
                    di=di, dm=dm,
                    u_i=u0_bc, u_j=u0_j,
                    rel_x=rel_x, rel_t=rel_t,
                    a_ij=a0_ij, a_j=a0_j,
                    t_i=t_bc, dx_grid=dx_grid,
                )
            return edge_in, gate, rel_x, rel_t

        # ---- collect features and gates ----
        gates: list[torch.Tensor] = []
        edge_inputs: list[torch.Tensor] = []
        for di, dm in offsets:
            edge_in, gate, rel_x, rel_t = _build_edge(di, dm)
            if self.radius_x is not None:
                gate = gate.masked_fill(rel_x.abs() > self.radius_x, 0.0)
            if self.radius_t is not None:
                gate = gate.masked_fill(rel_t.abs() > self.radius_t, 0.0)
            gates.append(gate)
            edge_inputs.append(edge_in)
        # Additive floor (NOT clamp): when all edges legitimately suppress, the
        # aggregate attenuates to zero rather than blowing up via 1/eps. This
        # was the source of isolated salt-and-pepper spikes in smooth regions.
        gate_sum = torch.stack(gates, dim=-2).sum(dim=-2) + 1e-3                # [B, nt, nx, 1]

        # ---- single batched MLP call over all offsets ----
        n_offsets = len(edge_inputs)
        edge_in_stacked = torch.stack(edge_inputs, dim=3)                        # [B, nt, nx, n_offsets, 14]
        msgs = self.edge_mlp(
            edge_in_stacked.reshape(-1, edge_in_stacked.shape[-1])
        ).reshape(B, nt, nx, n_offsets, -1)                                      # [B, nt, nx, n_offsets, d]
        gates_t = torch.stack(gates, dim=3)                                      # [B, nt, nx, n_offsets, 1]
        agg = (gates_t / gate_sum.unsqueeze(3) * msgs).sum(dim=3)               # [B, nt, nx, d]

        h_full = self.combine(torch.cat([h_node, agg], dim=-1))
        if return_intermediates:
            return h_full, h_node, h_full
        return h_full


# --------------------------------------------------------------------------- #
# Physics-gated space-time MP layer (v3)
# --------------------------------------------------------------------------- #
# Physics-gated space-time MP layer (v3)
# --------------------------------------------------------------------------- #
class _PhysicsSpaceTimeMPLayer(nn.Module):
    """Space-time MP over a product-box ball with analytical physics gate.

    Two edge MLPs — adj_msg and nonadj_msg — are routed by adjacency class:
    * Adjacent-spatial edges (dm==0, |di|==1): 2d+10 features, physics gate
      with upwind x entropy components.
    * All other edges (non-adj spatial, temporal, diagonal): 2d+6 features,
      CFL gate only (plus optional char-cone).

    Aggregation is gate-normalised: w_k = gate_k / sum(gate_j), summing to 1.

    Adjacent edge features (2d + 10):
        h_i, h_j, u_i, u_j, f_i, f_j, a_i, a_j, a_ij, sign(a_ij), upwind, sign(rel_x)
    Non-adjacent edge features (2d + 11):
        h_i, h_j, u_i, u_j, f_i, f_j, a_i, a_j, rel_x, rel_t, cfl, sign(rel_x), xi (= x_i / max(t_i, eps))
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
        causal_temporal: bool = True,
        use_char_cone: bool = False,
        d_hidden_nonadj: int | None = None,
        mask_same_t_nonadj: bool = True,
        temporal_gate_type: str = "cfl",
        include_flux: bool = True,
        pure_pairwise_edges: bool = False,
        **_ignored,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_t = k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.causal = causal_temporal
        self.use_char_cone = use_char_cone
        self.mask_same_t_nonadj = mask_same_t_nonadj
        self.include_flux = include_flux
        self.pure_pairwise_edges = pure_pairwise_edges
        if temporal_gate_type not in {"cfl", "none", "time"}:
            raise ValueError(
                f"temporal_gate_type must be one of 'cfl', 'none', 'time'; "
                f"got {temporal_gate_type!r}."
            )
        self.temporal_gate_type = temporal_gate_type
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        self.state_probe = nn.Linear(d_latent, 1)

        self.phys_temperature = nn.Parameter(torch.tensor(0.0))
        self.phys_gamma_entropy = nn.Parameter(torch.tensor(-2.0))
        if use_char_cone:
            self.phys_char_width = nn.Parameter(torch.tensor(0.0))
        self.phys_cfl_scale = nn.Parameter(torch.tensor(0.0))
        self.phys_time_decay = nn.Parameter(torch.tensor(0.0))

        # adj edge extras:
        #   pure_pairwise_edges=True: 4 -> [a_ij, sign(a_ij), upwind, sign(rel_x)]
        #   include_flux=True:       10 -> + u_i, u_j, f_i, f_j, a_i, a_j
        #   include_flux=False:       8 -> + u_i, u_j, a_i, a_j
        # nonadj edge extras:
        #   pure_pairwise_edges=True: 3 -> [rel_x, rel_t, sign(rel_x)]   (cfl,xi dropped: depend on a_i)
        #   include_flux=True:       11 -> + u_i, u_j, f_i, f_j, a_i, a_j, cfl, xi
        #   include_flux=False:       9 -> + u_i, u_j, a_i, a_j, cfl, xi
        if pure_pairwise_edges:
            adj_extra = 4
            nonadj_extra = 3
        elif include_flux:
            adj_extra = 10
            nonadj_extra = 11
        else:
            adj_extra = 8
            nonadj_extra = 9
        self.adj_msg = _make_mlp(2 * d_latent + adj_extra, d_hidden, d_latent, 3, activation)
        self.nonadj_msg = _make_mlp(2 * d_latent + nonadj_extra, dh_na, d_latent, 3, activation)
        print(
            f"[MP] include_flux={include_flux} pure_pairwise_edges={pure_pairwise_edges} | "
            f"adj_msg in=2*{d_latent}+{adj_extra}={2 * d_latent + adj_extra} "
            f"(extra: 4 pure-pairwise, 10 with flux, 8 without) | "
            f"nonadj_msg in=2*{d_latent}+{nonadj_extra}={2 * d_latent + nonadj_extra} "
            f"(extra: 3 pure-pairwise, 11 with flux, 9 without) | "
            f"temporal_gate_type={temporal_gate_type}"
        )

        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    def _ball_physics_gate(
        self,
        di: int, dm: int,
        u_i: torch.Tensor, u_j: torch.Tensor,
        rel_x: torch.Tensor, rel_t: torch.Tensor,
        a_i: torch.Tensor, a_j: torch.Tensor, a_ij: torch.Tensor,
        t_i: torch.Tensor, dx_grid: float,
    ) -> torch.Tensor:
        is_adj_sp = (dm == 0) and (abs(di) == 1)
        is_pure_sp = (dm == 0)

        if is_pure_sp and not is_adj_sp:
            return torch.zeros_like(rel_x)

        if is_adj_sp:
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
        else:
            gate = torch.ones_like(rel_x)

        if not is_pure_sp:                                                     # dm != 0
            if self.temporal_gate_type == "cfl":
                cfl_scale = F.softplus(self.phys_cfl_scale).clamp(min=1e-6)
                cfl = a_i.abs() * rel_t.abs() / dx_grid
                g_cfl = torch.exp(-cfl_scale * F.relu(cfl - 1.0) ** 2)
                gate = gate * g_cfl
            elif self.temporal_gate_type == "time":
                alpha_t = F.softplus(self.phys_time_decay).clamp(min=1e-6)
                r_t = float(abs(dm))
                g_time = 1.0 / (1.0 + alpha_t * r_t)
                gate = gate * g_time
            # "none" -> no temporal attenuation

        if self.use_char_cone:
            char_w = F.softplus(self.phys_char_width).clamp(min=1e-6)
            if is_pure_sp:
                t_ref = t_i
                char_miss = (rel_x + a_j * t_ref).abs()
            else:
                t_ref = rel_t
                char_miss = (rel_x - a_j * rel_t).abs()
            sigma = char_w * (dx_grid + a_j.abs() * t_ref.abs().clamp(min=1e-6))
            g_char = torch.exp(-0.5 * (char_miss / sigma.clamp(min=1e-6)) ** 2)
            gate = gate * g_char

        return gate

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        u0: torch.Tensor,
        shock_indicator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, nt, nx, d = h.shape
        dx_val = (x[0, 1] - x[0, 0]).abs().item()
        dt_val = (t[1] - t[0]).abs().item()

        k_x = (
            max(1, int(self.radius_x / dx_val + 0.5))
            if self.radius_x is not None else self.k_x
        )
        k_t = (
            max(1, int(self.radius_t / dt_val + 0.5))
            if self.radius_t is not None else self.k_t
        )

        u_hat = torch.sigmoid(self.state_probe(h)).squeeze(-1)                  # [B, nt, nx]
        u_hat_i = u_hat.unsqueeze(-1)                                           # [B, nt, nx, 1]

        # Joint space-time padding.
        h_pad = _pad_space_time(h, k_x, k_t)                                    # [B, nt+2k_t, nx+2k_x, d]
        u_hat_pad = _pad_space_time(u_hat.unsqueeze(-1), k_x, k_t).squeeze(-1)  # [B, nt+2k_t, nx+2k_x]
        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)  # [B, nx+2k_x]
        t_pad = F.pad(
            t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate",
        ).squeeze(0).squeeze(0)                                                  # [nt+2k_t]

        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        t_i = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)
        a_hat_i = 1.0 - 2.0 * u_hat_i
        f_hat_i = u_hat_i * (1.0 - u_hat_i)

        offsets = _enumerate_ball_offsets(k_x, k_t, self.causal)

        def _build_edge(di: int, dm: int):
            """Return ``(msg_in, is_adj_sp, gate, rel_x, rel_t)`` for one ball edge."""
            h_j = h_pad[:, k_t + dm : k_t + dm + nt, k_x + di : k_x + di + nx, :]
            u_hat_j = u_hat_pad[:, k_t + dm : k_t + dm + nt, k_x + di : k_x + di + nx].unsqueeze(-1)
            x_j = x_pad[:, k_x + di : k_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            t_j = t_pad[k_t + dm : k_t + dm + nt].view(1, nt, 1, 1).expand(B, nt, nx, 1)
            rel_x = x_j - x_i
            rel_t = t_j - t_i
            f_j = u_hat_j * (1.0 - u_hat_j)
            a_j = 1.0 - 2.0 * u_hat_j
            cfl = a_hat_i.abs() * rel_t.abs() / dx_val
            is_adj_sp = (dm == 0) and (abs(di) == 1)

            if is_adj_sp:
                _, _, _, _, _, _, a_ij, sign_a, upwind = \
                    _compute_adj_spatial_edge_feats(u_hat_i, u_hat_j, rel_x)
                if self.pure_pairwise_edges:
                    msg_in = torch.cat([
                        h, h_j,
                        a_ij, sign_a, upwind,
                        torch.sign(rel_x),
                    ], dim=-1)                                                  # 2d + 4
                elif self.include_flux:
                    msg_in = torch.cat([
                        h, h_j,
                        u_hat_i.expand_as(rel_x), u_hat_j,
                        f_hat_i, f_j, a_hat_i.expand_as(rel_x), a_j,
                        a_ij, sign_a, upwind,
                        torch.sign(rel_x),
                    ], dim=-1)                                                  # 2d + 10
                else:
                    msg_in = torch.cat([
                        h, h_j,
                        u_hat_i.expand_as(rel_x), u_hat_j,
                        a_hat_i.expand_as(rel_x), a_j,
                        a_ij, sign_a, upwind,
                        torch.sign(rel_x),
                    ], dim=-1)                                                  # 2d + 8
            else:
                a_ij = torch.zeros_like(rel_x)
                xi = x_i / t_i.clamp(min=1e-6)
                if self.pure_pairwise_edges:
                    msg_in = torch.cat([
                        h, h_j,
                        rel_x, rel_t,
                        torch.sign(rel_x),
                    ], dim=-1)                                                  # 2d + 3
                elif self.include_flux:
                    msg_in = torch.cat([
                        h, h_j,
                        u_hat_i.expand_as(rel_x), u_hat_j,
                        f_hat_i, f_j, a_hat_i.expand_as(rel_x), a_j,
                        rel_x, rel_t, cfl,
                        torch.sign(rel_x),
                        xi,
                    ], dim=-1)                                                  # 2d + 11
                else:
                    msg_in = torch.cat([
                        h, h_j,
                        u_hat_i.expand_as(rel_x), u_hat_j,
                        a_hat_i.expand_as(rel_x), a_j,
                        rel_x, rel_t, cfl,
                        torch.sign(rel_x),
                        xi,
                    ], dim=-1)                                                  # 2d + 9

            gate = self._ball_physics_gate(
                di=di, dm=dm,
                u_i=u_hat_i, u_j=u_hat_j,
                rel_x=rel_x, rel_t=rel_t,
                a_i=a_hat_i, a_j=a_j, a_ij=a_ij,
                t_i=t_i, dx_grid=dx_val,
            )
            return msg_in, is_adj_sp, gate, rel_x, rel_t

        # ---- collect features and gates, split by adjacency class ----
        adj_feats:    list[torch.Tensor] = []
        nonadj_feats: list[torch.Tensor] = []
        adj_gates:    list[torch.Tensor] = []
        nonadj_gates: list[torch.Tensor] = []
        for di, dm in offsets:
            if self.mask_same_t_nonadj and dm == 0 and abs(di) > 1:
                continue
            msg_in, is_adj_sp, gate, rel_x, rel_t = _build_edge(di, dm)
            if self.radius_x is not None:
                gate = gate.masked_fill(rel_x.abs() > self.radius_x, 0.0)
            if self.radius_t is not None:
                gate = gate.masked_fill(rel_t.abs() > self.radius_t, 0.0)
            if is_adj_sp:
                adj_feats.append(msg_in)
                adj_gates.append(gate)
            else:
                nonadj_feats.append(msg_in)
                nonadj_gates.append(gate)

        # gate normalisation over all edges
        # Additive floor (NOT clamp): when all edges legitimately suppress, the
        # aggregate attenuates to zero rather than blowing up via 1/eps. This
        # was the source of isolated salt-and-pepper spikes in smooth regions.
        all_gates = adj_gates + nonadj_gates                                      # [B, nt, nx, 1] each
        gate_sum = torch.stack(all_gates, dim=-2).sum(dim=-2) + 1e-3              # [B, nt, nx, 1]

        # ---- batched MLP pass: 2 adj calls fused, 24 nonadj calls fused ----
        n_adj    = len(adj_feats)
        n_nonadj = len(nonadj_feats)

        adj_in   = torch.stack(adj_feats,    dim=3)                               # [B, nt, nx, n_adj,    2d+10]
        nonadj_in = torch.stack(nonadj_feats, dim=3)                              # [B, nt, nx, n_nonadj, 2d+11]

        adj_out   = self.adj_msg(
            adj_in.reshape(-1, adj_in.shape[-1])
        ).reshape(B, nt, nx, n_adj, d)                                            # [B, nt, nx, n_adj,    d]
        nonadj_out = self.nonadj_msg(
            nonadj_in.reshape(-1, nonadj_in.shape[-1])
        ).reshape(B, nt, nx, n_nonadj, d)                                         # [B, nt, nx, n_nonadj, d]

        all_msgs  = torch.cat([adj_out, nonadj_out], dim=3)                       # [B, nt, nx, 26, d]
        all_gates_t = torch.stack(all_gates, dim=3)                               # [B, nt, nx, 26, 1]
        agg = (all_gates_t / gate_sum.unsqueeze(3) * all_msgs).sum(dim=3)         # [B, nt, nx, d]

        upd_in = torch.cat([h, agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)
        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class HypNO_ST3(nn.Module):
    """HypNO-ST v3 — separate adjacent / non-adjacent edge MLPs.

    See module docstring for the full list of changes from v2.
    """

    def __init__(
        self,
        stencil_k_x: int = 3,
        stencil_k_t: int = 2,
        d_latent: int = 128,
        d_hidden: int = 128,
        n_layers: int = 6,
        activation: str = "gelu",
        causal_temporal: bool = True,
        radius_x: float | None = None,
        radius_t: float | None = None,
        detector_path: str | None = None,
        detector_cfg: dict | None = None,
        readout: str = "gelu",
        encoder_scaling: str = "physics",
        encoder_type: str = "gnn",
        skip: bool = True,
        use_char_cone: bool = False,
        d_hidden_nonadj: int | None = None,
        use_checkpoint: bool = True,
        mask_same_t_nonadj: bool = True,
        temporal_gate_type: str = "cfl",
        include_flux: bool = True,
        pure_pairwise_edges: bool = False,
        **_ignored,
    ) -> None:
        super().__init__()
        print("=" * 60)
        print("[HypNO_ST3] constructing with parameters:")
        print(f"  stencil_k_x        = {stencil_k_x}")
        print(f"  stencil_k_t        = {stencil_k_t}")
        print(f"  d_latent           = {d_latent}")
        print(f"  d_hidden           = {d_hidden}")
        print(f"  d_hidden_nonadj    = {d_hidden_nonadj}")
        print(f"  n_layers           = {n_layers}")
        print(f"  activation         = {activation}")
        print(f"  causal_temporal    = {causal_temporal}")
        print(f"  radius_x           = {radius_x}")
        print(f"  radius_t           = {radius_t}")
        print(f"  encoder_scaling    = {encoder_scaling}")
        print(f"  encoder_type       = {encoder_type}")
        print(f"  readout            = {readout}")
        print(f"  skip               = {skip}")
        print(f"  use_char_cone      = {use_char_cone}")
        print(f"  use_checkpoint     = {use_checkpoint}")
        print(f"  mask_same_t_nonadj = {mask_same_t_nonadj}")
        print(f"  temporal_gate_type = {temporal_gate_type}")
        print(f"  include_flux       = {include_flux}")
        print(f"  pure_pairwise_edges= {pure_pairwise_edges}")
        print(f"  detector_path      = {detector_path}")
        if _ignored:
            print(f"  IGNORED kwargs     = {sorted(_ignored.keys())}")
        print("=" * 60)
        self.stencil_k_x = stencil_k_x
        self.stencil_k_t = stencil_k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.skip = skip
        self.use_checkpoint = use_checkpoint
        self.include_flux = include_flux
        self.pure_pairwise_edges = pure_pairwise_edges
        self.has_external_detector = False

        self.lifting = _SpaceTimeLiftingLayer(
            d_latent, d_hidden,
            stencil_k_x=stencil_k_x, stencil_k_t=stencil_k_t,
            activation=activation,
            radius_x=radius_x, radius_t=radius_t,
            encoder_scaling=encoder_scaling, encoder_type=encoder_type,
            use_char_cone=use_char_cone,
            causal_temporal=causal_temporal,
            include_flux=include_flux,
            pure_pairwise_edges=pure_pairwise_edges,
        )

        self.mp_layers = nn.ModuleList([
            _PhysicsSpaceTimeMPLayer(
                d_latent, d_hidden, stencil_k_x, stencil_k_t, activation,
                radius_x=radius_x, radius_t=radius_t,
                causal_temporal=causal_temporal,
                use_char_cone=use_char_cone,
                d_hidden_nonadj=d_hidden_nonadj,
                mask_same_t_nonadj=mask_same_t_nonadj,
                temporal_gate_type=temporal_gate_type,
                include_flux=include_flux,
                pure_pairwise_edges=pure_pairwise_edges,
            )
            for _ in range(n_layers)
        ])

        self.decoder = _make_mlp(d_latent, d_hidden, 1, 3, readout)

        if detector_path is not None:
            from hyperbolic_pde.models.shock_detector import ShockDetector
            det_cfg = cfg.get("shock_detector", {})
            self.external_detector = ShockDetector(
                d_latent=int(det_cfg.get("d_latent", 64)),
                d_hidden=int(det_cfg.get("d_hidden", 128)),
                n_layers=int(det_cfg.get("n_layers", 6)),
                activation=str(det_cfg.get("activation", "tanh")),
                ic_points=int(det_cfg.get("ic_points", 128)),
            )
            ckpt = torch.load(detector_path, map_location="cpu", weights_only=True)
            print(f"Loading detector from: {detector_path}")
            self.external_detector.load_state_dict(ckpt)
            self.external_detector.requires_grad_(False)
            self.external_detector.eval()
            self.has_external_detector = True

    def forward(
        self,
        u0: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        edge_feats_adj: torch.Tensor | None = None,
        edge_feats_nonadj: torch.Tensor | None = None,
        diagnose_lifting: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        u0                : [B, nx]
        x                 : [nx]
        t                 : [nt]
        edge_feats_adj    : [B, nx, 3, 13]      — precomputed adjacent edge feats
        edge_feats_nonadj : [B, nx, 2(k-1), 8]  — precomputed non-adj edge feats
                            (or None when k <= 1)
        """
        B, nx = u0.shape
        nt = t.shape[0]

        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)

        if diagnose_lifting:
            h, h_node_only, h_full_lift = self.lifting(
                u0, x, t,
                edge_feats_adj_pre=edge_feats_adj,
                edge_feats_nonadj_pre=edge_feats_nonadj,
                return_intermediates=True,
            )
        else:
            h = self.lifting(
                u0, x, t,
                edge_feats_adj_pre=edge_feats_adj,
                edge_feats_nonadj_pre=edge_feats_nonadj,
            )

        u_hats: list[torch.Tensor] = []
        u_coarse = torch.zeros(B, nt, nx, device=h.device)

        # When diagnosing lifting, prepend two pre-MP probes so the diagnostic
        # stack becomes [post_node, post_full, MP_0, MP_1, ...]. Both pre-MP
        # probes use MP-0's state_probe (biased — trained on post-MP-0 latents
        # — but the spatial pattern is what matters for diagnostics). When
        # diagnose_lifting=False, u_hats is the original per-MP-layer list,
        # so deep-supervision in train scripts is unchanged.
        if diagnose_lifting:
            probe0 = self.mp_layers[0].state_probe
            u_hats.append(torch.sigmoid(probe0(h_node_only)).squeeze(-1))
            u_hats.append(torch.sigmoid(probe0(h_full_lift)).squeeze(-1))

        for layer in self.mp_layers:
            if self.use_checkpoint:
                h = torch_checkpoint(layer, h, x, t, u0, use_reentrant=False)
            else:
                h = layer(h, x, t, u0)
            u_hats.append(torch.sigmoid(layer.state_probe(h)).squeeze(-1))

        shock_indicator = torch.zeros(B, nt, nx, device=h.device)

        out = self.decoder(h).squeeze(-1)
        if self.skip:
            u0_exp = u0.unsqueeze(1).expand(B, nt, nx)
            u_pred = u0_exp + out
        else:
            u_pred = out

        return u_pred, u_coarse, shock_indicator, u_hats
