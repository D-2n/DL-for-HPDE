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
* Each edge-MLP-bearing layer (lifting, PINN MP, Classic MP, WENO MP,
  Physics MP) now owns TWO edge MLPs — one over the adjacent input and
  one over the non-adjacent input.  Messages from both go into the same
  spatial aggregate.
* The non-adjacent MLP can use a smaller hidden width via
  ``d_hidden_nonadj`` (defaults to ``d_hidden``).
* Physics gate: upwind and Oleinik entropy require a_ij and are adj-only.
  On non-adjacent edges the gate is identically 1 (char-cone component
  still applies when ``use_char_cone`` is True — it is node-local).
* Pre-computed edge tensors: ``precompute_lwr_edge_features_v3`` returns
  ``(feats_adj [N,nx,3,13], feats_nonadj [N,nx,2(k-1),8])`` or
  ``(feats_adj, None)`` when ``k <= 1``.
* All st2 functionality toggles preserved: encoder_type (gnn/mlp),
  encoder_scaling (gate_net/upwind/physics), shock_mode
  (pinn/physics/classic/weno), unified_mp, use_char_cone,
  causal_temporal, loss_type, detector_path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
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
parser = argparse.ArgumentParser(description="Instantiate HypNO-ST v3")
parser.add_argument("--config", type=str, default=str(resolve_config_path(ROOT / "configs")))
args, _ = parser.parse_known_args()
cfg = load_config(Path(args.config))
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
# Adjacent static feature vector (13 slots, unchanged from v2):
# 0: u_i,  1: u_k,  2: du,  3: |du|,  4: u_avg,  5: rel_x,
# 6: f_i,  7: f_k,  8: a_i,  9: a_k,  10: a_ij (RH speed),  11: sign(a_ij),  12: upwind
N_EDGE_FEATS_ADJ = 13

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
    feats_adj    : Tensor [N, nx, 3, 13]        — offsets j in {-1, 0, 1}
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

            du_safe = torch.where(du.abs() < 1e-6, torch.ones_like(du), du)
            a_ik = torch.where(
                du.abs() < 1e-6,
                1.0 - 2.0 * u_avg,
                (f_k - f_i) / du_safe,
            )
            sign_a = torch.sign(a_ik)
            upwind = (a_ik * rel_x < 0).float()

            feat = torch.stack([
                u0, u_k,
                du, du.abs(), u_avg,
                rel_x,
                f_i, f_k,
                a_i, a_k, a_ik,
                sign_a, upwind,
            ], dim=-1)                                          # [N, nx, 13]
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
        1.0 - 2.0 * u_avg,
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

    Edge feature vector (unified, `2*0 + 15 = 15` static dims + t
    absorbed via ``t_i, t_j``):
        u0_i, u0_j, f0_i, f0_j, a0_i, a0_j, du0, |du0|,
        rel_x, rel_t, t_i, t_j, a0_ij, sign(a0_ij), is_adj_sp
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
        self.node_mlp = _make_mlp(3, d_hidden, d_latent, 2, activation)
        if encoder_type == "mlp":
            return

        # Static edge feature dim = 15 (see class docstring).
        edge_in = 15
        self.edge_mlp = _make_mlp(edge_in, d_hidden, d_latent, 2, activation)
        # Softmax-normalised attention score over the ball. Physics gate
        # multiplies after softmax (see ``forward``).
        self.attn_score = nn.Linear(edge_in, 1)

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

        node_in = torch.cat([u0_bc, x_bc, t_bc], dim=-1)
        h_node  = self.node_mlp(node_in)

        if self.encoder_type == "mlp":
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

        f0_i = u0_bc * (1.0 - u0_bc)
        a0_i = 1.0 - 2.0 * u0_bc

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
                u_avg = 0.5 * (u0_bc + u0_j)
                du_safe = torch.where(du0.abs() < 1e-6, torch.ones_like(du0), du0)
                a0_ij = torch.where(
                    du0.abs() < 1e-6,
                    1.0 - 2.0 * u_avg,
                    (f0_j - f0_i) / du_safe,
                )
                sign_a0 = torch.sign(a0_ij)
            else:
                a0_ij = torch.zeros_like(du0)
                sign_a0 = torch.zeros_like(du0)
            is_adj_flag = u0_bc.new_full(u0_bc.shape, 1.0 if is_adj_sp else 0.0)
            edge_in = torch.cat([
                u0_bc, u0_j,
                f0_i, f0_j,
                a0_i, a0_j,
                du0, du0.abs(),
                rel_x, rel_t,
                t_bc, t_j,
                a0_ij, sign_a0,
                is_adj_flag,
            ], dim=-1)                                                          # [..., 15]
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

        # ---- pass 1: scores only ----
        scores: list[torch.Tensor] = []
        for di, dm in offsets:
            edge_in, gate, rel_x, rel_t = _build_edge(di, dm)
            score = self.attn_score(edge_in) + torch.log(gate.clamp(min=1e-12))
            if self.radius_x is not None:
                score = score.masked_fill(rel_x.abs() > self.radius_x, float("-inf"))
            if self.radius_t is not None:
                score = score.masked_fill(rel_t.abs() > self.radius_t, float("-inf"))
            scores.append(score)
        alpha = F.softmax(torch.stack(scores, dim=-2), dim=-2)                  # [B, nt, nx, E, 1]

        # ---- pass 2: accumulate alpha_k * msg_k without stacking msgs ----
        agg = torch.zeros_like(h_node)
        for k, (di, dm) in enumerate(offsets):
            edge_in, _, _, _ = _build_edge(di, dm)
            msg = self.edge_mlp(edge_in)
            agg = agg + alpha[..., k, :] * msg

        return self.combine(torch.cat([h_node, agg], dim=-1))


# --------------------------------------------------------------------------- #
# PINN shock detector (unchanged)
# --------------------------------------------------------------------------- #
class _ShockDetectorPINN(nn.Module):
    def __init__(self, d_latent: int, d_hidden: int, activation: str) -> None:
        super().__init__()
        self.coarse_decoder = _make_mlp(d_latent, d_hidden, 1, 2, activation)

    def forward(
        self, h: torch.Tensor, dx: float, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_coarse = self.coarse_decoder(h).squeeze(-1)
        f = u_coarse * (1.0 - u_coarse)
        df_dx = torch.zeros_like(u_coarse)
        df_dx[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dx)
        df_dx[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dx
        df_dx[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dx
        du_dt = torch.zeros_like(u_coarse)
        du_dt[:, 1:-1, :] = (u_coarse[:, 2:, :] - u_coarse[:, :-2, :]) / (2.0 * dt)
        du_dt[:, 0, :] = (u_coarse[:, 1, :] - u_coarse[:, 0, :]) / dt
        du_dt[:, -1, :] = (u_coarse[:, -1, :] - u_coarse[:, -2, :]) / dt
        residual = (du_dt + df_dx).abs()
        r_max = residual.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        shock_indicator = residual / r_max
        return shock_indicator, u_coarse


# --------------------------------------------------------------------------- #
# PINN space-time MP layer (v3)
# --------------------------------------------------------------------------- #
class _PINNSpaceTimeMPLayer(nn.Module):
    """Factored space-time MP (v3). Separate adj / non-adj spatial MLPs."""

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
        d_hidden_nonadj: int | None = None,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_t = k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.causal = causal_temporal
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        self.state_probe = nn.Linear(d_latent, 1)

        # adjacent spatial: (h_i, h_j, rel_x, u_i, u_j, du, u_avg, f_i, f_j,
        #                    a_i, a_j, a_ij, sign_a, upwind) = 2d + 12
        self.sp_msg_adj = _make_mlp(2 * d_latent + 12, d_hidden, d_latent, 3, activation)

        # non-adjacent spatial: (h_i, h_j, u_i, u_j, f_i, f_j, a_i, a_j, x_i, x_j)
        #                     = 2d + 8
        self.sp_msg_nonadj = _make_mlp(2 * d_latent + 8, dh_na, d_latent, 3, activation)

        # temporal unchanged: (h_i, h_j, u_i, u_j, a_i, a_j, rel_t, cfl, x/t) = 2d + 7
        self.tp_msg = _make_mlp(2 * d_latent + 7, d_hidden, d_latent, 3, activation)

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
        B, nt, nx, d = h.shape

        dx_val = (x[0, 1] - x[0, 0]).abs().item()
        if self.radius_x is not None:
            k_x = max(1, int(self.radius_x / dx_val + 0.5))
        else:
            k_x = self.k_x
        if self.radius_t is not None:
            dt_val = (t[1] - t[0]).abs().item()
            k_t = max(1, int(self.radius_t / dt_val + 0.5))
        else:
            k_t = self.k_t

        s = shock_indicator.unsqueeze(-1)
        tau = 0.2
        p = 3.0
        s_eff = ((s - tau) / (1.0 - tau)).clamp(0.0, 1.0)
        alpha = (1.0 - s_eff).pow(p)

        u_hat = torch.sigmoid(self.state_probe(h)).squeeze(-1)
        u_hat_i = u_hat.unsqueeze(-1)

        # ---- spatial MP ----
        h_flat = h.reshape(B * nt, nx, d).permute(0, 2, 1)
        h_xp = F.pad(h_flat, (k_x, k_x), mode="replicate")
        h_xp = h_xp.permute(0, 2, 1).reshape(B, nt, nx + 2 * k_x, d)

        u_hat_xp = F.pad(u_hat, (k_x, k_x), mode="replicate")
        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)

        sp_agg = h.new_zeros(B, nt, nx, d)
        for j in range(-k_x, k_x + 1):
            is_adjacent = abs(j) <= 1
            h_j = h_xp[:, :, k_x + j : k_x + j + nx, :]
            x_j_val = x_pad[:, k_x + j : k_x + j + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            u_hat_j = u_hat_xp[:, :, k_x + j : k_x + j + nx].unsqueeze(-1)

            rel_x = x_j_val - x_i

            if is_adjacent:
                du, u_avg, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind = \
                    _compute_adj_spatial_edge_feats(u_hat_i, u_hat_j, rel_x)
                msg_in = torch.cat([
                    h, h_j,
                    rel_x,
                    u_hat_i.expand_as(rel_x), u_hat_j,
                    du, u_avg,
                    f_i, f_j, a_i, a_j, a_ij,
                    sign_a, upwind,
                ], dim=-1)
                msg = self.sp_msg_adj(msg_in)
            else:
                f_i, f_j, a_i, a_j = _compute_nonadj_pair_feats(u_hat_i, u_hat_j)
                msg_in = torch.cat([
                    h, h_j,
                    u_hat_i.expand_as(rel_x), u_hat_j,
                    f_i, f_j, a_i, a_j,
                    x_i, x_j_val,
                ], dim=-1)
                msg = self.sp_msg_nonadj(msg_in)

            contrib = alpha * msg
            if self.radius_x is not None:
                contrib = contrib * (rel_x.abs() <= self.radius_x)
            sp_agg = sp_agg + contrib

        # ---- temporal MP (causal: past only) ----
        h_flat_t = h.permute(0, 2, 1, 3).reshape(B * nx, nt, d).permute(0, 2, 1)
        h_tp = F.pad(h_flat_t, (k_t, k_t), mode="replicate")
        h_tp = h_tp.permute(0, 2, 1).reshape(B, nx, nt + 2 * k_t, d)
        h_tp = h_tp.permute(0, 2, 1, 3)

        u_hat_tp = F.pad(
            u_hat.permute(0, 2, 1), (k_t, k_t), mode="replicate"
        ).permute(0, 2, 1)

        t_pad = F.pad(
            t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate"
        ).squeeze(0).squeeze(0)

        t_range = range(-k_t, 1) if self.causal else range(-k_t, k_t + 1)

        t_i_abs = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)
        x_over_t = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1) / t_i_abs.clamp(min=1e-6)

        a_hat_i = 1.0 - 2.0 * u_hat_i

        tp_agg = h.new_zeros(B, nt, nx, d)
        for j in t_range:
            h_j = h_tp[:, k_t + j : k_t + j + nt, :, :]
            u_hat_j_t = u_hat_tp[:, k_t + j : k_t + j + nt, :].unsqueeze(-1)
            rel_t = (t_pad[k_t + j : k_t + j + nt] - t).view(1, nt, 1, 1).expand(B, nt, nx, 1)

            a_hat_j = 1.0 - 2.0 * u_hat_j_t
            cfl = a_hat_i.abs() * rel_t.abs() / dx_val

            msg_in = torch.cat([
                h, h_j,
                u_hat_i.expand_as(rel_t), u_hat_j_t,
                a_hat_i.expand_as(rel_t), a_hat_j,
                rel_t, cfl, x_over_t,
            ], dim=-1)
            msg = self.tp_msg(msg_in)

            contrib = alpha * msg
            if self.radius_t is not None:
                contrib = contrib * (rel_t.abs() <= self.radius_t)
            tp_agg = tp_agg + contrib

        upd_in = torch.cat([h, sp_agg + tp_agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)

        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# Classic space-time MP layer (v3)
# --------------------------------------------------------------------------- #
class _ClassicSpaceTimeMPLayer(nn.Module):
    """Space-time MP over a product-box ball, no physics gating.

    A single unified edge MLP consumes edges `(di, dm)` inside the
    Chebyshev ball `|di| <= k_x, |dm| <= k_t`, excluding the centre.
    Causality (``causal_temporal``) drops edges with ``dm > 0``,
    including diagonals.

    Edge feature vector (`2d + 15`):
        h_i, h_j,
        u0_i, u0_j, f0_i, f0_j, a0_i, a0_j,
        du0, |du0|,
        x_i, x_j, t_i, t_j,
        rel_x, rel_t,
        is_adj_sp
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
        **_ignored,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_t = k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.causal = causal_temporal
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()

        # 2d (h_i, h_j) + 15 static edge features (see class docstring).
        self.uni_msg = _make_mlp(2 * d_latent + 15, d_hidden, d_latent, 3, activation)
        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

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

        # Pad h jointly on space and time, plus the 1-D supports.
        h_pad = _pad_space_time(h, k_x, k_t)                                    # [B, nt+2k_t, nx+2k_x, d]
        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)  # [B, nx+2k_x]
        t_pad = F.pad(
            t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate",
        ).squeeze(0).squeeze(0)                                                  # [nt+2k_t]
        u0_pad = F.pad(u0.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1) # [B, nx+2k_x]

        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        t_i = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)
        u0_i = u0.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        f0_i = u0_i * (1.0 - u0_i)
        a0_i = 1.0 - 2.0 * u0_i

        offsets = _enumerate_ball_offsets(k_x, k_t, self.causal)

        agg = h.new_zeros(B, nt, nx, d)
        for di, dm in offsets:
            h_j = h_pad[:, k_t + dm : k_t + dm + nt, k_x + di : k_x + di + nx, :]
            x_j = x_pad[:, k_x + di : k_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            t_j = t_pad[k_t + dm : k_t + dm + nt].view(1, nt, 1, 1).expand(B, nt, nx, 1)
            u0_j = u0_pad[:, k_x + di : k_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)

            rel_x = x_j - x_i
            rel_t = t_j - t_i
            f0_j = u0_j * (1.0 - u0_j)
            a0_j = 1.0 - 2.0 * u0_j
            du0 = u0_j - u0_i

            is_adj_sp = (dm == 0) and (abs(di) == 1)
            is_adj_flag = h.new_full((B, nt, nx, 1), 1.0 if is_adj_sp else 0.0)

            msg_in = torch.cat([
                h, h_j,
                u0_i, u0_j,
                f0_i, f0_j, a0_i, a0_j,
                du0, du0.abs(),
                x_i, x_j, t_i, t_j,
                rel_x, rel_t,
                is_adj_flag,
            ], dim=-1)
            msg = self.uni_msg(msg_in)

            contrib = msg
            if self.radius_x is not None:
                contrib = contrib * (rel_x.abs() <= self.radius_x)
            if self.radius_t is not None:
                contrib = contrib * (rel_t.abs() <= self.radius_t)
            agg = agg + contrib

        upd_in = torch.cat([h, agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)
        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# WENO space-time MP layer (v3)
# --------------------------------------------------------------------------- #
class _WENOSpaceTimeMPLayer(nn.Module):
    """Factored space-time MP with WENO smoothness weighting (v3).

    Separate adj / non-adj spatial MLPs.  In unified_mp mode, ``uni_msg``
    handles adj spatial + temporal; non-adj spatial has its own MLP.
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
        d_hidden_nonadj: int | None = None,
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
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        self.state_probe = nn.Linear(d_latent, 1)

        if unified_mp:
            uni_in = 2 * d_latent + 12
            self.uni_msg = _make_mlp(uni_in, d_hidden, d_latent, 3, activation)
        else:
            self.sp_msg_adj = _make_mlp(2 * d_latent + 12, d_hidden, d_latent, 3, activation)
            self.tp_msg = _make_mlp(2 * d_latent + 7, d_hidden, d_latent, 3, activation)

        self.sp_msg_nonadj = _make_mlp(2 * d_latent + 8, dh_na, d_latent, 3, activation)

        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    @staticmethod
    def _scalar_beta_spatial(u: torch.Tensor) -> torch.Tensor:
        diff_fwd = F.pad(u[:, :, 1:] - u[:, :, :-1], (0, 1))
        diff_bwd = F.pad(u[:, :, :-1] - u[:, :, 1:], (1, 0))
        return diff_fwd ** 2 + diff_bwd ** 2

    @staticmethod
    def _scalar_beta_temporal(u: torch.Tensor) -> torch.Tensor:
        diff_fwd = F.pad(u[:, 1:, :] - u[:, :-1, :], (0, 0, 0, 1))
        diff_bwd = F.pad(u[:, :-1, :] - u[:, 1:, :], (0, 0, 1, 0))
        return diff_fwd ** 2 + diff_bwd ** 2

    @staticmethod
    def _spatial_beta(h: torch.Tensor) -> torch.Tensor:
        diff_fwd = F.pad(h[:, :, 1:, :] - h[:, :, :-1, :], (0, 0, 0, 1))
        diff_bwd = F.pad(h[:, :, :-1, :] - h[:, :, 1:, :], (0, 0, 1, 0))
        return (diff_fwd ** 2 + diff_bwd ** 2).sum(dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        u0: torch.Tensor,
        shock_indicator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, nt, nx, d = h.shape
        use_external = shock_indicator is not None

        dx_val = (x[0, 1] - x[0, 0]).abs().item()
        if self.radius_x is not None:
            k_x = max(1, int(self.radius_x / dx_val + 0.5))
        else:
            k_x = self.k_x
        if self.radius_t is not None:
            dt_val = (t[1] - t[0]).abs().item()
            k_t = max(1, int(self.radius_t / dt_val + 0.5))
        else:
            k_t = self.k_t

        use_weno = self.weno_p > 0 and not use_external

        if use_external:
            ext_weight = (1.0 - shock_indicator).unsqueeze(-1)

        u_hat = torch.sigmoid(self.state_probe(h)).squeeze(-1)
        u_hat_i = u_hat.unsqueeze(-1)

        # ---- spatial MP ----
        h_flat = h.reshape(B * nt, nx, d).permute(0, 2, 1)
        h_xp = F.pad(h_flat, (k_x, k_x), mode="replicate")
        h_xp = h_xp.permute(0, 2, 1).reshape(B, nt, nx + 2 * k_x, d)

        u_hat_xp = F.pad(u_hat, (k_x, k_x), mode="replicate")
        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)

        if use_weno:
            beta_x = self._scalar_beta_spatial(u_hat)
            beta_x_pad = F.pad(beta_x, (k_x, k_x), mode="replicate")
            sp_omega_raw = []
            for j in range(-k_x, k_x + 1):
                beta_j = beta_x_pad[:, :, k_x + j : k_x + j + nx]
                omega_j = 1.0 / (self.weno_eps + beta_j).pow(self.weno_p)
                if self.radius_x is not None:
                    x_j_v = x_pad[:, k_x + j : k_x + j + nx]
                    rel_x_j = (x_j_v - x[:, :nx]).abs()
                    r_mask = (rel_x_j <= self.radius_x).unsqueeze(1).expand_as(omega_j)
                    omega_j = omega_j * r_mask
                sp_omega_raw.append(omega_j)
            sp_omega_sum = torch.stack(sp_omega_raw, dim=0).sum(dim=0).clamp(min=1e-8)

        sp_agg = h.new_zeros(B, nt, nx, d)
        for idx, j in enumerate(range(-k_x, k_x + 1)):
            is_adjacent = abs(j) <= 1
            h_j = h_xp[:, :, k_x + j : k_x + j + nx, :]
            x_j_val = x_pad[:, k_x + j : k_x + j + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            u_hat_j = u_hat_xp[:, :, k_x + j : k_x + j + nx].unsqueeze(-1)

            rel_x = x_j_val - x_i

            if is_adjacent:
                du, u_avg, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind = \
                    _compute_adj_spatial_edge_feats(u_hat_i, u_hat_j, rel_x)

                if self.unified_mp:
                    cfl_sp = a_ij.abs() * (t[1] - t[0]).abs().item() / dx_val
                    msg_in = torch.cat([
                        h, h_j,
                        u_hat_i.expand_as(rel_x), u_hat_j,
                        f_i, f_j, a_i, a_j, a_ij,
                        sign_a, upwind,
                        rel_x, cfl_sp,
                        h.new_ones(B, nt, nx, 1),
                    ], dim=-1)
                    msg = self.uni_msg(msg_in)
                else:
                    msg_in = torch.cat([
                        h, h_j,
                        rel_x,
                        u_hat_i.expand_as(rel_x), u_hat_j,
                        du, u_avg,
                        f_i, f_j, a_i, a_j, a_ij,
                        sign_a, upwind,
                    ], dim=-1)
                    msg = self.sp_msg_adj(msg_in)
            else:
                f_i, f_j, a_i, a_j = _compute_nonadj_pair_feats(u_hat_i, u_hat_j)
                msg_in = torch.cat([
                    h, h_j,
                    u_hat_i.expand_as(rel_x), u_hat_j,
                    f_i, f_j, a_i, a_j,
                    x_i, x_j_val,
                ], dim=-1)
                msg = self.sp_msg_nonadj(msg_in)

            if use_external:
                contrib = msg * ext_weight
                if self.radius_x is not None:
                    contrib = contrib * (rel_x.abs() <= self.radius_x)
            elif use_weno:
                omega_norm = (sp_omega_raw[idx] / sp_omega_sum).unsqueeze(-1)
                contrib = msg * omega_norm
            else:
                contrib = msg
                if self.radius_x is not None:
                    contrib = contrib * (rel_x.abs() <= self.radius_x)

            sp_agg = sp_agg + contrib

        # ---- temporal MP (causal: past only) ----
        h_flat_t = h.permute(0, 2, 1, 3).reshape(B * nx, nt, d).permute(0, 2, 1)
        h_tp = F.pad(h_flat_t, (k_t, k_t), mode="replicate")
        h_tp = h_tp.permute(0, 2, 1).reshape(B, nx, nt + 2 * k_t, d)
        h_tp = h_tp.permute(0, 2, 1, 3)

        u_hat_tp = F.pad(
            u_hat.permute(0, 2, 1), (k_t, k_t), mode="replicate"
        ).permute(0, 2, 1)

        t_pad = F.pad(
            t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate"
        ).squeeze(0).squeeze(0)

        t_range = range(-k_t, 1) if self.causal else range(-k_t, k_t + 1)

        t_i_abs = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)
        x_over_t = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1) / t_i_abs.clamp(min=1e-6)

        a_hat_i = 1.0 - 2.0 * u_hat_i

        if use_weno:
            beta_t = self._scalar_beta_temporal(u_hat)
            beta_t_pad = F.pad(beta_t.permute(0, 2, 1), (k_t, k_t), mode="replicate").permute(0, 2, 1)
            tp_omega_raw = []
            for j in t_range:
                beta_j = beta_t_pad[:, k_t + j : k_t + j + nt, :]
                omega_j = 1.0 / (self.weno_eps + beta_j).pow(self.weno_p)
                if self.radius_t is not None:
                    rel_t_j = (t_pad[k_t + j : k_t + j + nt] - t).abs()
                    r_mask = (rel_t_j <= self.radius_t).view(1, nt, 1).expand_as(omega_j)
                    omega_j = omega_j * r_mask
                tp_omega_raw.append(omega_j)
            tp_omega_sum = torch.stack(tp_omega_raw, dim=0).sum(dim=0).clamp(min=1e-8)

        tp_agg = h.new_zeros(B, nt, nx, d)
        for idx, j in enumerate(t_range):
            h_j = h_tp[:, k_t + j : k_t + j + nt, :, :]
            u_hat_j_t = u_hat_tp[:, k_t + j : k_t + j + nt, :].unsqueeze(-1)
            rel_t = (t_pad[k_t + j : k_t + j + nt] - t).view(1, nt, 1, 1).expand(B, nt, nx, 1)

            a_hat_j = 1.0 - 2.0 * u_hat_j_t
            cfl = a_hat_i.abs() * rel_t.abs() / dx_val

            if self.unified_mp:
                a_avg_t = 0.5 * (a_hat_i + a_hat_j)
                msg_in = torch.cat([
                    h, h_j,
                    u_hat_i.expand_as(rel_t), u_hat_j_t,
                    u_hat_i * (1.0 - u_hat_i),
                    u_hat_j_t * (1.0 - u_hat_j_t),
                    a_hat_i.expand_as(rel_t), a_hat_j,
                    a_avg_t,
                    torch.sign(a_avg_t),
                    h.new_zeros(B, nt, nx, 1),
                    rel_t, cfl,
                    h.new_zeros(B, nt, nx, 1),
                ], dim=-1)
                msg = self.uni_msg(msg_in)
            else:
                msg_in = torch.cat([
                    h, h_j,
                    u_hat_i.expand_as(rel_t), u_hat_j_t,
                    a_hat_i.expand_as(rel_t), a_hat_j,
                    rel_t, cfl, x_over_t,
                ], dim=-1)
                msg = self.tp_msg(msg_in)

            if use_external:
                contrib = msg * ext_weight
            elif use_weno:
                omega_norm = (tp_omega_raw[idx] / tp_omega_sum).unsqueeze(-1)
                contrib = msg * omega_norm
            else:
                contrib = msg
            if self.radius_t is not None:
                contrib = contrib * (rel_t.abs() <= self.radius_t)

            tp_agg = tp_agg + contrib

        upd_in = torch.cat([h, sp_agg + tp_agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)

        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# Physics-gated space-time MP layer (v3)
# --------------------------------------------------------------------------- #
class _PhysicsSpaceTimeMPLayer(nn.Module):
    """Space-time MP over a product-box ball with analytical physics gate.

    A single unified edge MLP covers all `(di, dm)` edges in the
    Chebyshev ball (excluding the centre). The physics gate factors as:

    * ``g_upwind * g_entropy`` -- fires only on adjacent-spatial edges
      (``dm == 0, |di| == 1``); = 1 elsewhere. These are the
      Rankine-Hugoniot-based gates and are undefined for diagonals.
    * ``g_cfl`` -- fires on edges with ``dm != 0`` (any temporal
      component, including diagonals). Soft CFL penalty on
      ``|a_i| * |rel_t| / dx``.
    * ``g_char_cone`` (optional) -- fires on every edge.
      ``char_miss = |rel_x - a_j * rel_t|`` when ``rel_t != 0``, and
      falls back to ``|rel_x + a_j * t_i|`` on pure-spatial edges.

    Edge feature vector (`2d + 15`):
        h_i, h_j,
        u_hat_i, u_hat_j, f_i, f_j, a_i, a_j,
        du, u_avg, a_ij, sign(a_ij), upwind,
        rel_x, rel_t, cfl,
        is_adj_sp
    Adjacent-only features (``du, u_avg, a_ij, sign_a, upwind``) are
    zeroed on non-adj-spatial edges -- they're physically absent there.
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
        **_ignored,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_t = k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.causal = causal_temporal
        self.use_char_cone = use_char_cone
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()

        self.state_probe = nn.Linear(d_latent, 1)

        self.phys_temperature = nn.Parameter(torch.tensor(0.0))
        self.phys_gamma_entropy = nn.Parameter(torch.tensor(-2.0))
        if use_char_cone:
            self.phys_char_width = nn.Parameter(torch.tensor(0.0))
        self.phys_cfl_scale = nn.Parameter(torch.tensor(0.0))

        # 2d (h_i, h_j) + 15 static edge features (see class docstring).
        self.uni_msg = _make_mlp(2 * d_latent + 15, d_hidden, d_latent, 3, activation)
        # Attention score: scalar per edge, softmax-normalised over the
        # ball so neighbour weights sum to 1.  Physics gate multiplies
        # *after* softmax, so normalisation happens among the
        # un-attenuated neighbours.
        self.attn_score = nn.Linear(2 * d_latent + 15, 1)
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
            cfl_scale = F.softplus(self.phys_cfl_scale).clamp(min=1e-6)
            cfl = a_i.abs() * rel_t.abs() / dx_grid
            g_cfl = torch.exp(-cfl_scale * F.relu(cfl - 1.0) ** 2)
            gate = gate * g_cfl

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
            """Return ``(msg_in, gate, rel_x, rel_t)`` for one ball edge.

            Cheap: just tensor slicing + arithmetic. The expensive
            ``uni_msg`` MLP is *not* called here.
            """
            h_j = h_pad[:, k_t + dm : k_t + dm + nt, k_x + di : k_x + di + nx, :]
            u_hat_j = u_hat_pad[:, k_t + dm : k_t + dm + nt, k_x + di : k_x + di + nx].unsqueeze(-1)
            x_j = x_pad[:, k_x + di : k_x + di + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            t_j = t_pad[k_t + dm : k_t + dm + nt].view(1, nt, 1, 1).expand(B, nt, nx, 1)
            rel_x = x_j - x_i
            rel_t = t_j - t_i
            f_j = u_hat_j * (1.0 - u_hat_j)
            a_j = 1.0 - 2.0 * u_hat_j
            is_adj_sp = (dm == 0) and (abs(di) == 1)
            if is_adj_sp:
                du, u_avg, _, _, _, _, a_ij, sign_a, upwind = \
                    _compute_adj_spatial_edge_feats(u_hat_i, u_hat_j, rel_x)
            else:
                zero = torch.zeros_like(rel_x)
                du = u_avg = a_ij = sign_a = upwind = zero
            cfl = a_hat_i.abs() * rel_t.abs() / dx_val
            is_adj_flag = h.new_full((B, nt, nx, 1), 1.0 if is_adj_sp else 0.0)
            msg_in = torch.cat([
                h, h_j,
                u_hat_i.expand_as(rel_x), u_hat_j,
                f_hat_i, f_j, a_hat_i.expand_as(rel_x), a_j,
                du, u_avg, a_ij, sign_a, upwind,
                rel_x, rel_t, cfl,
                is_adj_flag,
            ], dim=-1)                                                          # 2d + 15
            gate = self._ball_physics_gate(
                di=di, dm=dm,
                u_i=u_hat_i, u_j=u_hat_j,
                rel_x=rel_x, rel_t=rel_t,
                a_i=a_hat_i, a_j=a_j, a_ij=a_ij,
                t_i=t_i, dx_grid=dx_val,
            )
            return msg_in, gate, rel_x, rel_t

        # ---- pass 1: scores only (cheap) ----
        scores: list[torch.Tensor] = []
        for di, dm in offsets:
            msg_in, gate, rel_x, rel_t = _build_edge(di, dm)
            score = self.attn_score(msg_in) + torch.log(gate.clamp(min=1e-12))
            if self.radius_x is not None:
                score = score.masked_fill(rel_x.abs() > self.radius_x, float("-inf"))
            if self.radius_t is not None:
                score = score.masked_fill(rel_t.abs() > self.radius_t, float("-inf"))
            scores.append(score)
        alpha = F.softmax(torch.stack(scores, dim=-2), dim=-2)                  # [B, nt, nx, E, 1]

        # ---- pass 2: accumulate alpha_k * msg_k without stacking msgs ----
        agg = h.new_zeros(B, nt, nx, d)
        for k, (di, dm) in enumerate(offsets):
            msg_in, _, _, _ = _build_edge(di, dm)
            msg = self.uni_msg(msg_in)
            agg = agg + alpha[..., k, :] * msg

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
        encoder_scaling: str = "gate_net",
        encoder_type: str = "gnn",
        skip: bool = True,
        use_char_cone: bool = False,
        d_hidden_nonadj: int | None = None,
    ) -> None:
        super().__init__()
        self.stencil_k_x = stencil_k_x
        self.stencil_k_t = stencil_k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.skip = skip
        self.shock_mode = shock_mode
        self.unified_mp = unified_mp
        self.has_external_detector = False

        self.encoder_type = encoder_type
        self.use_char_cone = use_char_cone
        self.lifting = _SpaceTimeLiftingLayer(
            d_latent, d_hidden,
            stencil_k_x=stencil_k_x, stencil_k_t=stencil_k_t,
            activation=activation,
            radius_x=radius_x, radius_t=radius_t,
            encoder_scaling=encoder_scaling, encoder_type=encoder_type,
            use_char_cone=use_char_cone,
            causal_temporal=causal_temporal,
        )

        if shock_mode == "pinn":
            self.shock_detector = _ShockDetectorPINN(d_latent, d_hidden, activation)
        else:
            self.shock_detector = None

        if shock_mode == "weno":
            self.mp_layers = nn.ModuleList([
                _WENOSpaceTimeMPLayer(
                    d_latent, d_hidden, stencil_k_x, stencil_k_t, activation,
                    weno_eps=weno_eps, weno_p=weno_p,
                    radius_x=radius_x, radius_t=radius_t,
                    causal_temporal=causal_temporal, unified_mp=unified_mp,
                    d_hidden_nonadj=d_hidden_nonadj,
                )
                for _ in range(n_layers)
            ])
        elif shock_mode == "physics":
            self.mp_layers = nn.ModuleList([
                _PhysicsSpaceTimeMPLayer(
                    d_latent, d_hidden, stencil_k_x, stencil_k_t, activation,
                    radius_x=radius_x, radius_t=radius_t,
                    causal_temporal=causal_temporal, unified_mp=unified_mp,
                    use_char_cone=use_char_cone,
                    d_hidden_nonadj=d_hidden_nonadj,
                )
                for _ in range(n_layers)
            ])
        elif shock_mode == "classic":
            self.mp_layers = nn.ModuleList([
                _ClassicSpaceTimeMPLayer(
                    d_latent, d_hidden, stencil_k_x, stencil_k_t, activation,
                    radius_x=radius_x, radius_t=radius_t,
                    causal_temporal=causal_temporal, unified_mp=unified_mp,
                    d_hidden_nonadj=d_hidden_nonadj,
                )
                for _ in range(n_layers)
            ])
        else:
            self.mp_layers = nn.ModuleList([
                _PINNSpaceTimeMPLayer(
                    d_latent, d_hidden, stencil_k_x, stencil_k_t, activation,
                    shock_delta=shock_delta, shock_threshold=shock_threshold,
                    radius_x=radius_x, radius_t=radius_t,
                    causal_temporal=causal_temporal,
                    d_hidden_nonadj=d_hidden_nonadj,
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

        h = self.lifting(
            u0, x, t,
            edge_feats_adj_pre=edge_feats_adj,
            edge_feats_nonadj_pre=edge_feats_nonadj,
        )

        ext_indicator = None
        if self.has_external_detector:
            self.external_detector.eval()
            with torch.enable_grad():
                ext_indicator, _ = self.external_detector(u0, x[0] if x.dim() > 1 else x, t)
            ext_indicator = ext_indicator.detach()

        u_hats: list[torch.Tensor] = []

        if self.shock_mode == "pinn":
            dx_val = (x[0, 1] - x[0, 0]).abs().item()
            dt_val = (t[1] - t[0]).abs().item()
            shock_indicator, u_coarse = self.shock_detector(h, dx_val, dt_val)
            shock_indicator_detached = shock_indicator.detach()
            si_for_mp = ext_indicator if ext_indicator is not None else shock_indicator_detached

            for layer in self.mp_layers:
                h = layer(h, x, t, u0, si_for_mp)
                u_hats.append(torch.sigmoid(layer.state_probe(h)).squeeze(-1))

        elif self.shock_mode == "physics":
            u_coarse = torch.zeros(B, nt, nx, device=h.device)

            for layer in self.mp_layers:
                h = layer(h, x, t, u0)
                u_hats.append(torch.sigmoid(layer.state_probe(h)).squeeze(-1))

            shock_indicator = torch.zeros(B, nt, nx, device=h.device)

        elif self.shock_mode == "classic":
            u_coarse = torch.zeros(B, nt, nx, device=h.device)

            for layer in self.mp_layers:
                h = layer(h, x, t, u0, shock_indicator=None)

            if ext_indicator is not None:
                shock_indicator = ext_indicator
            else:
                shock_indicator = torch.zeros(B, nt, nx, device=h.device)

        else:  # weno
            u_coarse = torch.zeros(B, nt, nx, device=h.device)

            for layer in self.mp_layers:
                h = layer(h, x, t, u0, shock_indicator=ext_indicator)
                u_hats.append(torch.sigmoid(layer.state_probe(h)).squeeze(-1))

            if ext_indicator is not None:
                shock_indicator = ext_indicator
            else:
                shock_indicator = _WENOSpaceTimeMPLayer._spatial_beta(h)
                si_max = shock_indicator.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                shock_indicator = shock_indicator / si_max

        out = self.decoder(h).squeeze(-1)
        if self.skip:
            u0_exp = u0.unsqueeze(1).expand(B, nt, nx)
            u_pred = u0_exp + out
        else:
            u_pred = out

        return u_pred, u_coarse, shock_indicator, u_hats
