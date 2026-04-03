"""HypNO-ST v2 — revised edge features and physics gating.

Changes from hypno_st.py
------------------------
* Feature 7 (slope = du / |dx|) removed entirely.
* Inter-node features (du, |du|, u_avg, a_ik, sign(a_ik), upwind) are masked
  to zero for non-adjacent edges (|offset| > 1) because these quantities are
  ill-defined when a shock may sit between the two nodes.
* Physics gate point 3 (shock attenuation) removed.
* Oleinik entropy gate is training-only; at inference it returns 1.  During
  training it uses ground-truth a_L / a_R passed via ``u_true``.
* Edge feature dimension: 14 (was 15).  +1 for time at runtime → 15 total
  input to edge MLP.
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
parser = argparse.ArgumentParser(description="Instantiate HypNO PINN")
parser.add_argument("--config", type=str, default=str(resolve_config_path(ROOT / "configs")))
args = parser.parse_args()
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


# ---- index constants for the 14 static edge features ---------------------- #
# 0: u_i,  1: u_k,  2: du,  3: |du|,  4: u_avg,
# 5: rel_x,  6: |dx|,
# 7: f_i,  8: f_k,  9: a_i,  10: a_k,  11: a_ik,  12: sign(a_ik),  13: upwind
# Features 2-4 and 11-13 are inter-node and get masked for |offset| > 1.
_INTERNODE_INDICES = [2, 3, 4, 11, 12, 13]
N_EDGE_FEATS = 14


def precompute_lwr_edge_features_v2(
    u0: torch.Tensor,
    x: torch.Tensor,
    stencil_k: int,
    radius_x: float | None = None,
) -> torch.Tensor:
    """Precompute the 14 static LWR edge features (v2).

    Inter-node features (du, |du|, u_avg, a_ik, sign_a, upwind) are zeroed
    for non-adjacent edges (|offset| > 1) since they are ill-defined when a
    shock may lie between the two nodes.  Slope is removed entirely.

    Parameters
    ----------
    u0 : Tensor [N, nx]
    x  : Tensor [nx]
    stencil_k : int
    radius_x  : float | None

    Returns
    -------
    Tensor [N, nx, 2*k_eff+1, 14]
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

    feats = []
    for j in range(-k, k + 1):
        u_k = u_pad[:, k + j : k + j + nx]
        x_k = x_pad[:, k + j : k + j + nx]

        du    = u_k - u0
        u_avg = 0.5 * (u0 + u_k)
        rel_x = x_k - x_exp
        abs_dx = rel_x.abs()

        f_i = u0 * (1.0 - u0)
        f_k = u_k * (1.0 - u_k)
        a_i = 1.0 - 2.0 * u0
        a_k = 1.0 - 2.0 * u_k

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
            rel_x, abs_dx,
            f_i, f_k,
            a_i, a_k, a_ik,
            sign_a, upwind,
        ], dim=-1)                                              # [N, nx, 14]

        # mask inter-node features for non-adjacent edges
        is_adjacent = abs(j) <= 1
        if not is_adjacent:
            mask = torch.ones(N_EDGE_FEATS, device=feat.device)
            for idx in _INTERNODE_INDICES:
                mask[idx] = 0.0
            feat = feat * mask

        feats.append(feat)

    return torch.stack(feats, dim=2)                            # [N, nx, 2k+1, 14]


# --------------------------------------------------------------------------- #
# Lifting layer
# --------------------------------------------------------------------------- #
class _SpaceTimeLiftingLayer(nn.Module):
    """Joint space-time lifting (v2).

    Edge MLP input is 15 = 14 static features + t.
    """

    def __init__(self, d_latent: int, d_hidden: int, stencil_k: int, activation: str,
                 radius_x: float | None = None,
                 encoder_scaling: str = "gate_net",
                 encoder_type: str = "gnn") -> None:
        super().__init__()
        self.k = stencil_k
        self.radius_x = radius_x
        self.encoder_scaling = encoder_scaling
        self.encoder_type = encoder_type
        self.node_mlp  = _make_mlp(3, d_hidden, d_latent, 2, activation)
        if encoder_type == "mlp":
            return
        self.edge_mlp  = _make_mlp(N_EDGE_FEATS + 1, d_hidden, d_latent, 2, activation)
        if encoder_scaling == "gate_net":
            self.gate_net  = _make_mlp(N_EDGE_FEATS + 1, d_hidden, 1, 2, activation)
        elif encoder_scaling == "upwind":
            self.upwind_alpha = nn.Parameter(torch.tensor(0.1))
        elif encoder_scaling == "physics":
            # 3 learnable scalars (shock attenuation removed)
            self.phys_temperature = nn.Parameter(torch.tensor(0.0))
            self.phys_gamma_entropy = nn.Parameter(torch.tensor(-2.0))
            self.phys_char_width = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(f"Unknown encoder_scaling: {encoder_scaling!r}")
        self.combine = _make_mlp(2 * d_latent, d_hidden, d_latent, 2, activation)

    def _physics_gate(
        self, u_i: torch.Tensor, u_j: torch.Tensor,
        rel_x: torch.Tensor, a_j: torch.Tensor,
        a_ij: torch.Tensor, t_val: torch.Tensor,
        dx_grid: float,
        u_true_i: torch.Tensor | None = None,
        u_true_j: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Composite physics gate (v2).

        Components (3 learnable scalars):
          1. Soft upwind:  sigmoid(-a_ij * rel_x / temperature)
          2. Oleinik entropy:  training-only, uses GT a_L / a_R
          4. Characteristic cone:  Gaussian centred on char foot from j
        """
        temperature = F.softplus(self.phys_temperature).clamp(min=1e-6)
        gamma_ent   = torch.sigmoid(self.phys_gamma_entropy)
        char_w      = F.softplus(self.phys_char_width).clamp(min=1e-6)

        # 1. Soft upwind
        g_upwind = torch.sigmoid(-a_ij * rel_x / temperature)

        # 2. Oleinik entropy (training only, from GT)
        if self.training and u_true_i is not None and u_true_j is not None:
            u_L = torch.where(rel_x > 0, u_true_i, u_true_j)
            u_R = torch.where(rel_x > 0, u_true_j, u_true_i)
            a_L = 1.0 - 2.0 * u_L
            a_R = 1.0 - 2.0 * u_R
            is_shock = (u_L > u_R).float()
            entropy_ok = ((a_L >= a_ij - 0.01) & (a_ij >= a_R - 0.01)).float()
            g_entropy = 1.0 - is_shock * (1.0 - entropy_ok) * (1.0 - gamma_ent)
        else:
            g_entropy = torch.ones_like(g_upwind)

        # 3. Characteristic cone
        char_miss = (rel_x + a_j * t_val).abs()
        sigma = char_w * (dx_grid + a_j.abs() * t_val.abs().clamp(min=1e-6))
        g_char = torch.exp(-0.5 * (char_miss / sigma.clamp(min=1e-6)) ** 2)

        return g_upwind * g_entropy * g_char

    def _get_max_k(self, x: torch.Tensor) -> int:
        dx = (x[0, 1] - x[0, 0]).abs().item()
        return max(1, int(self.radius_x / dx + 0.5))

    def forward(
        self,
        u0: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        edge_feats_pre: torch.Tensor | None = None,
        u_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, nx = u0.shape
        nt = t.shape[0]

        u0_bc = u0.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        x_bc  = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        t_bc  = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)

        node_in = torch.cat([u0_bc, x_bc, t_bc], dim=-1)
        h_node  = self.node_mlp(node_in)

        if self.encoder_type == "mlp":
            return h_node

        if edge_feats_pre is not None:
            n_neighbors = edge_feats_pre.shape[2]
            ef = edge_feats_pre.unsqueeze(1).expand(B, nt, nx, n_neighbors, N_EDGE_FEATS)
            t_exp = t_bc.unsqueeze(3).expand(B, nt, nx, n_neighbors, 1)
            ef = torch.cat([ef, t_exp], dim=-1)

            msg  = self.edge_mlp(ef)
            if self.encoder_scaling == "gate_net":
                gate = torch.sigmoid(self.gate_net(ef))
            elif self.encoder_scaling == "upwind":
                upwind = ef[..., 13:14]
                alpha = torch.sigmoid(self.upwind_alpha)
                gate = upwind + alpha * (1.0 - upwind)
            else:  # physics
                dx_grid = (x[0, 1] - x[0, 0]).abs().item()
                # For Oleinik during training, broadcast u_true to edge shape
                u_true_i_ef = None
                u_true_j_ef = None
                if self.training and u_true is not None:
                    # u_true: [B, nt, nx] — we need [B, nt, nx, n_neighbors, 1]
                    # u_true_i is just u_true at node i, broadcast across neighbors
                    u_true_i_ef = u_true.unsqueeze(3).unsqueeze(-1).expand(B, nt, nx, n_neighbors, 1)
                    # u_true_j requires padding like edge_feats_pre was built
                    k_eff = n_neighbors // 2
                    u_true_pad = F.pad(u_true, (k_eff, k_eff), mode="replicate")
                    u_true_j_list = []
                    for j in range(-k_eff, k_eff + 1):
                        u_true_j_list.append(u_true_pad[:, :, k_eff + j : k_eff + j + nx])
                    # [B, nt, nx, n_neighbors]
                    u_true_j_ef = torch.stack(u_true_j_list, dim=3).unsqueeze(-1)

                gate = self._physics_gate(
                    u_i=ef[..., 0:1], u_j=ef[..., 1:2],
                    rel_x=ef[..., 5:6], a_j=ef[..., 10:11],
                    a_ij=ef[..., 11:12], t_val=ef[..., N_EDGE_FEATS:N_EDGE_FEATS+1],
                    dx_grid=dx_grid,
                    u_true_i=u_true_i_ef, u_true_j=u_true_j_ef,
                )
            contrib = gate * msg
            if self.radius_x is not None:
                rel_x = ef[..., 5:6]
                contrib = contrib * (rel_x.abs() <= self.radius_x)

            agg = contrib.sum(dim=3)
        else:
            # --- fallback: compute edge features on the fly ---
            k = self._get_max_k(x) if self.radius_x is not None else self.k

            u_pad = F.pad(u0.unsqueeze(1), (k, k), mode="replicate").squeeze(1)
            x_pad = F.pad(x.unsqueeze(1),  (k, k), mode="replicate").squeeze(1)

            # For Oleinik: pad u_true along spatial dim
            u_true_pad = None
            if self.training and u_true is not None:
                u_true_pad = F.pad(u_true, (k, k), mode="replicate")

            agg = torch.zeros_like(h_node)
            for j in range(-k, k + 1):
                is_adjacent = abs(j) <= 1

                u_k = u_pad[:, k + j : k + j + nx]
                x_k = x_pad[:, k + j : k + j + nx]

                u_k_bc = u_k.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
                x_k_bc = x_k.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
                rel_x  = x_k_bc - x_bc

                du     = u_k_bc - u0_bc
                abs_dx = rel_x.abs()
                u_avg  = 0.5 * (u0_bc + u_k_bc)

                f_i = u0_bc * (1.0 - u0_bc)
                f_k = u_k_bc * (1.0 - u_k_bc)
                a_i = 1.0 - 2.0 * u0_bc
                a_k = 1.0 - 2.0 * u_k_bc

                du_safe = torch.where(du.abs() < 1e-6, torch.ones_like(du), du)
                a_ik = torch.where(du.abs() < 1e-6, 1.0 - 2.0 * u_avg, (f_k - f_i) / du_safe)

                sign_a = torch.sign(a_ik)
                upwind = (a_ik * rel_x < 0).float()

                # mask inter-node features for non-adjacent
                if not is_adjacent:
                    du     = torch.zeros_like(du)
                    u_avg  = torch.zeros_like(u_avg)
                    a_ik   = torch.zeros_like(a_ik)
                    sign_a = torch.zeros_like(sign_a)
                    upwind = torch.zeros_like(upwind)

                edge_in = torch.cat([
                    u0_bc, u_k_bc,
                    du, du.abs(), u_avg,
                    rel_x, abs_dx,
                    f_i, f_k,
                    a_i, a_k, a_ik,
                    sign_a, upwind,
                    t_bc,
                ], dim=-1)
                msg = self.edge_mlp(edge_in)

                if self.encoder_scaling == "gate_net":
                    gate = torch.sigmoid(self.gate_net(edge_in))
                elif self.encoder_scaling == "upwind":
                    alpha = torch.sigmoid(self.upwind_alpha)
                    gate = upwind + alpha * (1.0 - upwind)
                else:  # physics
                    dx_grid = (x[0, 1] - x[0, 0]).abs().item()
                    u_true_i_val = None
                    u_true_j_val = None
                    if self.training and u_true_pad is not None:
                        u_true_i_val = u_true[:, :, :].unsqueeze(-1)  # [B, nt, nx, 1]
                        u_true_j_val = u_true_pad[:, :, k + j : k + j + nx].unsqueeze(-1)
                    gate = self._physics_gate(
                        u_i=u0_bc, u_j=u_k_bc,
                        rel_x=rel_x, a_j=a_k,
                        a_ij=a_ik, t_val=t_bc,
                        dx_grid=dx_grid,
                        u_true_i=u_true_i_val, u_true_j=u_true_j_val,
                    )
                contrib = gate * msg

                if self.radius_x is not None:
                    contrib = contrib * (rel_x.abs() <= self.radius_x)
                agg = agg + contrib

        return self.combine(torch.cat([h_node, agg], dim=-1))


# --------------------------------------------------------------------------- #
# PINN shock detector (unchanged from v1)
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
# Helper: compute spatial edge features from decoded state (for MP layers)
# --------------------------------------------------------------------------- #
def _compute_spatial_edge_feats(
    u_hat_i: torch.Tensor, u_hat_j: torch.Tensor,
    rel_x: torch.Tensor, is_adjacent: bool,
) -> tuple[torch.Tensor, ...]:
    """Return (du, u_avg, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind).

    Inter-node features are zeroed when ``is_adjacent`` is False.
    """
    abs_dx = rel_x.abs()
    du     = u_hat_j - u_hat_i
    u_avg  = 0.5 * (u_hat_i + u_hat_j)

    f_i = u_hat_i * (1.0 - u_hat_i)
    f_j = u_hat_j * (1.0 - u_hat_j)
    a_i = 1.0 - 2.0 * u_hat_i
    a_j = 1.0 - 2.0 * u_hat_j

    du_safe = torch.where(du.abs() < 1e-6, torch.ones_like(du), du)
    a_ij = torch.where(
        du.abs() < 1e-6,
        1.0 - 2.0 * u_avg,
        (f_j - f_i) / du_safe,
    )
    sign_a = torch.sign(a_ij)
    upwind = (a_ij * rel_x < 0).float()

    if not is_adjacent:
        z = torch.zeros_like(du)
        du, u_avg, a_ij, sign_a, upwind = z, z, z, z, z

    return du, u_avg, abs_dx, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind


# --------------------------------------------------------------------------- #
# PINN space-time MP layer (v2)
# --------------------------------------------------------------------------- #
class _PINNSpaceTimeMPLayer(nn.Module):
    """Factored space-time MP with signed shock attenuation (v2).

    Spatial edge features (no slope, inter-node masked for non-adjacent):
      (h_i, h_j, rel_x, |dx|, u_hat_i, u_hat_j, du, u_avg,
       f_i, f_j, a_i, a_j, a_ij, sign_a, upwind)
    Temporal edge features (unchanged):
      (h_i, h_j, u_hat_i, u_hat_j, a_i, a_j, rel_t, cfl, x/t)
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
        self.causal = causal_temporal
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()

        self.state_probe = nn.Linear(d_latent, 1)

        # spatial: (h_i, h_j, rel_x, |dx|, u_i, u_j, du, u_avg,
        #           f_i, f_j, a_i, a_j, a_ij, sign_a, upwind) = 2d + 13
        sp_in = 2 * d_latent + 13
        self.sp_msg = _make_mlp(sp_in, d_hidden, d_latent, 3, activation)

        # temporal unchanged: (h_i, h_j, u_i, u_j, a_i, a_j, rel_t, cfl, x/t) = 2d + 7
        tp_in = 2 * d_latent + 7
        self.tp_msg = _make_mlp(tp_in, d_hidden, d_latent, 3, activation)

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
            du, u_avg, abs_dx, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind = \
                _compute_spatial_edge_feats(u_hat_i, u_hat_j, rel_x, is_adjacent)

            msg_in = torch.cat([
                h, h_j,
                rel_x, abs_dx,
                u_hat_i.expand_as(rel_x), u_hat_j,
                du, u_avg,
                f_i, f_j, a_i, a_j, a_ij,
                sign_a, upwind,
            ], dim=-1)
            msg = self.sp_msg(msg_in)

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
# Classic space-time MP layer (v2 — no physics features, just masking du)
# --------------------------------------------------------------------------- #
class _ClassicSpaceTimeMPLayer(nn.Module):
    """Factored space-time MP with no weighing (v2)."""

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
        unified_mp: bool = False,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_t = k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.causal = causal_temporal
        self.unified_mp = unified_mp
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()

        if unified_mp:
            uni_in = 2 * d_latent + 6
            self.uni_msg = _make_mlp(uni_in, d_hidden, d_latent, 3, activation)
        else:
            # (h_i, h_j, x_i, x_j, rel_x, du0, abs_du0) — du0/abs_du0 masked for non-adj
            sp_in = 2 * d_latent + 5
            tp_in = 2 * d_latent + 4
            self.sp_msg = _make_mlp(sp_in, d_hidden, d_latent, 3, activation)
            self.tp_msg = _make_mlp(tp_in, d_hidden, d_latent, 3, activation)

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

        # ---- spatial MP ----
        h_flat = h.reshape(B * nt, nx, d).permute(0, 2, 1)
        h_xp = F.pad(h_flat, (k_x, k_x), mode="replicate")
        h_xp = h_xp.permute(0, 2, 1).reshape(B, nt, nx + 2 * k_x, d)

        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        u0_pad = F.pad(u0.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)

        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)

        sp_agg = h.new_zeros(B, nt, nx, d)
        for j in range(-k_x, k_x + 1):
            is_adjacent = abs(j) <= 1
            h_j = h_xp[:, :, k_x + j : k_x + j + nx, :]
            x_j_val = x_pad[:, k_x + j : k_x + j + nx].unsqueeze(1).unsqueeze(-1)
            rel_x = (x_j_val - x_i)
            du0 = (u0 - u0_pad[:, k_x + j : k_x + j + nx]).unsqueeze(1).unsqueeze(-1)
            abs_du0 = du0.abs()

            # mask du0 for non-adjacent
            if not is_adjacent:
                du0 = torch.zeros_like(du0)
                abs_du0 = torch.zeros_like(abs_du0)

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

            contrib = msg
            if self.radius_x is not None:
                contrib = contrib * (rel_x.abs() <= self.radius_x)

            sp_agg = sp_agg + contrib

        # ---- temporal MP ----
        h_flat_t = h.permute(0, 2, 1, 3).reshape(B * nx, nt, d).permute(0, 2, 1)
        h_tp = F.pad(h_flat_t, (k_t, k_t), mode="replicate")
        h_tp = h_tp.permute(0, 2, 1).reshape(B, nx, nt + 2 * k_t, d)
        h_tp = h_tp.permute(0, 2, 1, 3)

        t_pad = F.pad(t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate").squeeze(0).squeeze(0)
        t_range = range(-k_t, 1) if self.causal else range(-k_t, k_t + 1)

        t_i_abs = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)
        x_over_t = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1) / t_i_abs.clamp(min=1e-6)

        tp_agg = h.new_zeros(B, nt, nx, d)
        for j in t_range:
            h_j = h_tp[:, k_t + j : k_t + j + nt, :, :]
            t_j = t_pad[k_t + j : k_t + j + nt]
            t_j_abs = t_j.view(1, nt, 1, 1).expand(B, nt, nx, 1)
            rel_t = (t_j - t).view(1, nt, 1, 1).expand(B, nt, nx, 1)

            if self.unified_mp:
                zeros = h.new_zeros(B, nt, nx, 1)
                is_sp = zeros
                msg_in = torch.cat([h, h_j, t_i_abs, t_j_abs, rel_t, x_over_t, zeros, is_sp], dim=-1)
                msg = self.uni_msg(msg_in)
            else:
                msg_in = torch.cat([h, h_j, t_i_abs, t_j_abs, rel_t, x_over_t], dim=-1)
                msg = self.tp_msg(msg_in)

            contrib = msg
            if self.radius_t is not None:
                contrib = contrib * (rel_t.abs() <= self.radius_t)

            tp_agg = tp_agg + contrib

        upd_in = torch.cat([h, sp_agg + tp_agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)

        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# WENO space-time MP layer (v2)
# --------------------------------------------------------------------------- #
class _WENOSpaceTimeMPLayer(nn.Module):
    """Factored space-time MP with WENO smoothness weighting (v2).

    Spatial edge features: (h_i, h_j, rel_x, |dx|, u_hat_i, u_hat_j, du, u_avg,
                            f_i, f_j, a_i, a_j, a_ij, sign_a, upwind)
    = 2*d_latent + 13  (slope removed, -1 from v1)

    Inter-node features masked for non-adjacent edges.
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

        self.state_probe = nn.Linear(d_latent, 1)

        if unified_mp:
            # (h_i, h_j, u_i, u_j, f_i, f_j, a_i, a_j, a_ij,
            #  sign_a, upwind, rel_pos, |rel_pos|, cfl, is_spatial)
            uni_in = 2 * d_latent + 13
            self.uni_msg = _make_mlp(uni_in, d_hidden, d_latent, 3, activation)
        else:
            # spatial: (h_i, h_j, rel_x, |dx|, u_i, u_j, du, u_avg,
            #           f_i, f_j, a_i, a_j, a_ij, sign_a, upwind) = 2d + 13
            sp_in = 2 * d_latent + 13
            self.sp_msg = _make_mlp(sp_in, d_hidden, d_latent, 3, activation)

            tp_in = 2 * d_latent + 7
            self.tp_msg = _make_mlp(tp_in, d_hidden, d_latent, 3, activation)

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
            du, u_avg, abs_dx, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind = \
                _compute_spatial_edge_feats(u_hat_i, u_hat_j, rel_x, is_adjacent)

            if self.unified_mp:
                cfl_sp = a_ij.abs() * (t[1] - t[0]).abs().item() / dx_val
                msg_in = torch.cat([
                    h, h_j,
                    u_hat_i.expand_as(rel_x), u_hat_j,
                    f_i, f_j, a_i, a_j, a_ij,
                    sign_a, upwind,
                    rel_x, abs_dx, cfl_sp,
                    h.new_ones(B, nt, nx, 1),
                ], dim=-1)
                msg = self.uni_msg(msg_in)
            else:
                msg_in = torch.cat([
                    h, h_j,
                    rel_x, abs_dx,
                    u_hat_i.expand_as(rel_x), u_hat_j,
                    du, u_avg,
                    f_i, f_j, a_i, a_j, a_ij,
                    sign_a, upwind,
                ], dim=-1)
                msg = self.sp_msg(msg_in)

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
                    rel_t, rel_t.abs(), cfl,
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
                if self.radius_t is not None:
                    contrib = contrib * (rel_t.abs() <= self.radius_t)
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
# Physics-gated space-time MP layer (v2)
# --------------------------------------------------------------------------- #
class _PhysicsSpaceTimeMPLayer(nn.Module):
    """Factored space-time MP with analytical physics gate (v2).

    Changes from v1:
      - Shock attenuation gate (point 3) removed
      - Oleinik entropy gate is training-only, uses GT u_true
      - Slope removed from edge features
      - Inter-node features masked for non-adjacent edges
      - 4 learnable scalars → 3 (spatial) + 1 (temporal CFL)
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
        unified_mp: bool = False,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_t = k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.causal = causal_temporal
        self.unified_mp = unified_mp
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()

        self.state_probe = nn.Linear(d_latent, 1)

        # 3 spatial + 1 temporal learnable scalars
        self.phys_temperature = nn.Parameter(torch.tensor(0.0))
        self.phys_gamma_entropy = nn.Parameter(torch.tensor(-2.0))
        self.phys_char_width = nn.Parameter(torch.tensor(0.0))
        self.phys_cfl_scale = nn.Parameter(torch.tensor(0.0))

        if unified_mp:
            uni_in = 2 * d_latent + 13
            self.uni_msg = _make_mlp(uni_in, d_hidden, d_latent, 3, activation)
        else:
            # 2d + 13 (slope removed → -1 from v1's 2d+14)
            sp_in = 2 * d_latent + 13
            self.sp_msg = _make_mlp(sp_in, d_hidden, d_latent, 3, activation)
            tp_in = 2 * d_latent + 7
            self.tp_msg = _make_mlp(tp_in, d_hidden, d_latent, 3, activation)

        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    def _spatial_physics_gate(
        self, u_i: torch.Tensor, u_j: torch.Tensor,
        rel_x: torch.Tensor, a_j: torch.Tensor,
        a_ij: torch.Tensor,
        t_val: torch.Tensor, dx_grid: float,
        u_true_i: torch.Tensor | None = None,
        u_true_j: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Physics gate for spatial messages (v2): upwind + entropy(train) + char cone."""
        temperature = F.softplus(self.phys_temperature).clamp(min=1e-6)
        gamma_ent   = torch.sigmoid(self.phys_gamma_entropy)
        char_w      = F.softplus(self.phys_char_width).clamp(min=1e-6)

        g_upwind = torch.sigmoid(-a_ij * rel_x / temperature)

        # Oleinik: training only, from GT
        if self.training and u_true_i is not None and u_true_j is not None:
            u_L = torch.where(rel_x > 0, u_true_i, u_true_j)
            u_R = torch.where(rel_x > 0, u_true_j, u_true_i)
            a_L = 1.0 - 2.0 * u_L
            a_R = 1.0 - 2.0 * u_R
            is_shock = (u_L > u_R).float()
            entropy_ok = ((a_L >= a_ij - 0.01) & (a_ij >= a_R - 0.01)).float()
            g_entropy = 1.0 - is_shock * (1.0 - entropy_ok) * (1.0 - gamma_ent)
        else:
            g_entropy = torch.ones_like(g_upwind)

        char_miss = (rel_x + a_j * t_val).abs()
        sigma = char_w * (dx_grid + a_j.abs() * t_val.abs().clamp(min=1e-6))
        g_char = torch.exp(-0.5 * (char_miss / sigma.clamp(min=1e-6)) ** 2)

        return g_upwind * g_entropy * g_char

    def _temporal_physics_gate(
        self, a_i: torch.Tensor, rel_t: torch.Tensor, dx_grid: float,
    ) -> torch.Tensor:
        cfl_scale = F.softplus(self.phys_cfl_scale).clamp(min=1e-6)
        cfl = a_i.abs() * rel_t.abs() / dx_grid
        return torch.exp(-cfl_scale * F.relu(cfl - 1.0) ** 2)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        u0: torch.Tensor,
        shock_indicator: torch.Tensor | None = None,
        u_true: torch.Tensor | None = None,
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

        u_hat = torch.sigmoid(self.state_probe(h)).squeeze(-1)
        u_hat_i = u_hat.unsqueeze(-1)

        # Prepare GT for Oleinik in physics gate
        u_true_xp = None
        if self.training and u_true is not None:
            u_true_xp = F.pad(u_true, (k_x, k_x), mode="replicate")

        # ---- spatial MP ----
        h_flat = h.reshape(B * nt, nx, d).permute(0, 2, 1)
        h_xp = F.pad(h_flat, (k_x, k_x), mode="replicate")
        h_xp = h_xp.permute(0, 2, 1).reshape(B, nt, nx + 2 * k_x, d)

        u_hat_xp = F.pad(u_hat, (k_x, k_x), mode="replicate")
        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)

        t_i = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)

        sp_agg = h.new_zeros(B, nt, nx, d)
        for j in range(-k_x, k_x + 1):
            is_adjacent = abs(j) <= 1
            h_j = h_xp[:, :, k_x + j : k_x + j + nx, :]
            x_j_val = x_pad[:, k_x + j : k_x + j + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            u_hat_j = u_hat_xp[:, :, k_x + j : k_x + j + nx].unsqueeze(-1)

            rel_x = x_j_val - x_i
            du, u_avg, abs_dx, f_i, f_j, a_i_val, a_j_val, a_ij, sign_a, upwind = \
                _compute_spatial_edge_feats(u_hat_i, u_hat_j, rel_x, is_adjacent)

            if self.unified_mp:
                cfl_sp = a_ij.abs() * (t[1] - t[0]).abs().item() / dx_val
                msg_in = torch.cat([
                    h, h_j,
                    u_hat_i.expand_as(rel_x), u_hat_j,
                    f_i, f_j, a_i_val, a_j_val, a_ij,
                    sign_a, upwind,
                    rel_x, abs_dx, cfl_sp,
                    h.new_ones(B, nt, nx, 1),
                ], dim=-1)
                msg = self.uni_msg(msg_in)
            else:
                msg_in = torch.cat([
                    h, h_j,
                    rel_x, abs_dx,
                    u_hat_i.expand_as(rel_x), u_hat_j,
                    du, u_avg,
                    f_i, f_j, a_i_val, a_j_val, a_ij,
                    sign_a, upwind,
                ], dim=-1)
                msg = self.sp_msg(msg_in)

            # physics gate with GT Oleinik
            u_true_i_val = None
            u_true_j_val = None
            if self.training and u_true_xp is not None:
                u_true_i_val = u_true.unsqueeze(-1)  # [B, nt, nx, 1]
                u_true_j_val = u_true_xp[:, :, k_x + j : k_x + j + nx].unsqueeze(-1)

            gate = self._spatial_physics_gate(
                u_i=u_hat_i, u_j=u_hat_j,
                rel_x=rel_x, a_j=a_j_val,
                a_ij=a_ij, t_val=t_i, dx_grid=dx_val,
                u_true_i=u_true_i_val, u_true_j=u_true_j_val,
            )
            contrib = msg * gate
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
                    rel_t, rel_t.abs(), cfl,
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

            gate = self._temporal_physics_gate(
                a_i=a_hat_i, rel_t=rel_t, dx_grid=dx_val,
            )
            contrib = msg * gate
            if self.radius_t is not None:
                contrib = contrib * (rel_t.abs() <= self.radius_t)

            tp_agg = tp_agg + contrib

        upd_in = torch.cat([h, sp_agg + tp_agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)

        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class HypNO_ST2(nn.Module):
    """HypNO-ST v2 — revised edge features and physics gating.

    See module docstring for full list of changes from v1.
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
        self.lifting = _SpaceTimeLiftingLayer(
            d_latent, d_hidden, stencil_k_x, activation, radius_x=radius_x,
            encoder_scaling=encoder_scaling, encoder_type=encoder_type,
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
                )
                for _ in range(n_layers)
            ])
        elif shock_mode == "physics":
            self.mp_layers = nn.ModuleList([
                _PhysicsSpaceTimeMPLayer(
                    d_latent, d_hidden, stencil_k_x, stencil_k_t, activation,
                    radius_x=radius_x, radius_t=radius_t,
                    causal_temporal=causal_temporal, unified_mp=unified_mp,
                )
                for _ in range(n_layers)
            ])
        elif shock_mode == "classic":
            self.mp_layers = nn.ModuleList([
                _ClassicSpaceTimeMPLayer(
                    d_latent, d_hidden, stencil_k_x, stencil_k_t, activation,
                    radius_x=radius_x, radius_t=radius_t,
                    causal_temporal=causal_temporal, unified_mp=unified_mp,
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
            print(f'Loading detector from: {detector_path}')
            self.external_detector.load_state_dict(ckpt)
            self.external_detector.requires_grad_(False)
            self.external_detector.eval()
            self.has_external_detector = True

    def forward(
        self,
        u0: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        edge_feats_pre: torch.Tensor | None = None,
        u_true: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        u0:             [B, nx]   initial condition
        x:              [nx]      spatial coordinates
        t:              [nt]      time coordinates
        edge_feats_pre: [B, nx, 2k+1, 14] precomputed static edge features (v2)
        u_true:         [B, nt, nx] ground truth (optional, for Oleinik during training)

        Returns:
            u_pred, u_coarse, shock_indicator
        """
        B, nx = u0.shape
        nt = t.shape[0]

        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)

        # --- P: joint space-time lifting ---
        h = self.lifting(u0, x, t, edge_feats_pre=edge_feats_pre, u_true=u_true)

        # --- external detector (frozen, if available) ---
        ext_indicator = None
        if self.has_external_detector:
            self.external_detector.eval()
            with torch.enable_grad():
                ext_indicator, _ = self.external_detector(u0, x[0] if x.dim() > 1 else x, t)
            ext_indicator = ext_indicator.detach()

        if self.shock_mode == "pinn":
            dx_val = (x[0, 1] - x[0, 0]).abs().item()
            dt_val = (t[1] - t[0]).abs().item()
            shock_indicator, u_coarse = self.shock_detector(h, dx_val, dt_val)
            shock_indicator_detached = shock_indicator.detach()
            si_for_mp = ext_indicator if ext_indicator is not None else shock_indicator_detached

            for layer in self.mp_layers:
                h = layer(h, x, t, u0, si_for_mp)

        elif self.shock_mode == "physics":
            u_coarse = torch.zeros(B, nt, nx, device=h.device)

            for layer in self.mp_layers:
                h = layer(h, x, t, u0, u_true=u_true)

            shock_indicator = torch.zeros(B, nt, nx, device=h.device)

        elif self.shock_mode == "classic":
            u_coarse = torch.zeros(B, nt, nx, device=h.device)

            for layer in self.mp_layers:
                h = layer(h, x, t, u0, shock_indicator=None)

            if ext_indicator is not None:
                shock_indicator = ext_indicator
            else:
                shock_indicator = torch.zeros(B, nt, nx, device=h.device)

        else:
            u_coarse = torch.zeros(B, nt, nx, device=h.device)

            for layer in self.mp_layers:
                h = layer(h, x, t, u0, shock_indicator=ext_indicator)

            if ext_indicator is not None:
                shock_indicator = ext_indicator
            else:
                shock_indicator = _WENOSpaceTimeMPLayer._spatial_beta(h)
                si_max = shock_indicator.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                shock_indicator = shock_indicator / si_max

        # --- Q: decoder ---
        out = self.decoder(h).squeeze(-1)
        if self.skip:
            u0_exp = u0.unsqueeze(1).expand(B, nt, nx)
            u_pred = u0_exp + out
        else:
            u_pred = out

        return u_pred, u_coarse, shock_indicator
