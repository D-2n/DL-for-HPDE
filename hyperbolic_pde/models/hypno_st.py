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


def precompute_lwr_edge_features(
    u0: torch.Tensor,
    x: torch.Tensor,
    stencil_k: int,
    radius_x: float | None = None,
) -> torch.Tensor:
    """Precompute the 15 static LWR edge features for the lifting layer.

    These features depend only on ``u0`` and ``x`` (not on ``t``), so they can
    be computed once per sample and reused across all time-steps.

    Parameters
    ----------
    u0 : Tensor [N, nx]
        Initial conditions.
    x : Tensor [nx]
        Spatial grid (1-D, shared across samples).
    stencil_k : int
        Stencil half-width configured on the model.
    radius_x : float | None
        If set, the effective stencil is derived from ``radius_x / dx``.

    Returns
    -------
    Tensor [N, nx, 2*k_eff+1, 15]
        Static edge features for every spatial neighbour offset.
    """
    if radius_x is not None:
        dx = (x[1] - x[0]).abs().item()
        k = max(1, int(radius_x / dx + 0.5))
    else:
        k = stencil_k

    N, nx = u0.shape

    u_pad = F.pad(u0.unsqueeze(1), (k, k), mode="replicate").squeeze(1)   # [N, nx+2k]
    x_exp = x.unsqueeze(0).expand(N, -1)                                  # [N, nx]
    x_pad = F.pad(x_exp.unsqueeze(1), (k, k), mode="replicate").squeeze(1)  # [N, nx+2k]

    feats = []
    for j in range(-k, k + 1):
        u_k = u_pad[:, k + j : k + j + nx]     # [N, nx]
        x_k = x_pad[:, k + j : k + j + nx]     # [N, nx]

        du    = u_k - u0
        u_avg = 0.5 * (u0 + u_k)
        rel_x = x_k - x_exp
        abs_dx = rel_x.abs()
        slope  = du / rel_x.abs().clamp(min=1e-6) * rel_x.sign()

        f_i = u0 * (1.0 - u0)
        f_k = u_k * (1.0 - u_k)
        a_i = 1.0 - 2.0 * u0
        a_k = 1.0 - 2.0 * u_k

        du_safe = torch.where(du.abs() < 1e-6, torch.ones_like(du), du)
        a_ik = torch.where(
            du.abs() < 1e-6,
            1.0 - 2.0 * u_avg,       # fallback: f'(u_avg) for vanishing jump
            (f_k - f_i) / du_safe,    # Rankine-Hugoniot interface speed
        )

        sign_a = torch.sign(a_ik)
        upwind = (a_ik * rel_x < 0).float()

        feat = torch.stack([
            u0, u_k,
            du, du.abs(), u_avg,
            rel_x, abs_dx, slope,
            f_i, f_k,
            a_i, a_k, a_ik,
            sign_a, upwind,
        ], dim=-1)                              # [N, nx, 15]
        feats.append(feat)

    return torch.stack(feats, dim=2)            # [N, nx, 2k+1, 15]


class _SpaceTimeLiftingLayer(nn.Module):
    """Joint space-time lifting. Each node (x_i, t_j) receives features
    [u0_i, x_i, t_j], so x and t are never separated.
    Produces h[B, nt, nx, d_latent] directly — no tiling or time-mixing needed.
    """

    def __init__(self, d_latent: int, d_hidden: int, stencil_k: int, activation: str,
                 radius_x: float | None = None) -> None:
        super().__init__()
        self.k = stencil_k
        self.radius_x = radius_x
        self.node_mlp  = _make_mlp(3, d_hidden, d_latent, 2, activation)   # [u0_i, x_i, t_j]
        self.edge_mlp  = _make_mlp(16, d_hidden, d_latent, 2, activation)   # 15 LWR edge feats + t
        #self.gate_net  = _make_mlp(16, d_hidden, 1,       2, activation)
        self.combine   = _make_mlp(2 * d_latent, d_hidden, d_latent, 2, activation)

    def _get_max_k(self, x: torch.Tensor) -> int:
        dx = (x[0, 1] - x[0, 0]).abs().item()
        return max(1, int(self.radius_x / dx + 0.5))

    def forward(
        self,
        u0: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        edge_feats_pre: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # u0: [B, nx],  x: [B, nx],  t: [nt]
        B, nx = u0.shape
        nt = t.shape[0]

        # broadcast all inputs to [B, nt, nx, 1]
        u0_bc = u0.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        x_bc  = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
        t_bc  = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)

        # node embedding
        node_in = torch.cat([u0_bc, x_bc, t_bc], dim=-1)   # [B, nt, nx, 3]
        h_node  = self.node_mlp(node_in)                    # [B, nt, nx, d_latent]

        if edge_feats_pre is not None:
            # --- batched path using precomputed static edge features ---
            # edge_feats_pre: [B, nx, 2k+1, 15]
            n_neighbors = edge_feats_pre.shape[2]
            # expand to [B, nt, nx, 2k+1, 15]
            ef = edge_feats_pre.unsqueeze(1).expand(B, nt, nx, n_neighbors, 15)
            # append t_bc → [B, nt, nx, 2k+1, 16]
            t_exp = t_bc.unsqueeze(3).expand(B, nt, nx, n_neighbors, 1)
            ef = torch.cat([ef, t_exp], dim=-1)

            contrib  = self.edge_mlp(ef)                        # [B, nt, nx, 2k+1, d_latent]
           # gate = torch.sigmoid(self.gate_net(ef))         # [B, nt, nx, 2k+1, 1]
            #contrib = gate * msg                            # [B, nt, nx, 2k+1, d_latent]
            #contrib = msg
            if self.radius_x is not None:
                rel_x = ef[..., 5:6]                        # feature index 5 = rel_x
                contrib = contrib * (rel_x.abs() <= self.radius_x)

            agg = contrib.sum(dim=3)                        # [B, nt, nx, d_latent]
        else:
            # --- fallback: compute edge features on the fly ---
            k = self._get_max_k(x) if self.radius_x is not None else self.k

            u_pad = F.pad(u0.unsqueeze(1), (k, k), mode="replicate").squeeze(1)
            x_pad = F.pad(x.unsqueeze(1),  (k, k), mode="replicate").squeeze(1)

            agg = torch.zeros_like(h_node)
            for j in range(-k, k + 1):
                u_k = u_pad[:, k + j : k + j + nx]
                x_k = x_pad[:, k + j : k + j + nx]

                u_k_bc = u_k.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
                x_k_bc = x_k.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
                rel_x  = x_k_bc - x_bc

                du     = u_k_bc - u0_bc
                abs_dx = rel_x.abs()
                slope  = du / rel_x.abs().clamp(min=1e-6) * rel_x.sign()

                u_avg = 0.5 * (u0_bc + u_k_bc)

                f_i = u0_bc * (1.0 - u0_bc)
                f_k = u_k_bc * (1.0 - u_k_bc)
                a_i = 1.0 - 2.0 * u0_bc
                a_k = 1.0 - 2.0 * u_k_bc

                a_ik = (f_k - f_i) / du.abs().clamp(min=1e-6) * du.sign()
                a_ik = torch.where(du.abs() < 1e-6, 1.0 - 2.0 * u_avg, a_ik)

                sign_a = torch.sign(a_ik)
                upwind = (a_ik * rel_x < 0).float()

                edge_in = torch.cat([
                    u0_bc, u_k_bc,
                    du, du.abs(), u_avg,
                    rel_x, abs_dx, slope,
                    f_i, f_k,
                    a_i, a_k, a_ik,
                    sign_a, upwind,
                    t_bc
                ], dim=-1)
                contrib  = self.edge_mlp(edge_in)
                #gate = torch.sigmoid(self.gate_net(edge_in))
                #contrib = gate * msg

                if self.radius_x is not None:
                    contrib = contrib * (rel_x.abs() <= self.radius_x)
                agg = agg + contrib

        return self.combine(torch.cat([h_node, agg], dim=-1))   # [B, nt, nx, d_latent]


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
    """Factored space-time MP with signed shock attenuation.

    Each layer decodes a provisional scalar state u_hat from the latent field
    and builds LWR-aware edge features from the decoded state.  In shock
    regions (identified by the PINN detector), messages are multiplicatively
    attenuated while preserving sign.

    Spatial edge features:
      (h_i, h_j, rel_x, |dx|, u_hat_i, u_hat_j, du, u_avg, slope,
       f_i, f_j, a_i, a_j, a_ij, sign_a, upwind)
    Temporal edge features:
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

        # decode provisional scalar state from latent field
        self.state_probe = nn.Linear(d_latent, 1)

        # spatial: (h_i, h_j, rel_x, |dx|, u_i, u_j, du, u_avg, slope,
        #           f_i, f_j, a_i, a_j, a_ij, sign_a, upwind)
        sp_in = 2 * d_latent + 14
        self.sp_msg = _make_mlp(sp_in, d_hidden, d_latent, 3, activation)

        # temporal: (h_i, h_j, u_i, u_j, a_i, a_j, rel_t, cfl, x/t)
        tp_in = 2 * d_latent + 7
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
        u0:               [B, nx]   (unused — kept for interface compat)
        shock_indicator:  [B, nt, nx]   (0 = smooth, 1 = strong shock)
        """
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

        # signed attenuation: 1 in smooth regions, 0 at shocks
        s = shock_indicator.unsqueeze(-1)
        tau = 0.2
        p = 3.0

        s_eff = ((s - tau) / (1.0 - tau)).clamp(0.0, 1.0)
        alpha = (1.0 - s_eff).pow(p)
        # decode provisional scalar state from latent field
        u_hat = torch.sigmoid(self.state_probe(h)).squeeze(-1)  # [B, nt, nx]
        u_hat_i = u_hat.unsqueeze(-1)                           # [B, nt, nx, 1]

        # ---- spatial message passing ----
        h_flat = h.reshape(B * nt, nx, d).permute(0, 2, 1)
        h_xp = F.pad(h_flat, (k_x, k_x), mode="replicate")
        h_xp = h_xp.permute(0, 2, 1).reshape(B, nt, nx + 2 * k_x, d)

        u_hat_xp = F.pad(u_hat, (k_x, k_x), mode="replicate")  # [B, nt, nx+2k_x]
        x_pad = F.pad(x.unsqueeze(1), (k_x, k_x), mode="replicate").squeeze(1)
        x_i = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)

        sp_agg = h.new_zeros(B, nt, nx, d)
        for j in range(-k_x, k_x + 1):
            h_j = h_xp[:, :, k_x + j : k_x + j + nx, :]
            x_j_val = x_pad[:, k_x + j : k_x + j + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            u_hat_j = u_hat_xp[:, :, k_x + j : k_x + j + nx].unsqueeze(-1)

            rel_x  = x_j_val - x_i
            abs_dx = rel_x.abs()

            du     = u_hat_j - u_hat_i
            u_avg  = 0.5 * (u_hat_i + u_hat_j)
            slope  = du / abs_dx.clamp(min=1e-6) * rel_x.sign()

            f_i = u_hat_i * (1.0 - u_hat_i)
            f_j = u_hat_j * (1.0 - u_hat_j)
            a_i = 1.0 - 2.0 * u_hat_i
            a_j = 1.0 - 2.0 * u_hat_j

            # Rankine-Hugoniot interface speed, fallback to f'(u_avg) for small du
            du_safe = torch.where(du.abs() < 1e-6, torch.ones_like(du), du)
            a_ij = torch.where(
                du.abs() < 1e-6,
                1.0 - 2.0 * u_avg,
                (f_j - f_i) / du_safe,
            )
            sign_a = torch.sign(a_ij)
            upwind = (a_ij * rel_x < 0).float()

            msg_in = torch.cat([
                h, h_j,
                rel_x, abs_dx,
                u_hat_i.expand_as(rel_x), u_hat_j,
                du, u_avg, slope,
                f_i, f_j, a_i, a_j, a_ij,
                sign_a, upwind,
            ], dim=-1)
            msg = self.sp_msg(msg_in)

            contrib = alpha * msg
            if self.radius_x is not None:
                contrib = contrib * (rel_x.abs() <= self.radius_x)
            sp_agg = sp_agg + contrib

        # ---- temporal message passing (causal: past only) ----
        h_flat_t = h.permute(0, 2, 1, 3).reshape(B * nx, nt, d).permute(0, 2, 1)
        h_tp = F.pad(h_flat_t, (k_t, k_t), mode="replicate")
        h_tp = h_tp.permute(0, 2, 1).reshape(B, nx, nt + 2 * k_t, d)
        h_tp = h_tp.permute(0, 2, 1, 3)                        # [B, nt+2k_t, nx, d]

        u_hat_tp = F.pad(
            u_hat.permute(0, 2, 1), (k_t, k_t), mode="replicate"
        ).permute(0, 2, 1)                                      # [B, nt+2k_t, nx]

        t_pad = F.pad(
            t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate"
        ).squeeze(0).squeeze(0)

        t_range = range(-k_t, 1) if self.causal else range(-k_t, k_t + 1)

        # self-similarity variable x/t (auxiliary, not main physics feature)
        t_i_abs = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)
        x_over_t = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1) / t_i_abs.clamp(min=1e-6)

        a_hat_i = 1.0 - 2.0 * u_hat_i                          # [B, nt, nx, 1]

        tp_agg = h.new_zeros(B, nt, nx, d)
        for j in t_range:
            h_j = h_tp[:, k_t + j : k_t + j + nt, :, :]
            u_hat_j_t = u_hat_tp[:, k_t + j : k_t + j + nt, :].unsqueeze(-1)
            rel_t = (t_pad[k_t + j : k_t + j + nt] - t).view(1, nt, 1, 1).expand(B, nt, nx, 1)

            a_hat_j = 1.0 - 2.0 * u_hat_j_t
            cfl = a_hat_i.abs() * rel_t.abs() / dx_val          # CFL-like propagation ratio

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

        # ---- combine: σ(K(v) + W·v) ----
        upd_in = torch.cat([h, sp_agg + tp_agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)

        return self.act(h_nonlocal + h_local)

class _ClassicSpaceTimeMPLayer(nn.Module):
    """Factored space-time MP with no weighing."""

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
# space-time MP layer with WENO smoothness-weighted messages + causal temporal
# --------------------------------------------------------------------------- #
class _WENOSpaceTimeMPLayer(nn.Module):
    """Factored space-time MP with WENO-inspired smoothness weighting.

    Each layer decodes a provisional scalar state u_hat from the latent field,
    then uses it for both WENO smoothness indicators and LWR-aware edge
    features.

    WENO weight:  ω_j = 1 / (ε + β_j)^p
    where β_j is computed from local differences of the decoded scalar state
    u_hat (not the latent field h).

    Spatial edge features:
      (h_i, h_j, rel_x, |dx|, u_hat_i, u_hat_j, du, u_avg, slope,
       f_i, f_j, a_i, a_j, a_ij, sign_a, upwind)
    Temporal edge features:
      (h_i, h_j, u_hat_i, u_hat_j, a_i, a_j, rel_t, cfl, x/t)

    When unified_mp=True, a single MLP handles both spatial and temporal
    messages with a shared LWR-aware feature set + is_spatial flag.
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

        # decode provisional scalar state from latent field
        self.state_probe = nn.Linear(d_latent, 1)

        if unified_mp:
            # (h_i, h_j, u_i, u_j, f_i, f_j, a_i, a_j, a_ij,
            #  sign_a, upwind, rel_pos, |rel_pos|, cfl, is_spatial)
            uni_in = 2 * d_latent + 13
            self.uni_msg = _make_mlp(uni_in, d_hidden, d_latent, 3, activation)
        else:
            # spatial: (h_i, h_j, rel_x, |dx|, u_i, u_j, du, u_avg, slope,
            #           f_i, f_j, a_i, a_j, a_ij, sign_a, upwind)
            sp_in = 2 * d_latent + 14
            self.sp_msg = _make_mlp(sp_in, d_hidden, d_latent, 3, activation)

            # temporal: (h_i, h_j, u_i, u_j, a_i, a_j, rel_t, cfl, x/t)
            tp_in = 2 * d_latent + 7
            self.tp_msg = _make_mlp(tp_in, d_hidden, d_latent, 3, activation)

        # update + local linear (same σ(K(v) + W·v) structure)
        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
        self.W = nn.Linear(d_latent, d_latent)

    # -- smoothness indicators from decoded scalar state ----------------------

    @staticmethod
    def _scalar_beta_spatial(u: torch.Tensor) -> torch.Tensor:
        """Smoothness from scalar field along spatial dim.  u: [B, nt, nx] -> [B, nt, nx]."""
        diff_fwd = F.pad(u[:, :, 1:] - u[:, :, :-1], (0, 1))
        diff_bwd = F.pad(u[:, :, :-1] - u[:, :, 1:], (1, 0))
        return diff_fwd ** 2 + diff_bwd ** 2

    @staticmethod
    def _scalar_beta_temporal(u: torch.Tensor) -> torch.Tensor:
        """Smoothness from scalar field along temporal dim.  u: [B, nt, nx] -> [B, nt, nx]."""
        diff_fwd = F.pad(u[:, 1:, :] - u[:, :-1, :], (0, 0, 0, 1))
        diff_bwd = F.pad(u[:, :-1, :] - u[:, 1:, :], (0, 0, 1, 0))
        return diff_fwd ** 2 + diff_bwd ** 2

    # -- legacy latent-space beta (kept for visualisation fallback) -----------

    @staticmethod
    def _spatial_beta(h: torch.Tensor) -> torch.Tensor:
        """Smoothness indicator along spatial dim.  h: [B, nt, nx, d] -> [B, nt, nx]."""
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
        """
        h:                [B, nt, nx, d]
        x:                [B, nx]
        t:                [nt]
        u0:               [B, nx]   (unused — kept for interface compat)
        shock_indicator:  [B, nt, nx] optional — from external detector.
                          When provided, messages are weighted by (1 - indicator)
                          instead of WENO smoothness.
        """
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
            ext_weight = (1.0 - shock_indicator).unsqueeze(-1)   # [B, nt, nx, 1]

        # decode provisional scalar state from latent field
        u_hat = torch.sigmoid(self.state_probe(h)).squeeze(-1)   # [B, nt, nx]
        u_hat_i = u_hat.unsqueeze(-1)                            # [B, nt, nx, 1]

        # ---- spatial message passing ----
        h_flat = h.reshape(B * nt, nx, d).permute(0, 2, 1)
        h_xp = F.pad(h_flat, (k_x, k_x), mode="replicate")
        h_xp = h_xp.permute(0, 2, 1).reshape(B, nt, nx + 2 * k_x, d)

        u_hat_xp = F.pad(u_hat, (k_x, k_x), mode="replicate")  # [B, nt, nx+2k_x]
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
            h_j = h_xp[:, :, k_x + j : k_x + j + nx, :]
            x_j_val = x_pad[:, k_x + j : k_x + j + nx].unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1)
            u_hat_j = u_hat_xp[:, :, k_x + j : k_x + j + nx].unsqueeze(-1)

            rel_x  = x_j_val - x_i
            abs_dx = rel_x.abs()

            du     = u_hat_j - u_hat_i
            u_avg  = 0.5 * (u_hat_i + u_hat_j)
            slope  = du / abs_dx.clamp(min=1e-6) * rel_x.sign()

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

            if self.unified_mp:
                cfl_sp = a_ij.abs() * (t[1] - t[0]).abs().item() / dx_val
                msg_in = torch.cat([
                    h, h_j,
                    u_hat_i.expand_as(rel_x), u_hat_j,
                    f_i, f_j, a_i, a_j, a_ij,
                    sign_a, upwind,
                    rel_x, abs_dx, cfl_sp,
                    h.new_ones(B, nt, nx, 1),   # is_spatial = 1
                ], dim=-1)
                msg = self.uni_msg(msg_in)
            else:
                msg_in = torch.cat([
                    h, h_j,
                    rel_x, abs_dx,
                    u_hat_i.expand_as(rel_x), u_hat_j,
                    du, u_avg, slope,
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

        # ---- temporal message passing (causal: past only) ----
        h_flat_t = h.permute(0, 2, 1, 3).reshape(B * nx, nt, d).permute(0, 2, 1)
        h_tp = F.pad(h_flat_t, (k_t, k_t), mode="replicate")
        h_tp = h_tp.permute(0, 2, 1).reshape(B, nx, nt + 2 * k_t, d)
        h_tp = h_tp.permute(0, 2, 1, 3)                         # [B, nt+2k_t, nx, d]

        u_hat_tp = F.pad(
            u_hat.permute(0, 2, 1), (k_t, k_t), mode="replicate"
        ).permute(0, 2, 1)                                       # [B, nt+2k_t, nx]

        t_pad = F.pad(
            t.unsqueeze(0).unsqueeze(0), (k_t, k_t), mode="replicate"
        ).squeeze(0).squeeze(0)

        t_range = range(-k_t, 1) if self.causal else range(-k_t, k_t + 1)

        t_i_abs = t.view(1, nt, 1, 1).expand(B, nt, nx, 1)
        x_over_t = x.unsqueeze(1).unsqueeze(-1).expand(B, nt, nx, 1) / t_i_abs.clamp(min=1e-6)

        a_hat_i = 1.0 - 2.0 * u_hat_i                           # [B, nt, nx, 1]

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
                # for temporal edges: a_ij = avg speed, upwind = 0
                a_avg_t = 0.5 * (a_hat_i + a_hat_j)
                msg_in = torch.cat([
                    h, h_j,
                    u_hat_i.expand_as(rel_t), u_hat_j_t,
                    u_hat_i * (1.0 - u_hat_i),   # f_i
                    u_hat_j_t * (1.0 - u_hat_j_t),  # f_j
                    a_hat_i.expand_as(rel_t), a_hat_j,
                    a_avg_t,                      # a_ij (average for temporal)
                    torch.sign(a_avg_t),          # sign_a
                    h.new_zeros(B, nt, nx, 1),    # upwind = 0 for temporal
                    rel_t, rel_t.abs(), cfl,
                    h.new_zeros(B, nt, nx, 1),    # is_spatial = 0
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

        # ---- combine: σ(K(v) + W·v) ----
        upd_in = torch.cat([h, sp_agg + tp_agg], dim=-1)
        h_nonlocal = self.update_net(upd_in)
        h_local = self.W(h)

        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# main model
# --------------------------------------------------------------------------- #
class HypNO_ST(nn.Module):
    """Hyperbolic Neural Operator with joint space-time lifting.

    Replaces the separate spatial lifting + time embedding pathway with a
    single _SpaceTimeLiftingLayer that directly produces the full
    [B, nt, nx, d_latent] spacetime field.

    Each MP layer decodes a provisional scalar state u_hat from the latent
    field and builds LWR-aware edge features from it (flux, characteristic
    speed, interface speed, upwind direction, CFL ratio).

    shock_mode controls how discontinuities are handled in message passing:
      - "pinn": PINN coarse decoder → PDE residual → shock indicator →
                signed attenuation of messages (smooth, sign-preserving).
                Computed once before MP (static).
      - "weno": WENO smoothness indicator on decoded scalar state u_hat →
                weight messages by 1/(ε + β)^p.
                Recomputed at each MP layer (adaptive).

    Common architecture: P -> [σ(K(v) + W·v)]^T -> Q
    LWR-aware edge features:
      - Spatial: (h_i, h_j, rel_x, |dx|, u_hat_i, u_hat_j, du, u_avg,
                  slope, f_i, f_j, a_i, a_j, a_ij, sign_a, upwind)
      - Temporal: (h_i, h_j, u_hat_i, u_hat_j, a_i, a_j, rel_t, cfl, x/t)
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
    ) -> None:
        super().__init__()
        self.stencil_k_x = stencil_k_x
        self.stencil_k_t = stencil_k_t
        self.radius_x = radius_x
        self.radius_t = radius_t
        self.shock_mode = shock_mode
        self.unified_mp = unified_mp
        self.has_external_detector = False

        # P: joint space-time lifting (produces [B, nt, nx, d_latent] directly)
        self.lifting = _SpaceTimeLiftingLayer(
            d_latent, d_hidden, stencil_k_x, activation, radius_x=radius_x
        )

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
        elif shock_mode == "classic":
            self.mp_layers = nn.ModuleList([
                _ClassicSpaceTimeMPLayer(
                    d_latent, d_hidden, stencil_k_x, stencil_k_t, activation,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        u0: [B, nx]   initial condition
        x:  [nx]      spatial coordinates
        t:  [nt]      time coordinates
        edge_feats_pre: [B, nx, 2k+1, 15] precomputed static edge features (optional)
        Returns:
            u_pred:          [B, nt, nx]  main prediction
            u_coarse:        [B, nt, nx]  PINN coarse prediction (zeros if weno)
            shock_indicator: [B, nt, nx]  shock indicator (WENO spatial beta if weno)
        """
        B, nx = u0.shape
        nt = t.shape[0]

        if x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)

        # --- P: joint space-time lifting ---
        h = self.lifting(u0, x, t, edge_feats_pre=edge_feats_pre)          # [B, nt, nx, d]

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
            dx_val = (x[0, 1] - x[0, 0]).abs().item()
            dt_val = (t[1] - t[0]).abs().item()
            shock_indicator, u_coarse = self.shock_detector(h, dx_val, dt_val)
            shock_indicator_detached = shock_indicator.detach()
            si_for_mp = ext_indicator if ext_indicator is not None else shock_indicator_detached

            for layer in self.mp_layers:
                h = layer(h, x, t, u0, si_for_mp)

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

        # --- Q: decoder with skip from u0 ---
        correction = self.decoder(h).squeeze(-1)                         # [B, nt, nx]
        u0_exp = u0.unsqueeze(1).expand(B, nt, nx)
        u_pred = u0_exp + correction

        return u_pred, u_coarse, shock_indicator
