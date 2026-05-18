"""HypNO-ST v3 — 2D scalar LWR variant.

Fork of ``hypno_st3.py`` for the isotropic 2D conservation law
    u_t + f(u)_x + g(u)_y = 0,    f(u) = g(u) = u(1-u)
    a(u) = f'(u) = 1 - 2u,        b(u) = g'(u) = 1 - 2u

Differences from the 1D model
-----------------------------
* All latents are ``[B, nt, ny, nx, d]`` (one extra spatial axis).
* Space-time stencil is a 3D Chebyshev ball ``(di_x, di_y, dm)`` with
  ``|di_x| <= k_x``, ``|di_y| <= k_y``, ``-k_t <= dm <= 0`` (causal).
* Three adjacency classes instead of two:
    - ``adj_x``  : ``dm==0, |di_x|==1, di_y==0``     -> x-flux physics gate
    - ``adj_y``  : ``dm==0, di_x==0, |di_y|==1``     -> y-flux physics gate
    - ``nonadj`` : everything else                   -> CFL gate only
* Per-axis Rankine-Hugoniot speeds: ``a_ij`` along x, ``b_ij`` along y.

Fixed configuration (from the 1D ablation findings)
---------------------------------------------------
The 1D ablation suite showed two settings consistently win, so they are
baked in here rather than left as flags:

* **Flux on the node MLP.** Node MLP input is ``[u0, x, y, t, f0, a0,
  g0, b0]`` (8 dims). ``f0=g0`` and ``a0=b0`` under isotropic flux but
  kept as separate fields so an anisotropic flux needs no input-dim
  change.
* **Pure-pairwise edges.** All edge MLPs (lifting + MP) consume only
  pair-intrinsic features plus the latents ``h_i, h_j``. No broadcast
  node-natives (``u0_i, f0_i, ...``) on edges -- neighbour state reaches
  the edge MLP only through the latent ``h_j``.

The physics gate is always on (no ``classic`` mode).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


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
    """3D product-box space-time stencil (Chebyshev ball).

    Returns all ``(di_x, di_y, dm)`` with ``|di_x| <= k_x``,
    ``|di_y| <= k_y`` and ``|dm| <= k_t``, excluding the centre
    ``(0, 0, 0)``. With ``causal=True``, only ``dm <= 0`` is returned.
    """
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
    """Replicate-pad ``h [B, nt, ny, nx, d]`` on all three grid axes.

    Output shape ``[B, nt+2k_t, ny+2k_y, nx+2k_x, d]``.
    """
    # F.pad on a 5D tensor pads the last 3 dims; we want to pad nt, ny, nx.
    # Move d to position 1 -> [B, d, nt, ny, nx], pad (x, y, t), move back.
    h_pad = F.pad(
        h.permute(0, 4, 1, 2, 3),                      # [B, d, nt, ny, nx]
        (k_x, k_x, k_y, k_y, k_t, k_t),
        mode="replicate",
    ).permute(0, 2, 3, 4, 1)                           # [B, nt+2k_t, ny+2k_y, nx+2k_x, d]
    return h_pad


def _adjacency_class(di_x: int, di_y: int, dm: int) -> str:
    """Classify a ball offset as 'adj_x', 'adj_y', or 'nonadj'."""
    if dm == 0 and abs(di_x) == 1 and di_y == 0:
        return "adj_x"
    if dm == 0 and di_x == 0 and abs(di_y) == 1:
        return "adj_y"
    return "nonadj"


def _axial_physics_gate(
    u_i: torch.Tensor, u_j: torch.Tensor,
    rel: torch.Tensor, s_ij: torch.Tensor,
    temperature: torch.Tensor, gamma_ent: torch.Tensor,
) -> torch.Tensor:
    """Upwind x Oleinik entropy gate along one axis.

    ``rel`` is the relative coordinate along the edge's axis (``rel_x``
    for adj_x, ``rel_y`` for adj_y) and ``s_ij`` is the corresponding RH
    speed (``a_ij`` / ``b_ij``).
    """
    g_upwind = torch.sigmoid(-s_ij * torch.sign(rel) / temperature)

    u_L = torch.where(rel > 0, u_i, u_j)
    u_R = torch.where(rel > 0, u_j, u_i)
    a_L = 1.0 - 2.0 * u_L
    a_R = 1.0 - 2.0 * u_R
    is_shock = (u_L > u_R).float()
    entropy_ok = ((a_L >= s_ij - 0.01) & (s_ij >= a_R - 0.01)).float()
    g_entropy = 1.0 - is_shock * (1.0 - entropy_ok) * (1.0 - gamma_ent)
    return g_upwind * g_entropy


# Edge MLP input dims (pure-pairwise, fixed).
#   lifting edge: [du0, sign(rel_x), sign(rel_y), rel_t, t_i, t_j,
#                  a0_ij, b0_ij, is_adj_x, is_adj_y]                 -> 10
#   MP adj_x/adj_y: [h_i, h_j, du, s_ij, sign(s_ij), upwind, sign(rel)] -> 2d + 5
#   MP nonadj: [h_i, h_j, du, rel_x, rel_y, rel_t, sign(rel_x), sign(rel_y)] -> 2d + 6
_LIFT_EDGE_DIM = 10
_MP_ADJ_EXTRA = 5
_MP_NONADJ_EXTRA = 6


# --------------------------------------------------------------------------- #
# Lifting layer
# --------------------------------------------------------------------------- #
class _SpaceTimeLiftingLayer2D(nn.Module):
    """2D space-time lifting over a 3D product-box ball neighbourhood.

    Node MLP input (8 dims, flux on node):
        u0, x, y, t, f0, a0, g0, b0

    Pure-pairwise edge MLP input (10 dims):
        du0, sign(rel_x), sign(rel_y), rel_t, t_i, t_j,
        a0_ij, b0_ij, is_adj_x, is_adj_y

    The per-axis RH speeds ``a0_ij`` / ``b0_ij`` are non-zero only on the
    corresponding adjacency class; on all other edges they are 0.

    Physics gate: the x-upwind/entropy component fires only on ``adj_x``
    edges, the y-component only on ``adj_y`` edges; ``nonadj`` edges pass
    through with gate 1.
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        stencil_k_x: int,
        stencil_k_y: int,
        stencil_k_t: int,
        activation: str = "gelu",
        encoder_type: str = "gnn",
        causal_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.k_x = stencil_k_x
        self.k_y = stencil_k_y
        self.k_t = stencil_k_t
        self.encoder_type = encoder_type
        self.causal = causal_temporal

        self.node_mlp = _make_mlp(8, d_hidden, d_latent, 2, activation)
        print("[Lifting2D] node_mlp in=8 (u0, x, y, t, f0, a0, g0, b0)")
        if encoder_type == "mlp":
            return

        self.edge_mlp = _make_mlp(_LIFT_EDGE_DIM, d_hidden, d_latent, 2, activation)
        print(f"[Lifting2D] edge_mlp in={_LIFT_EDGE_DIM} (pure-pairwise)")

        # Per-axis learnable physics-gate parameters.
        self.phys_temperature_x = nn.Parameter(torch.tensor(0.0))
        self.phys_temperature_y = nn.Parameter(torch.tensor(0.0))
        self.phys_gamma_entropy_x = nn.Parameter(torch.tensor(-2.0))
        self.phys_gamma_entropy_y = nn.Parameter(torch.tensor(-2.0))

        self.combine = _make_mlp(2 * d_latent, d_hidden, d_latent, 2, activation)

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

        # Broadcast node coordinates to [B, nt, ny, nx, 1].
        u0_bc = u0.view(B, 1, ny, nx, 1).expand(B, nt, ny, nx, 1)
        x_bc = x.view(1, 1, 1, nx, 1).expand(B, nt, ny, nx, 1)
        y_bc = y.view(1, 1, ny, 1, 1).expand(B, nt, ny, nx, 1)
        t_bc = t.view(1, nt, 1, 1, 1).expand(B, nt, ny, nx, 1)

        f0_i = u0_bc * (1.0 - u0_bc)
        g0_i = f0_i                          # isotropic flux: g == f
        a0_i = 1.0 - 2.0 * u0_bc
        b0_i = a0_i

        node_in = torch.cat([u0_bc, x_bc, y_bc, t_bc, f0_i, a0_i, g0_i, b0_i], dim=-1)
        h_node = self.node_mlp(node_in)

        if self.encoder_type == "mlp":
            if return_intermediates:
                return h_node, h_node, h_node
            return h_node

        k_x, k_y, k_t = self.k_x, self.k_y, self.k_t

        # Pad u0 spatially (t-invariant), and pad the coordinate vectors.
        u0_pad = F.pad(u0, (k_x, k_x, k_y, k_y), mode="replicate")    # [B, ny+2k_y, nx+2k_x]
        x_pad = F.pad(x.view(1, 1, nx), (k_x, k_x), mode="replicate").view(-1)
        y_pad = F.pad(y.view(1, 1, ny), (k_y, k_y), mode="replicate").view(-1)
        t_pad = F.pad(t.view(1, 1, nt), (k_t, k_t), mode="replicate").view(-1)

        offsets = _enumerate_ball_offsets_2d(k_x, k_y, k_t, self.causal)

        edge_inputs: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []

        for di_x, di_y, dm in offsets:
            cls = _adjacency_class(di_x, di_y, dm)

            u0_j = u0_pad[
                :, k_y + di_y : k_y + di_y + ny, k_x + di_x : k_x + di_x + nx,
            ].view(B, 1, ny, nx, 1).expand(B, nt, ny, nx, 1)
            x_j = x_pad[k_x + di_x : k_x + di_x + nx].view(1, 1, 1, nx, 1).expand(B, nt, ny, nx, 1)
            y_j = y_pad[k_y + di_y : k_y + di_y + ny].view(1, 1, ny, 1, 1).expand(B, nt, ny, nx, 1)
            t_j = t_pad[k_t + dm : k_t + dm + nt].view(1, nt, 1, 1, 1).expand(B, nt, ny, nx, 1)

            rel_x = x_j - x_bc
            rel_y = y_j - y_bc
            rel_t = t_j - t_bc
            f0_j = u0_j * (1.0 - u0_j)
            g0_j = f0_j
            du0 = u0_j - u0_bc

            a0_ij = torch.zeros_like(du0)
            b0_ij = torch.zeros_like(du0)
            du_safe = torch.where(du0.abs() < 1e-6, torch.ones_like(du0), du0)
            if cls == "adj_x":
                a0_ij = torch.where(
                    du0.abs() < 1e-6, a0_i.expand_as(du0), (f0_j - f0_i) / du_safe,
                )
            elif cls == "adj_y":
                b0_ij = torch.where(
                    du0.abs() < 1e-6, b0_i.expand_as(du0), (g0_j - g0_i) / du_safe,
                )

            is_adj_x = u0_bc.new_full(u0_bc.shape, 1.0 if cls == "adj_x" else 0.0)
            is_adj_y = u0_bc.new_full(u0_bc.shape, 1.0 if cls == "adj_y" else 0.0)

            edge_in = torch.cat([
                du0,
                torch.sign(rel_x), torch.sign(rel_y), rel_t,
                t_bc, t_j,
                a0_ij, b0_ij,
                is_adj_x, is_adj_y,
            ], dim=-1)                                                # [..., 10]

            if cls == "adj_x":
                temp = F.softplus(self.phys_temperature_x).clamp(min=1e-6)
                gam = torch.sigmoid(self.phys_gamma_entropy_x)
                gate = _axial_physics_gate(u0_bc, u0_j, rel_x, a0_ij, temp, gam)
            elif cls == "adj_y":
                temp = F.softplus(self.phys_temperature_y).clamp(min=1e-6)
                gam = torch.sigmoid(self.phys_gamma_entropy_y)
                gate = _axial_physics_gate(u0_bc, u0_j, rel_y, b0_ij, temp, gam)
            else:
                gate = torch.ones_like(du0)

            edge_inputs.append(edge_in)
            gates.append(gate)

        # Additive floor (not clamp): if every edge legitimately suppresses,
        # the aggregate attenuates to 0 rather than blowing up via 1/eps.
        gate_sum = torch.stack(gates, dim=-2).sum(dim=-2) + 1e-3       # [B,nt,ny,nx,1]

        n_offsets = len(edge_inputs)
        edge_in_stacked = torch.stack(edge_inputs, dim=4)             # [B,nt,ny,nx,n_off,10]
        msgs = self.edge_mlp(
            edge_in_stacked.reshape(-1, edge_in_stacked.shape[-1])
        ).reshape(B, nt, ny, nx, n_offsets, -1)                       # [B,nt,ny,nx,n_off,d]
        gates_t = torch.stack(gates, dim=4)                           # [B,nt,ny,nx,n_off,1]
        agg = (gates_t / gate_sum.unsqueeze(4) * msgs).sum(dim=4)     # [B,nt,ny,nx,d]

        h_full = self.combine(torch.cat([h_node, agg], dim=-1))
        if return_intermediates:
            return h_full, h_node, h_full
        return h_full


# --------------------------------------------------------------------------- #
# Physics-gated space-time MP layer (2D)
# --------------------------------------------------------------------------- #
class _PhysicsSpaceTimeMPLayer2D(nn.Module):
    """2D space-time MP over a 3D ball with analytical physics gates.

    Three pure-pairwise edge MLPs routed by adjacency class:
      * ``adj_x_msg``  (2d + 5): x-adjacent edges; x-upwind x x-entropy gate.
      * ``adj_y_msg``  (2d + 5): y-adjacent edges; y-upwind x y-entropy gate.
      * ``nonadj_msg`` (2d + 6): temporal / diagonal / long-range edges;
        CFL gate only. Pure-spatial non-axial edges carry no message.

    Aggregation is gate-normalised: ``w = gate / sum(gate)``.

    adj_x features (2d + 5):
        h_i, h_j, du, a_ij, sign(a_ij), upwind_x, sign(rel_x)
    adj_y is the mirror with b_ij, rel_y.
    nonadj features (2d + 6):
        h_i, h_j, du, rel_x, rel_y, rel_t, sign(rel_x), sign(rel_y)
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        k_x: int,
        k_y: int,
        k_t: int,
        activation: str = "gelu",
        causal_temporal: bool = True,
        d_hidden_nonadj: int | None = None,
        shared_decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.k_x = k_x
        self.k_y = k_y
        self.k_t = k_t
        self.causal = causal_temporal
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        if shared_decoder is None:
            raise ValueError("_PhysicsSpaceTimeMPLayer2D requires a shared_decoder.")
        # Bypass nn.Module.__setattr__ so the decoder's parameters are not
        # double-registered on this MP layer.
        object.__setattr__(self, "_shared_decoder", shared_decoder)

        # Per-axis physics gate parameters.
        self.phys_temperature_x = nn.Parameter(torch.tensor(0.0))
        self.phys_temperature_y = nn.Parameter(torch.tensor(0.0))
        self.phys_gamma_entropy_x = nn.Parameter(torch.tensor(-2.0))
        self.phys_gamma_entropy_y = nn.Parameter(torch.tensor(-2.0))
        self.phys_cfl_scale = nn.Parameter(torch.tensor(0.0))

        self.adj_x_msg = _make_mlp(2 * d_latent + _MP_ADJ_EXTRA, d_hidden, d_latent, 3, activation)
        self.adj_y_msg = _make_mlp(2 * d_latent + _MP_ADJ_EXTRA, d_hidden, d_latent, 3, activation)
        self.nonadj_msg = _make_mlp(2 * d_latent + _MP_NONADJ_EXTRA, dh_na, d_latent, 3, activation)
        print(
            f"[MP2D] pure-pairwise | "
            f"adj_x/adj_y in=2*{d_latent}+{_MP_ADJ_EXTRA}={2 * d_latent + _MP_ADJ_EXTRA} | "
            f"nonadj in=2*{d_latent}+{_MP_NONADJ_EXTRA}={2 * d_latent + _MP_NONADJ_EXTRA}"
        )

        self.update_net = _make_mlp(2 * d_latent, d_hidden, d_latent, 3, activation)
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
        dx_val = float((x[1] - x[0]).abs())
        dy_val = float((y[1] - y[0]).abs())
        k_x, k_y, k_t = self.k_x, self.k_y, self.k_t

        # Internal u_hat for the physics gates: shared decoder + clamp to
        # [0, 1] so the gates see admissible densities even early in training.
        u_hat = self._shared_decoder(h).squeeze(-1).clamp(0.0, 1.0)     # [B,nt,ny,nx]
        u_hat_i = u_hat.unsqueeze(-1)                                   # [B,nt,ny,nx,1]
        a_hat_i = 1.0 - 2.0 * u_hat_i
        f_hat_i = u_hat_i * (1.0 - u_hat_i)

        h_pad = _pad_space_time_2d(h, k_x, k_y, k_t)
        u_hat_pad = _pad_space_time_2d(u_hat.unsqueeze(-1), k_x, k_y, k_t).squeeze(-1)
        x_pad = F.pad(x.view(1, 1, nx), (k_x, k_x), mode="replicate").view(-1)
        y_pad = F.pad(y.view(1, 1, ny), (k_y, k_y), mode="replicate").view(-1)
        t_pad = F.pad(t.view(1, 1, nt), (k_t, k_t), mode="replicate").view(-1)

        x_i = x.view(1, 1, 1, nx, 1).expand(B, nt, ny, nx, 1)
        y_i = y.view(1, 1, ny, 1, 1).expand(B, nt, ny, nx, 1)
        t_i = t.view(1, nt, 1, 1, 1).expand(B, nt, ny, nx, 1)

        offsets = _enumerate_ball_offsets_2d(k_x, k_y, k_t, self.causal)

        adj_x_feats: list[torch.Tensor] = []
        adj_y_feats: list[torch.Tensor] = []
        nonadj_feats: list[torch.Tensor] = []
        adj_x_gates: list[torch.Tensor] = []
        adj_y_gates: list[torch.Tensor] = []
        nonadj_gates: list[torch.Tensor] = []

        for di_x, di_y, dm in offsets:
            cls = _adjacency_class(di_x, di_y, dm)
            # Pure-spatial non-axial edges (dm==0, not adj_x/adj_y) carry no
            # message: skip them entirely.
            if dm == 0 and cls == "nonadj":
                continue

            h_j = h_pad[
                :, k_t + dm : k_t + dm + nt,
                k_y + di_y : k_y + di_y + ny,
                k_x + di_x : k_x + di_x + nx, :,
            ]
            u_hat_j = u_hat_pad[
                :, k_t + dm : k_t + dm + nt,
                k_y + di_y : k_y + di_y + ny,
                k_x + di_x : k_x + di_x + nx,
            ].unsqueeze(-1)
            x_j = x_pad[k_x + di_x : k_x + di_x + nx].view(1, 1, 1, nx, 1).expand(B, nt, ny, nx, 1)
            y_j = y_pad[k_y + di_y : k_y + di_y + ny].view(1, 1, ny, 1, 1).expand(B, nt, ny, nx, 1)
            t_j = t_pad[k_t + dm : k_t + dm + nt].view(1, nt, 1, 1, 1).expand(B, nt, ny, nx, 1)

            rel_x = x_j - x_i
            rel_y = y_j - y_i
            rel_t = t_j - t_i
            f_j = u_hat_j * (1.0 - u_hat_j)
            g_j = f_j
            du = u_hat_j - u_hat_i
            du_safe = torch.where(du.abs() < 1e-6, torch.ones_like(du), du)

            if cls == "adj_x":
                a_ij = torch.where(
                    du.abs() < 1e-6, a_hat_i.expand_as(du), (f_j - f_hat_i) / du_safe,
                )
                upwind_x = (a_ij * rel_x < 0).float()
                msg_in = torch.cat([
                    h, h_j,
                    du, a_ij, torch.sign(a_ij), upwind_x, torch.sign(rel_x),
                ], dim=-1)                                              # 2d + 5
                temp = F.softplus(self.phys_temperature_x).clamp(min=1e-6)
                gam = torch.sigmoid(self.phys_gamma_entropy_x)
                gate = _axial_physics_gate(u_hat_i, u_hat_j, rel_x, a_ij, temp, gam)
                adj_x_feats.append(msg_in)
                adj_x_gates.append(gate)

            elif cls == "adj_y":
                b_ij = torch.where(
                    du.abs() < 1e-6, a_hat_i.expand_as(du), (g_j - f_hat_i) / du_safe,
                )
                upwind_y = (b_ij * rel_y < 0).float()
                msg_in = torch.cat([
                    h, h_j,
                    du, b_ij, torch.sign(b_ij), upwind_y, torch.sign(rel_y),
                ], dim=-1)                                              # 2d + 5
                temp = F.softplus(self.phys_temperature_y).clamp(min=1e-6)
                gam = torch.sigmoid(self.phys_gamma_entropy_y)
                gate = _axial_physics_gate(u_hat_i, u_hat_j, rel_y, b_ij, temp, gam)
                adj_y_feats.append(msg_in)
                adj_y_gates.append(gate)

            else:  # nonadj with dm != 0 (temporal / diagonal / long-range)
                # CFL over both axes: dt * (|a|/dx + |b|/dy).
                cfl = a_hat_i.abs() * rel_t.abs() * (1.0 / dx_val + 1.0 / dy_val)
                msg_in = torch.cat([
                    h, h_j,
                    du, rel_x, rel_y, rel_t,
                    torch.sign(rel_x), torch.sign(rel_y),
                ], dim=-1)                                              # 2d + 6
                cfl_scale = F.softplus(self.phys_cfl_scale).clamp(min=1e-6)
                gate = torch.exp(-cfl_scale * F.relu(cfl - 1.0) ** 2)
                nonadj_feats.append(msg_in)
                nonadj_gates.append(gate)

        all_gates = adj_x_gates + adj_y_gates + nonadj_gates
        gate_sum = torch.stack(all_gates, dim=-2).sum(dim=-2) + 1e-3   # [B,nt,ny,nx,1]

        def _batched(mlp: nn.Module, feats: list[torch.Tensor]) -> torch.Tensor:
            stacked = torch.stack(feats, dim=4)                        # [B,nt,ny,nx,n,F]
            n = stacked.shape[4]
            return mlp(
                stacked.reshape(-1, stacked.shape[-1])
            ).reshape(B, nt, ny, nx, n, d)

        msg_chunks: list[torch.Tensor] = []
        if adj_x_feats:
            msg_chunks.append(_batched(self.adj_x_msg, adj_x_feats))
        if adj_y_feats:
            msg_chunks.append(_batched(self.adj_y_msg, adj_y_feats))
        if nonadj_feats:
            msg_chunks.append(_batched(self.nonadj_msg, nonadj_feats))

        all_msgs = torch.cat(msg_chunks, dim=4)                        # [B,nt,ny,nx,n_all,d]
        all_gates_t = torch.stack(all_gates, dim=4)                    # [B,nt,ny,nx,n_all,1]
        agg = (all_gates_t / gate_sum.unsqueeze(4) * all_msgs).sum(dim=4)

        h_nonlocal = self.update_net(torch.cat([h, agg], dim=-1))
        h_local = self.W(h)
        return self.act(h_nonlocal + h_local)


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #
class HypNO_ST3_2D(nn.Module):
    """HypNO-ST v3, 2D scalar LWR variant.

    Lifting layer -> ``n_layers`` physics-gated MP layers -> shared decoder.
    The shared decoder is used both inside each MP layer (for the gate
    ``u_hat``) and as the deep-supervision readout for every captured latent.

    Flux-on-node and pure-pairwise edges are fixed (see module docstring).
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
        **_ignored,
    ) -> None:
        super().__init__()
        print("=" * 60)
        print("[HypNO_ST3_2D] constructing:")
        print(f"  stencil_k_x/y/t  = {stencil_k_x}/{stencil_k_y}/{stencil_k_t}")
        print(f"  d_latent         = {d_latent}")
        print(f"  d_hidden         = {d_hidden}")
        print(f"  d_hidden_nonadj  = {d_hidden_nonadj}")
        print(f"  n_layers         = {n_layers}")
        print(f"  encoder_type     = {encoder_type}")
        print(f"  readout          = {readout}")
        print(f"  skip             = {skip}")
        print(f"  use_checkpoint   = {use_checkpoint}")
        if _ignored:
            print(f"  IGNORED kwargs   = {sorted(_ignored.keys())}")
        print("=" * 60)
        self.skip = skip
        self.use_checkpoint = use_checkpoint

        self.lifting = _SpaceTimeLiftingLayer2D(
            d_latent, d_hidden,
            stencil_k_x=stencil_k_x, stencil_k_y=stencil_k_y, stencil_k_t=stencil_k_t,
            activation=activation, encoder_type=encoder_type,
            causal_temporal=causal_temporal,
        )

        # Build the decoder before the MP layers so it can be shared.
        self.decoder = _make_mlp(d_latent, d_hidden, 1, 3, readout)

        self.mp_layers = nn.ModuleList([
            _PhysicsSpaceTimeMPLayer2D(
                d_latent, d_hidden, stencil_k_x, stencil_k_y, stencil_k_t,
                activation=activation, causal_temporal=causal_temporal,
                d_hidden_nonadj=d_hidden_nonadj,
                shared_decoder=self.decoder,
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
        x  : [nx]   y : [ny]   t : [nt]

        Returns ``(u_pred, u_hats)`` where ``u_pred`` is ``[B, nt, ny, nx]``
        and ``u_hats`` is the per-MP-layer decoder readout list (deep
        supervision).
        """
        B, ny, nx = u0.shape
        nt = t.shape[0]

        h = self.lifting(u0, x, y, t)

        def _decode(h_in: torch.Tensor) -> torch.Tensor:
            d_out = self.decoder(h_in).squeeze(-1)                     # [B,nt,ny,nx]
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
