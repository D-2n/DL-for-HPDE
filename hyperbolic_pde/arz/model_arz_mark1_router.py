"""HypNO-ARZ Mark 1 (router-aware): the exact pre-split single-MLP model.

This is the ARZ operator *before* the Mark 2 two-family GNL/LD split. It keeps
the single shared `adj_msg` / `nonadj_msg` edge MLPs and the pure-pairwise edge
design of `HypNO_ARZ` (model_arz.py), with ONE deliberate change:

  * The adjacent-spatial edge message now carries the family-routing scalar
    `theta = |dw| / (|dw| + |dv| + eps)` -- the same quantity Mark 2 uses to
    blend its two family branches and that the gate already consumes here.
    Adjacent message dim: 2d+3 -> 2d+4. Non-adjacent edges are untouched (2d+3).

Everything else (lifting, gates, decoder, deep supervision, checkpoint loading)
is byte-for-byte the Mark 1 model. The lifting layer (`_ArzLifting`, reused from
model_arz.py) already carried theta in its adjacent edge vector, so no change is
needed there -- only the processor's adjacent message gained it.

Rationale: probe whether handing the *router* signal to the message (not just
the gate) recovers the family discrimination that the Mark 2 split was meant to
provide, without paying for two separate MLPs per family.

Decoder outputs (rho, w) -- 2 channels.
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

# Reuse the Mark 1 closures, lifting, and entropy flag verbatim.
from hyperbolic_pde.arz.model_arz import (
    _ArzLifting,
    _entropy_bad_1shock,
    _p,
    _dp,
)


# --------------------------------------------------------------------------- #
# Physics-gated MP layer (router-aware adjacent message)
# --------------------------------------------------------------------------- #
class _ArzMPLayerM1R(nn.Module):
    """Two-edge-MLP space-time MP for ARZ (pure-pairwise, router-aware).

    Adjacent edges (2d + 4):
        [h_i, h_j, lam1_ij, lam2_ij, sign(rel_x), theta]
    Non-adjacent edges (2d + 3):
        [h_i, h_j, rel_x_feat, rel_t_feat, sign(rel_x)]

    Identical to Mark 1's `_ArzMPLayer` except the adjacent message appends the
    family-routing scalar theta (which the gate already uses). theta in {0,1}
    softly tags the edge's dominant family: ~1 = LD / 2-contact (dw dominates),
    ~0 = GNL / 1-wave (dv dominates).
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
            raise ValueError(f"double_batch requires k_x even; got {k_x}.")
        self.k_x = k_x
        self.k_t = k_t
        self.causal = causal_temporal
        self.normalize_edge_offsets = normalize_edge_offsets
        self.double_batch = double_batch
        self.neighborhood_spacing = neighborhood_spacing
        self.act = nn.GELU() if activation == "gelu" else nn.Tanh()
        dh_na = d_hidden if d_hidden_nonadj is None else d_hidden_nonadj

        if shared_decoder is None:
            raise ValueError("_ArzMPLayerM1R requires a shared_decoder")
        object.__setattr__(self, "_shared_decoder", shared_decoder)

        self.phys_temp1 = nn.Parameter(torch.tensor(0.0))
        self.phys_temp2 = nn.Parameter(torch.tensor(0.0))
        self.phys_gamma = nn.Parameter(torch.tensor(-2.0))
        self.phys_cfl_scale = nn.Parameter(torch.tensor(0.0))

        # Router-aware adjacent message: 2d+4 (the +1 is theta).
        self.adj_msg = _make_mlp(2 * d_latent + 4, d_hidden, d_latent, 3, activation)
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

                # Router-aware pure-pairwise: interface eigenvalues + the family
                # router theta. theta is the same scalar the gate consumes; here
                # it also tags the message so the shared adj MLP can specialize
                # its output by family without a hard branch split.
                msg_in = torch.cat([
                    h, h_j,
                    lam1_ij, lam2_ij,
                    r, theta,
                ], dim=-1)  # 2d + 4
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
class HypNO_ARZ_Mark1Router(nn.Module):
    """HypNO operator for 1D ARZ -- Mark 1 with router-aware adjacent message.

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
            print(f"[HypNO_ARZ_Mark1Router] IGNORED kwargs = {sorted(_ignored.keys())}")
        print(
            f"[HypNO_ARZ_Mark1Router] kx={stencil_k_x} kt={stencil_k_t} "
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
            _ArzMPLayerM1R(
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
# Constructor kwargs + checkpoint loading
# --------------------------------------------------------------------------- #
def cfg_to_kwargs_m1r(model_cfg: dict) -> dict:
    """Translate a config block into HypNO_ARZ_Mark1Router constructor kwargs."""
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


def load_hypno_arz_mark1_router_from_checkpoint(
    ckpt_path,
    device: str = "cpu",
    config_path=None,
    model_section: Optional[str] = None,
):
    """Reconstruct a HypNO_ARZ_Mark1Router from a bare-state_dict checkpoint.

    Mirrors load_hypno_arz_from_checkpoint: the trainer writes the architecture
    into run_dir/config.yaml, auto-located alongside the checkpoint unless
    config_path= is given. model_section defaults to 'hypno_arz_mark1_router'.
    """
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

    if model_section is None:
        model_section = "hypno_arz_mark1_router"
    if model_section not in cfg:
        raise KeyError(
            f"Section {model_section!r} not found in {config_path}; available: "
            f"{sorted(k for k in cfg if isinstance(cfg.get(k), dict))}"
        )
    model_cfg = cfg.get(model_section, {})
    kwargs = cfg_to_kwargs_m1r(model_cfg)
    model = HypNO_ARZ_Mark1Router(**kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    tau = None
    for sec in ("arz_data", "arz_trial", "arz_riemann_trial"):
        if sec in cfg and isinstance(cfg[sec], dict) and "tau" in cfg[sec]:
            tau = float(cfg[sec]["tau"])
            break
    return model, tau
