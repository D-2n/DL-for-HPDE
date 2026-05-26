"""ARZ training losses (plan §6).

L_state    : MSE on (rho, w); optional channel weighting.
L_cons     : rho-only mass conservation (integral of rho is invariant).
             y is NOT conserved under relaxation -- do not penalise its sum.
L_balance  : discrete d/dt integral(y) ~ integral((y_eq - y)/tau).
L_probe    : deep supervision over (rho, w) decoded at each MP layer.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def _p(rho): return rho + rho * rho
def _y_eq(rho): return rho + rho * rho * rho


def state_loss(
    rho_pred: torch.Tensor, w_pred: torch.Tensor,
    rho_gt: torch.Tensor,   w_gt:   torch.Tensor,
    w_weight: float = 1.0,
) -> torch.Tensor:
    """MSE on (rho, w). w_weight reweights the w channel."""
    l_rho = F.mse_loss(rho_pred, rho_gt)
    l_w   = F.mse_loss(w_pred,   w_gt)
    return l_rho + w_weight * l_w


def rho_mass_loss(
    rho_pred: torch.Tensor, rho_gt: torch.Tensor,
) -> torch.Tensor:
    """Penalise drift in spatially-integrated rho (mass) per timestep.

    Inputs shaped [B, nt, nx]. We compare integral over x at each t -- and
    require the predicted integral to track the ground-truth integral. (For
    periodic BCs the GT integral is constant; for ghost BCs it can change,
    so tracking GT is the correct framing rather than constancy from t=0.)
    """
    mass_pred = rho_pred.sum(dim=-1)        # [B, nt]
    mass_gt   = rho_gt.sum(dim=-1)
    return F.mse_loss(mass_pred, mass_gt)


def balance_residual_loss(
    rho_pred: torch.Tensor, w_pred: torch.Tensor, t: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Discrete check of d/dt integral(y) ~ integral((y_eq - y)/tau).

    Uses centred differences in time on integral(y) over x. Inputs [B, nt, nx].
    """
    y = rho_pred * w_pred                     # [B, nt, nx]
    I = y.sum(dim=-1)                          # [B, nt]
    # centred difference; one-sided at the ends
    dt = (t[1:] - t[:-1]).clamp(min=1e-8)      # [nt-1]
    dIdt = (I[:, 1:] - I[:, :-1]) / dt         # [B, nt-1]
    # Source integral evaluated at the midpoint of each interval (average of
    # the two endpoints) to match the centred difference.
    src = (_y_eq(rho_pred) - y) / tau           # [B, nt, nx]
    src_int = src.sum(dim=-1)                  # [B, nt]
    src_mid = 0.5 * (src_int[:, 1:] + src_int[:, :-1])
    return F.mse_loss(dIdt, src_mid)


def probe_loss(
    u_hats: List[torch.Tensor],
    rho_gt: torch.Tensor, w_gt: torch.Tensor,
    w_weight: float = 1.0,
) -> torch.Tensor:
    """Mean MSE across all per-layer decoder readouts."""
    target = torch.stack([rho_gt, w_weight * w_gt], dim=-1)
    target_rho = rho_gt
    target_w   = w_gt * w_weight
    losses = []
    for uh in u_hats:
        # uh has shape [B, nt, nx, 2]
        l_rho = F.mse_loss(uh[..., 0], target_rho)
        l_w   = F.mse_loss(uh[..., 1], target_w / max(w_weight, 1e-8) * w_weight)
        losses.append(l_rho + w_weight * l_w)
    if not losses:
        return torch.tensor(0.0, device=rho_gt.device)
    return torch.stack(losses).mean()


def total_loss(
    rho_pred: torch.Tensor, w_pred: torch.Tensor,
    u_hats: List[torch.Tensor],
    rho_gt: torch.Tensor, w_gt: torch.Tensor,
    t: torch.Tensor, tau: float,
    lam_state: float = 1.0,
    lam_cons:  float = 0.1,
    lam_bal:   float = 0.1,
    lam_probe: float = 0.1,
    w_weight:  float = 1.0,
) -> dict:
    """Compose the four components; return both the scalar and a per-term dict."""
    L_state = state_loss(rho_pred, w_pred, rho_gt, w_gt, w_weight=w_weight)
    L_cons  = rho_mass_loss(rho_pred, rho_gt)
    L_bal   = balance_residual_loss(rho_pred, w_pred, t, tau)
    L_probe = probe_loss(u_hats, rho_gt, w_gt, w_weight=w_weight)
    L = lam_state * L_state + lam_cons * L_cons + lam_bal * L_bal + lam_probe * L_probe
    return {
        "loss": L,
        "L_state": L_state.detach(),
        "L_cons":  L_cons.detach(),
        "L_bal":   L_bal.detach(),
        "L_probe": L_probe.detach(),
    }
