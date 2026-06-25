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


# Closures keyed off the global pressure_form switch (see physics_arz).
from hyperbolic_pde.arz import physics_arz as _P_MOD


def _p(rho):
    if _P_MOD._P_FORM == "rho":
        return rho
    return rho + rho * rho


def _dist(diff: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Elementwise distance, mirroring train_hypno_st3's `_dist`.

    'mae' -> |diff|, 'mse' -> diff^2, 'huber' -> smooth-L1 (delta=0.1).
    The caller takes the mean (so this returns a per-element tensor).
    """
    lt = loss_type.lower()
    if lt == "mae":
        return diff.abs()
    if lt == "mse":
        return diff.pow(2)
    if lt == "huber":
        # Huber/smooth-L1 around delta: quadratic for |diff|<delta, linear past.
        delta = 0.1
        a = diff.abs()
        return torch.where(a < delta, 0.5 * diff.pow(2) / delta, a - 0.5 * delta)
    raise ValueError(f"loss_type must be 'mae', 'mse', or 'huber', got {loss_type!r}")


def _y_eq(rho):
    if _P_MOD._P_FORM == "rho":
        return rho                  # rho * w_eq, w_eq = 1
    return rho + rho * rho * rho


def state_loss(
    rho_pred: torch.Tensor, w_pred: torch.Tensor,
    rho_gt: torch.Tensor,   w_gt:   torch.Tensor,
    w_weight: float = 1.0,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Distance on (rho, w). w_weight reweights the w channel; loss_type selects
    the elementwise distance (mae/mse/huber)."""
    l_rho = _dist(rho_pred - rho_gt, loss_type).mean()
    l_w   = _dist(w_pred   - w_gt,   loss_type).mean()
    return l_rho + w_weight * l_w


def rho_mass_loss(
    rho_pred: torch.Tensor, rho_gt: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Penalise drift in spatially-integrated rho (mass) per timestep.

    Compares mean-over-x of rho at each t (so the loss is O(1) and resolution-
    invariant). The predicted mean must track the ground-truth mean.
    """
    mass_pred = rho_pred.mean(dim=-1)        # [B, nt]
    mass_gt   = rho_gt.mean(dim=-1)
    return _dist(mass_pred - mass_gt, loss_type).mean()


def balance_residual_loss(
    rho_pred: torch.Tensor, w_pred: torch.Tensor, t: torch.Tensor,
    tau: float, loss_type: str = "mse",
) -> torch.Tensor:
    """Discrete check of d/dt integral(y) ~ integral((y_eq - y)/tau).

    Both sides are O(1/tau). To keep this term O(1) regardless of tau (so a
    fixed lambda_balance is meaningful), we compare tau*dI/dt to tau*int(S):
    that's a rescaling of the same identity by a constant factor tau. Both
    sides are then O(1), and additionally normalised per-cell by dividing by
    nx (so the loss is comparable across grid resolutions).
    """
    B, nt, nx = rho_pred.shape
    y = rho_pred * w_pred                       # [B, nt, nx]
    I = y.sum(dim=-1) / nx                       # [B, nt]   — mean over x
    dt = (t[1:] - t[:-1]).clamp(min=1e-8)        # [nt-1]
    dIdt = (I[:, 1:] - I[:, :-1]) / dt           # [B, nt-1]
    # Source mean over x.
    src_mean = (_y_eq(rho_pred) - y).mean(dim=-1)   # [B, nt], note: no /tau
    # Multiply LHS by tau to cancel the 1/tau scale on the RHS.
    src_avg = src_mean[:, :-1] * 0.5 + src_mean[:, 1:] * 0.5
    return _dist(tau * dIdt - src_avg, loss_type).mean()


def probe_loss(
    u_hats: List[torch.Tensor],
    rho_gt: torch.Tensor, w_gt: torch.Tensor,
    w_weight: float = 1.0,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Mean (rho, w) distance across all per-layer decoder readouts. w_weight
    reweights w; loss_type selects the elementwise distance (mae/mse/huber)."""
    losses = []
    for uh in u_hats:
        # uh has shape [B, nt, nx, 2] = (rho, w).
        l_rho = _dist(uh[..., 0] - rho_gt, loss_type).mean()
        l_w   = _dist(uh[..., 1] - w_gt,   loss_type).mean()
        losses.append(l_rho + w_weight * l_w)
    if not losses:
        return torch.tensor(0.0, device=rho_gt.device)
    return torch.stack(losses).mean()


def state_loss_rv(
    rho_pred: torch.Tensor, v_pred: torch.Tensor,
    rho_gt: torch.Tensor,   v_gt:   torch.Tensor,
    v_weight: float = 1.0,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Distance on the PRIMITIVE pair (rho, v) -- Mark2 frame. v_weight reweights
    v; loss_type selects mae/mse/huber elementwise."""
    l_rho = _dist(rho_pred - rho_gt, loss_type).mean()
    l_v   = _dist(v_pred   - v_gt,   loss_type).mean()
    return l_rho + v_weight * l_v


def probe_loss_rv(
    u_hats_rv: List[torch.Tensor],
    rho_gt: torch.Tensor, v_gt: torch.Tensor,
    v_weight: float = 1.0,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Mean (rho, v) distance across per-layer readouts (each [B,nt,nx,2]=(rho,v))."""
    losses = []
    for uh in u_hats_rv:
        l_rho = _dist(uh[..., 0] - rho_gt, loss_type).mean()
        l_v   = _dist(uh[..., 1] - v_gt,   loss_type).mean()
        losses.append(l_rho + v_weight * l_v)
    if not losses:
        return torch.tensor(0.0, device=rho_gt.device)
    return torch.stack(losses).mean()


def mark2_total_loss(
    rho_pred: torch.Tensor, v_pred: torch.Tensor,
    u_hats_rv: List[torch.Tensor],
    rho_gt: torch.Tensor, v_gt: torch.Tensor,
    *,
    lambda_state: float = 1.0,
    lambda_conservation: float = 1.0,
    lambda_probe: float = 0.01,
    v_weight: float = 1.0,
    loss_type: str = "mse",
    return_info: bool = True,
) -> tuple:
    """HypNO-ARZ Mark2 objective (homogeneous): (rho,v)-frame state + probe,
    plus rho mass conservation. No relaxation-balance term (tau=inf).
    loss_type selects the elementwise distance (mae/mse/huber) for all terms.

    Returns (scalar_loss, info_dict). v_gt is computed by the caller from
    w_gt via v = w - p(rho_gt). When return_info=False the per-term floats
    (and their CUDA syncs) are skipped and info is None -- use on non-logging
    training steps where only the scalar loss is needed for backward().
    """
    L_state = state_loss_rv(rho_pred, v_pred, rho_gt, v_gt, v_weight=v_weight, loss_type=loss_type)
    L_cons  = rho_mass_loss(rho_pred, rho_gt, loss_type=loss_type)
    L_probe = probe_loss_rv(u_hats_rv, rho_gt, v_gt, v_weight=v_weight, loss_type=loss_type)
    L = lambda_state * L_state + lambda_conservation * L_cons + lambda_probe * L_probe
    if not return_info:
        return L, None
    info = {
        "state": float(L_state.detach().item()),
        "cons":  float(L_cons.detach().item()),
        "bal":   0.0,
        "probe": float(L_probe.detach().item()),
        "total": float(L.detach().item()),
    }
    return L, info


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
    loss_type: str = "mse",
) -> dict:
    """Compose the four components; return both the scalar and a per-term dict.
    loss_type selects the elementwise distance (mae/mse/huber) for all terms."""
    L_state = state_loss(rho_pred, w_pred, rho_gt, w_gt, w_weight=w_weight, loss_type=loss_type)
    L_cons  = rho_mass_loss(rho_pred, rho_gt, loss_type=loss_type)
    L_bal   = balance_residual_loss(rho_pred, w_pred, t, tau, loss_type=loss_type)
    L_probe = probe_loss(u_hats, rho_gt, w_gt, w_weight=w_weight, loss_type=loss_type)
    L = lam_state * L_state + lam_cons * L_cons + lam_bal * L_bal + lam_probe * L_probe
    return {
        "loss": L,
        "L_state": L_state.detach(),
        "L_cons":  L_cons.detach(),
        "L_bal":   L_bal.detach(),
        "L_probe": L_probe.detach(),
    }
