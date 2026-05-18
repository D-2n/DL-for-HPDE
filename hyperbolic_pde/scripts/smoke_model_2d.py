"""Smoke test for the full 2D model (HypNO_ST3_2D).

Runs a forward + backward pass on a tiny dummy batch and checks:
  - u_pred shape is [B, nt, ny, nx]
  - u_hats is a list of length n_layers, each [B, nt, ny, nx]
  - all outputs finite
  - backward produces finite gradients on the model parameters
Tested with skip on/off.

Run from repo root:
    python hyperbolic_pde/scripts/smoke_model_2d.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hyperbolic_pde.models.hypno_st3_2d import HypNO_ST3_2D


def _count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def run_case(skip: bool) -> None:
    B, ny, nx, nt = 2, 12, 12, 6
    n_layers = 3
    d_latent, d_hidden = 32, 48

    torch.manual_seed(0)
    u0 = torch.rand(B, ny, nx)
    x = torch.linspace(-1.0, 1.0, nx)
    y = torch.linspace(-1.0, 1.0, ny)
    t = torch.linspace(0.0, 1.0, nt)

    model = HypNO_ST3_2D(
        stencil_k_x=2, stencil_k_y=2, stencil_k_t=2,
        d_latent=d_latent, d_hidden=d_hidden, n_layers=n_layers,
        encoder_type="gnn", skip=skip, use_checkpoint=True,
    )
    model.train()

    u_pred, u_hats = model(u0, x, y, t)

    expected = (B, nt, ny, nx)
    ok_pred = tuple(u_pred.shape) == expected
    ok_hats_len = len(u_hats) == n_layers
    ok_hats_shape = all(tuple(h.shape) == expected for h in u_hats)
    ok_finite = torch.isfinite(u_pred).all().item() and all(
        torch.isfinite(h).all().item() for h in u_hats
    )

    target = torch.rand_like(u_pred)
    loss = (u_pred - target).abs().mean()
    for h in u_hats:
        loss = loss + 0.05 * (h - target).abs().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    grad_finite = all(g is not None and torch.isfinite(g).all().item() for g in grads)

    print(
        f"  [skip={skip!s:5s}] u_pred={tuple(u_pred.shape)} "
        f"pred_ok={ok_pred} hats_len_ok={ok_hats_len} hats_shape_ok={ok_hats_shape} "
        f"finite={ok_finite} grad_ok={grad_finite} params={_count_params(model):,}"
    )
    assert ok_pred, f"u_pred shape mismatch: {tuple(u_pred.shape)} != {expected}"
    assert ok_hats_len, f"u_hats length {len(u_hats)} != {n_layers}"
    assert ok_hats_shape, "u_hats element shape mismatch"
    assert ok_finite, "non-finite values in model output"
    assert grad_finite, "non-finite gradient"


def main() -> None:
    print("=== 2D full-model smoke test ===")
    run_case(skip=True)
    run_case(skip=False)
    print("ALL CASES PASSED")


if __name__ == "__main__":
    main()
