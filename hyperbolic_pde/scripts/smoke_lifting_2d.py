"""Smoke test for the 2D lifting layer (_SpaceTimeLiftingLayer2D).

Builds a dummy [B, ny, nx] input, runs it through the lifting layer in
both encoder_type modes (gnn / mlp), and checks:
  - output shape is [B, nt, ny, nx, d_latent]
  - no NaN / Inf in the output
  - a backward pass produces finite gradients

Run from repo root:
    python hyperbolic_pde/scripts/smoke_lifting_2d.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hyperbolic_pde.models.hypno_st3_2d import (
    _SpaceTimeLiftingLayer2D,
    _enumerate_ball_offsets_2d,
)


def _count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def run_case(encoder_type: str) -> None:
    B, ny, nx, nt = 2, 12, 12, 6
    k_x = k_y = k_t = 2
    d_latent, d_hidden = 32, 48

    torch.manual_seed(0)
    u0 = torch.rand(B, ny, nx)
    x = torch.linspace(-1.0, 1.0, nx)
    y = torch.linspace(-1.0, 1.0, ny)
    t = torch.linspace(0.0, 1.0, nt)
    u0.requires_grad_(True)

    layer = _SpaceTimeLiftingLayer2D(
        d_latent=d_latent, d_hidden=d_hidden,
        stencil_k_x=k_x, stencil_k_y=k_y, stencil_k_t=k_t,
        encoder_type=encoder_type,
    )

    h = layer(u0, x, y, t)
    expected = (B, nt, ny, nx, d_latent)
    ok_shape = tuple(h.shape) == expected
    ok_finite = torch.isfinite(h).all().item()

    loss = h.pow(2).mean()
    loss.backward()
    grad_finite = u0.grad is not None and torch.isfinite(u0.grad).all().item()

    n_off = len(_enumerate_ball_offsets_2d(k_x, k_y, k_t, causal=True))
    print(
        f"  [type={encoder_type:4s}] out={tuple(h.shape)} "
        f"shape_ok={ok_shape} finite={ok_finite} grad_ok={grad_finite} "
        f"params={_count_params(layer):,} n_offsets={n_off}"
    )
    assert ok_shape, f"shape mismatch: got {tuple(h.shape)}, want {expected}"
    assert ok_finite, "non-finite values in lifting output"
    assert grad_finite, "non-finite gradient through lifting layer"


def main() -> None:
    print("=== 2D lifting layer smoke test ===")
    run_case("gnn")
    run_case("mlp")
    print("ALL CASES PASSED")


if __name__ == "__main__":
    main()
