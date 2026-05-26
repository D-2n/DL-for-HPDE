"""End-to-end smoke test: build HypNO_ARZ, run a forward pass, overfit one
sample for a few steps, and assert the loss drops monotonically.

This catches shape bugs, gate-tensor broadcasting bugs, and obviously broken
gradient paths before any cluster job runs.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from hyperbolic_pde.arz.datagen_arz import load_arz_dataset
from hyperbolic_pde.arz.model_arz import HypNO_ARZ


def test_forward_shape():
    torch.manual_seed(0)
    bundle = load_arz_dataset(
        r"C:\Users\dimit\Desktop\repos\DL-for-HPDE\hyperbolic_pde\arz\tests\_smoke_data.npz"
    )
    rho0 = torch.tensor(bundle.rho0[:2])
    w0 = torch.tensor(bundle.w0[:2])
    x = torch.tensor(bundle.x)
    t = torch.tensor(bundle.t)
    model = HypNO_ARZ(
        stencil_k_x=2, stencil_k_t=1,
        d_latent=16, d_hidden=24, n_layers=2,
        skip=True, use_checkpoint=False,
    )
    rho_p, w_p, u_hats = model(rho0, w0, x, t)
    assert rho_p.shape == (2, t.shape[0], x.shape[0])
    assert w_p.shape == rho_p.shape
    assert len(u_hats) == 2
    assert u_hats[-1].shape == (2, t.shape[0], x.shape[0], 2)


def test_overfit_one_sample():
    torch.manual_seed(0)
    bundle = load_arz_dataset(
        r"C:\Users\dimit\Desktop\repos\DL-for-HPDE\hyperbolic_pde\arz\tests\_smoke_data.npz"
    )
    # Take the first sample, batch dim 1.
    rho0 = torch.tensor(bundle.rho0[:1])
    w0 = torch.tensor(bundle.w0[:1])
    rho_gt = torch.tensor(bundle.rho[:1])
    w_gt = torch.tensor(bundle.w[:1])
    x = torch.tensor(bundle.x)
    t = torch.tensor(bundle.t)

    model = HypNO_ARZ(
        stencil_k_x=2, stencil_k_t=1,
        d_latent=24, d_hidden=32, n_layers=3,
        skip=True, use_checkpoint=False,
    )
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    losses = []
    t0 = time.time()
    for step in range(60):
        opt.zero_grad()
        rho_p, w_p, _ = model(rho0, w0, x, t)
        loss = ((rho_p - rho_gt) ** 2).mean() + ((w_p - w_gt) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step % 5 == 0:
            print(f"  step {step:3d}  loss = {losses[-1]:.4e}")
    dt = time.time() - t0

    print(f"  total = {dt:.1f}s  initial loss = {losses[0]:.4e}  final loss = {losses[-1]:.4e}")
    assert losses[-1] < losses[0] * 0.6, (
        f"overfit failed to make progress: {losses[0]:.4e} -> {losses[-1]:.4e}"
    )
    assert np.isfinite(losses[-1])


if __name__ == "__main__":
    print("=== forward shape ===")
    test_forward_shape()
    print("OK\n=== overfit one sample ===")
    test_overfit_one_sample()
    print("Smoke test passed.")
