"""Sanity tests for ARZ closures (plan §2)."""
from __future__ import annotations

import numpy as np

from hyperbolic_pde.arz import physics_arz as P


def test_lwr_limit_numpy():
    """F1(rho, w_eq(rho)) must equal rho*(1-rho) — the LWR flux — to 1e-6."""
    rho = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    w = P.w_eq(rho)
    F1, _F2 = P.flux_from_rw(rho, w)
    F1_lwr = rho * (1.0 - rho)
    err = np.max(np.abs(F1 - F1_lwr))
    assert err < 1e-12, f"LWR-limit flux mismatch: max |dF1| = {err:.3e}"


def test_to_primitive_roundtrip():
    rho = np.linspace(0.01, 1.0, 64)
    w = np.linspace(0.5, 2.0, 64)
    _, y = P.to_conserved(rho, w)
    rho2, w2, v2 = P.to_primitive(rho, y)
    assert np.allclose(w2, w, atol=1e-10)
    assert np.allclose(v2, w - P.pressure(rho), atol=1e-10)


def test_eigenvalues_ordering():
    """lambda1 <= lambda2 always (= when rho*p'(rho) = 0, i.e. rho=0)."""
    rho = np.linspace(0.0, 1.0, 65)
    w = np.linspace(0.5, 2.0, 65)
    lam1, lam2 = P.eigenvalues(rho, w)
    assert np.all(lam1 <= lam2 + 1e-12)


def test_relaxation_zero_at_equilibrium():
    rho = np.linspace(0.0, 1.0, 33)
    w = P.w_eq(rho)
    _, y = P.to_conserved(rho, w)
    _, S_y = P.relaxation_source(rho, y, tau=0.1)
    assert np.allclose(S_y, 0.0, atol=1e-12)


def test_torch_matches_numpy():
    import torch
    rho_np = np.linspace(0.0, 1.0, 33)
    w_np = np.linspace(0.5, 2.0, 33)
    rho_t = torch.tensor(rho_np)
    w_t = torch.tensor(w_np)
    F1_np, F2_np = P.flux_from_rw(rho_np, w_np)
    F1_t, F2_t = P.flux_from_rw(rho_t, w_t)
    assert np.allclose(F1_t.numpy(), F1_np, atol=1e-12)
    assert np.allclose(F2_t.numpy(), F2_np, atol=1e-12)


if __name__ == "__main__":
    test_lwr_limit_numpy()
    test_to_primitive_roundtrip()
    test_eigenvalues_ordering()
    test_relaxation_zero_at_equilibrium()
    test_torch_matches_numpy()
    print("All physics_arz tests passed.")
