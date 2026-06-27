"""Equivalence guards for the HypNO_ARZ_Orig performance flags.

The frozen baseline (`model_arz_orig`) gained two opt-in speed flags:

  * ``fast_static_cache`` -- caches the grid-only edge geometry and evaluates
    the lifting's non-adjacent edge MLP at B=1, then broadcasts. Must produce
    the SAME output + gradients as the default path (it only reorders/reuses
    constant work).
  * ``checkpoint_stride`` -- checkpoints only every k-th MP layer. Gradient
    checkpointing is exact by construction, so any stride must match the
    no-checkpoint result.

These tests build two models that share weights and assert their forward
outputs and parameter gradients agree to fp64 tolerance. They run on CPU.
"""
from __future__ import annotations

import copy

import torch

try:
    import pytest
except ImportError:  # allow `python -m ...` without pytest installed
    pytest = None

from hyperbolic_pde.arz import physics_arz as _P_MOD
from hyperbolic_pde.arz.model_arz_orig import HypNO_ARZ_Orig


if pytest is not None:
    @pytest.fixture(autouse=True)
    def _prho():
        """Run under the project-default p(rho)=rho closure; restore afterward.

        Equivalence holds for any closure, but pinning it keeps the test aligned
        with the project convention and isolated from other tests' global state.
        """
        prev = _P_MOD.get_pressure_form()
        _P_MOD.set_pressure_form("rho")
        yield
        _P_MOD.set_pressure_form(prev)


_BASE_KWARGS = dict(
    stencil_k_x=3,
    stencil_k_t=2,
    d_latent=8,
    d_hidden=12,
    n_layers=3,
    activation="gelu",
    causal_temporal=True,
    decoder_depth=2,
    skip=False,
    use_relaxation_features=False,
    normalize_edge_offsets=True,
    new_entropy=True,
)


def _make_inputs(B=2, nx=11, nt=6, dtype=torch.float64, seed=0):
    """Physically valid ARZ inputs (rho>0) on a unit grid, p(rho)=rho."""
    g = torch.Generator().manual_seed(seed)
    rho0 = (0.2 + 0.6 * torch.rand(B, nx, generator=g)).to(dtype)
    v0 = (0.1 + 0.7 * torch.rand(B, nx, generator=g)).to(dtype)
    w0 = v0 + rho0  # p(rho)=rho -> w = v + p
    x = torch.linspace(-1.0, 1.0, nx, dtype=dtype)
    t = torch.linspace(0.0, 1.0, nt, dtype=dtype)
    return rho0, w0, x, t


def _build_pair(flag_overrides_a, flag_overrides_b, dtype=torch.float64):
    """Two models with IDENTICAL weights, differing only in the perf flags."""
    torch.manual_seed(1234)
    model_a = HypNO_ARZ_Orig(**_BASE_KWARGS, **flag_overrides_a).to(dtype)
    model_b = HypNO_ARZ_Orig(**_BASE_KWARGS, **flag_overrides_b).to(dtype)
    model_b.load_state_dict(copy.deepcopy(model_a.state_dict()))
    return model_a, model_b


def _forward_and_grads(model, rho0, w0, x, t):
    model.zero_grad(set_to_none=True)
    rho_p, w_p, u_hats = model(rho0, w0, x, t)
    # A scalar that touches every output + every deep-supervision head.
    loss = rho_p.pow(2).mean() + w_p.pow(2).mean()
    loss = loss + sum(u.pow(2).mean() for u in u_hats)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    return rho_p.detach(), w_p.detach(), grads


def _assert_close(out_a, out_b, atol, rtol, label):
    rho_a, w_a, g_a = out_a
    rho_b, w_b, g_b = out_b
    rho_err = (rho_a - rho_b).abs().max().item()
    w_err = (w_a - w_b).abs().max().item()
    assert rho_err <= atol, f"[{label}] rho max-abs err {rho_err:.3e} > {atol:.1e}"
    assert w_err <= atol, f"[{label}] w max-abs err {w_err:.3e} > {atol:.1e}"
    assert g_a.keys() == g_b.keys(), f"[{label}] grad key mismatch"
    worst = 0.0
    worst_name = ""
    for n in g_a:
        e = (g_a[n] - g_b[n]).abs().max().item()
        if e > worst:
            worst, worst_name = e, n
    assert worst <= atol, (
        f"[{label}] grad max-abs err {worst:.3e} ({worst_name}) > {atol:.1e}"
    )
    return rho_err, w_err, worst


def test_fast_static_cache_matches_default_fp64():
    """fast_static_cache=True must equal the default path (output + grads)."""
    rho0, w0, x, t = _make_inputs(dtype=torch.float64)
    slow, fast = _build_pair(
        dict(use_checkpoint=False, fast_static_cache=False),
        dict(use_checkpoint=False, fast_static_cache=True),
    )
    out_slow = _forward_and_grads(slow, rho0, w0, x, t)
    out_fast = _forward_and_grads(fast, rho0, w0, x, t)
    rho_err, w_err, g_err = _assert_close(out_slow, out_fast, atol=1e-10, rtol=1e-7,
                                          label="fast_static_cache")
    print(f"fast_static_cache fp64: rho={rho_err:.2e} w={w_err:.2e} grad={g_err:.2e}")


def test_message_factor_matches_default_fp64():
    """fast_message_factor=True must equal the default path (output + grads).

    This factors the non-adjacent message MLP's first Linear into block matmuls,
    a reassociation of the same sum over 2d+4 terms -> fp64-tight (not bitwise).
    """
    rho0, w0, x, t = _make_inputs(dtype=torch.float64)
    slow, fast = _build_pair(
        dict(use_checkpoint=False, fast_message_factor=False),
        dict(use_checkpoint=False, fast_message_factor=True),
    )
    out_slow = _forward_and_grads(slow, rho0, w0, x, t)
    out_fast = _forward_and_grads(fast, rho0, w0, x, t)
    rho_err, w_err, g_err = _assert_close(out_slow, out_fast, atol=1e-9, rtol=1e-7,
                                          label="fast_message_factor")
    print(f"fast_message_factor fp64: rho={rho_err:.2e} w={w_err:.2e} grad={g_err:.2e}")


def test_message_factor_composes_with_static_cache_fp64():
    """fast_message_factor + fast_static_cache together == default."""
    rho0, w0, x, t = _make_inputs(dtype=torch.float64)
    slow, fast = _build_pair(
        dict(use_checkpoint=False, fast_static_cache=False, fast_message_factor=False),
        dict(use_checkpoint=False, fast_static_cache=True, fast_message_factor=True),
    )
    out_slow = _forward_and_grads(slow, rho0, w0, x, t)
    out_fast = _forward_and_grads(fast, rho0, w0, x, t)
    rho_err, w_err, g_err = _assert_close(out_slow, out_fast, atol=1e-9, rtol=1e-7,
                                          label="factor+cache")
    print(f"factor+cache fp64: rho={rho_err:.2e} w={w_err:.2e} grad={g_err:.2e}")


def test_checkpoint_stride_matches_no_checkpoint_fp64():
    """Any checkpoint_stride must equal the no-checkpoint result (exact)."""
    rho0, w0, x, t = _make_inputs(dtype=torch.float64)
    ref_model, _ = _build_pair(
        dict(use_checkpoint=False), dict(use_checkpoint=False))
    ref = _forward_and_grads(ref_model, rho0, w0, x, t)
    for stride in (1, 2, 3, 99):
        torch.manual_seed(1234)
        ck = HypNO_ARZ_Orig(
            **_BASE_KWARGS, use_checkpoint=True, checkpoint_stride=stride
        ).to(torch.float64)
        ck.load_state_dict(copy.deepcopy(ref_model.state_dict()))
        out = _forward_and_grads(ck, rho0, w0, x, t)
        rho_err, w_err, g_err = _assert_close(ref, out, atol=1e-10, rtol=1e-7,
                                              label=f"ckpt_stride={stride}")
        print(f"ckpt_stride={stride} fp64: rho={rho_err:.2e} w={w_err:.2e} grad={g_err:.2e}")


def test_all_flags_together_fp32():
    """All three flags at once in fp32 (the training dtype family) stay close."""
    rho0, w0, x, t = _make_inputs(dtype=torch.float32)
    slow, fast = _build_pair(
        dict(use_checkpoint=False, fast_static_cache=False, fast_message_factor=False),
        dict(use_checkpoint=True, checkpoint_stride=2,
             fast_static_cache=True, fast_message_factor=True),
        dtype=torch.float32,
    )
    out_slow = _forward_and_grads(slow, rho0, w0, x, t)
    out_fast = _forward_and_grads(fast, rho0, w0, x, t)
    rho_err, w_err, g_err = _assert_close(out_slow, out_fast, atol=2e-5, rtol=1e-4,
                                          label="all_flags_fp32")
    print(f"all flags fp32: rho={rho_err:.2e} w={w_err:.2e} grad={g_err:.2e}")


if __name__ == "__main__":
    _P_MOD.set_pressure_form("rho")
    print("=== fast_static_cache (fp64) ===")
    test_fast_static_cache_matches_default_fp64()
    print("=== fast_message_factor (fp64) ===")
    test_message_factor_matches_default_fp64()
    print("=== factor + cache (fp64) ===")
    test_message_factor_composes_with_static_cache_fp64()
    print("=== checkpoint_stride (fp64) ===")
    test_checkpoint_stride_matches_no_checkpoint_fp64()
    print("=== all flags (fp32) ===")
    test_all_flags_together_fp32()
    print("All equivalence checks passed.")
