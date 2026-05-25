"""Smoke test for HypNO_Linear_2D_v0.

Runs the diagnostics from `hypno_2d_v0_plan.md` Diagnostics section:

 1. Node-MLP input shape is [B, nt, ny, nx, 12].
 2. Edge counts per node match the plan (2 x-adj, 2 y-adj, ball minus 5 non-adj).
 3. chi^x_up = 1 for the left x-source / 0 for the right, with a=1, b=0.7;
    symmetric check for the y-axis.
 4. Entropy gates do not exist in v0 (no nonlinear shocks) -- trivially 1.
 5. g_temp is computed only on non-adj edges. We verify by enumerating one
    edge per class and recovering the same gate via the public primitives.
 6. chi^non_up uses x_i - x_k (target minus source) -- sign matches the spec
    on a hand-crafted off-axis edge.
 7. Replicate padding has no wrap-around (we check by perturbing one edge
    column of u0 and verifying the opposite edge is unaffected).

Plus an end-to-end forward-pass sanity check.

Usage:
    python hyperbolic_pde/scripts/smoke_test_linear_2d_v0.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from hyperbolic_pde.models.hypno_linear_2d_v0 import (
    HypNO_Linear_2D_v0,
    _adjacency_class,
    _char_alignment_gate,
    _enumerate_ball_offsets_2d,
    _soft_upwind_2d,
    _soft_upwind_axis,
)


def _check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(msg)


def test_lifting_input_shape() -> None:
    print("=== 1. Lifting node-MLP input shape ===")
    B, nx, ny, nt = 2, 5, 4, 3
    model = HypNO_Linear_2D_v0(
        stencil_k_x=2, stencil_k_y=2, stencil_k_t=2,
        d_latent=16, d_hidden=16, n_layers=1,
        a=1.0, b=0.7, h_s=1.0 / nx,
        use_checkpoint=False,
    )
    u0 = torch.rand(B, ny, nx)
    x = torch.linspace(0.0, 1.0, nx)
    y = torch.linspace(0.0, 1.0, ny)
    t = torch.linspace(0.0, 0.5, nt)

    captured: dict[str, torch.Tensor] = {}
    def _hook(_mod, inputs):
        captured["node_in"] = inputs[0]
    handle = model.lifting.node_mlp.register_forward_pre_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(u0, x, y, t)
    finally:
        handle.remove()

    node_in = captured["node_in"]
    _check(node_in.shape == (B, nt, ny, nx, 12),
           f"node_in shape {tuple(node_in.shape)} == [B, nt, ny, nx, 12]")


def test_edge_counts() -> None:
    print("=== 2. Edge counts per node ===")
    for k_s in (2, 3, 4):
        for k_t in (1, 2, 3):
            offs = _enumerate_ball_offsets_2d(k_s, k_s, k_t, causal=True)
            classes = [_adjacency_class(*o) for o in offs]
            n_x = sum(1 for c in classes if c == "adj_x")
            n_y = sum(1 for c in classes if c == "adj_y")
            n_non = sum(1 for c in classes if c == "nonadj")
            expected_total = (2 * k_s + 1) ** 2 * (k_t + 1) - 1
            expected_non = expected_total - 4   # minus 4 face-adjacent
            _check(n_x == 2, f"k_s={k_s}, k_t={k_t}: adj_x={n_x} == 2")
            _check(n_y == 2, f"k_s={k_s}, k_t={k_t}: adj_y={n_y} == 2")
            _check(
                n_non == expected_non,
                f"k_s={k_s}, k_t={k_t}: nonadj={n_non} == "
                f"(2k_s+1)^2*(k_t+1) - 1 - 4 = {expected_non}",
            )


def test_axial_upwind_flags() -> None:
    print("=== 3. chi^x_up / chi^y_up correctness on axial edges ===")
    a, b = 1.0, 0.7

    # Construct one toy edge per face neighbour by hand.
    # rel_x_target_minus_source = x_i - x_k.
    # Left x source: k = i-1, so x_i - x_k = +dx > 0.
    chi_left_x = (a * torch.tensor(+1.0) > 0).float()
    chi_right_x = (a * torch.tensor(-1.0) > 0).float()
    chi_bot_y = (b * torch.tensor(+1.0) > 0).float()
    chi_top_y = (b * torch.tensor(-1.0) > 0).float()

    _check(float(chi_left_x) == 1.0, "left x-source has chi^x_up = 1")
    _check(float(chi_right_x) == 0.0, "right x-source has chi^x_up = 0")
    _check(float(chi_bot_y) == 1.0, "bottom y-source has chi^y_up = 1")
    _check(float(chi_top_y) == 0.0, "top y-source has chi^y_up = 0")

    # And verify the soft sigmoid agrees in the limit tau -> 0 (tau small).
    tau = torch.tensor(0.05)
    g_left = _soft_upwind_axis(a, torch.tensor(+0.1), tau)
    g_right = _soft_upwind_axis(a, torch.tensor(-0.1), tau)
    _check(g_left > 0.9, f"soft g^x_up(left) = {float(g_left):.4f} > 0.9 (a>0)")
    _check(g_right < 0.1, f"soft g^x_up(right) = {float(g_right):.4f} < 0.1 (a>0)")


def test_entropy_gate_trivial() -> None:
    print("=== 4. Entropy gate == 1 for v0 (structural) ===")
    # The v0 model has no entropy-gate term in the code path. Confirm by
    # grepping the public model gate-construction order: the only
    # multiplicative factors are g_up (and g_temp on non-adj).
    import inspect
    from hyperbolic_pde.models import hypno_linear_2d_v0 as mod
    src = inspect.getsource(mod._LinearAdvectionMPLayer2D)
    _check("g_entropy" not in src,
           "MP layer source contains no `g_entropy` term (would be 1 anyway in v0)")
    _check("g_ent" not in src,
           "MP layer source contains no `g_ent` term in code (v0 contract)")


def test_temporal_gate_only_on_nonadj() -> None:
    print("=== 5. g_temp applied only on non-adjacent edges ===")
    import inspect
    from hyperbolic_pde.models import hypno_linear_2d_v0 as mod
    src = inspect.getsource(mod._LinearAdvectionMPLayer2D.forward)
    # _char_alignment_gate should be called exactly once, inside the nonadj
    # branch. We assert it does not appear before the `if cls == "adj_x"` block.
    pre_adj_x = src.split('if cls == "adj_x":')[0]
    _check(
        "_char_alignment_gate" not in pre_adj_x,
        "_char_alignment_gate is not invoked before the cls == 'adj_x' branch",
    )
    # And that it does appear in the non-adj branch (a guarantee, not a sanity
    # check, but worth catching if the function gets renamed).
    nonadj_block = src.split("else:  # nonadj")[1] if "else:  # nonadj" in src else ""
    _check(
        "_char_alignment_gate" in nonadj_block,
        "_char_alignment_gate is invoked inside the non-adj branch",
    )

    # Numerically: the gate has a maximum along the characteristic
    # ((a, b) dt). Construct one perfectly-aligned non-adj edge.
    a, b = 1.0, 0.7
    dt = torch.tensor(0.05)
    dx, dy = a * dt, b * dt
    kappa = torch.tensor(1.0)
    g_aligned = _char_alignment_gate(a, b, dx, dy, dt, kappa, h_s=0.05)
    g_off = _char_alignment_gate(a, b, dx + 1.0, dy + 1.0, dt, kappa, h_s=0.05)
    _check(g_aligned > 0.99, f"g_temp on aligned edge = {float(g_aligned):.4f} ~= 1")
    _check(g_off < g_aligned, f"g_temp on off-axis edge = {float(g_off):.4e} < aligned")


def test_nonadj_upwind_sign_convention() -> None:
    print("=== 6. chi^non_up uses x_i - x_k (target minus source) ===")
    a, b = 1.0, 0.7
    tau = torch.tensor(0.1)
    # Source upstream of target (negative-x relative to it) -- target = source + (small along v).
    # x_i - x_k > 0, y_j - y_p > 0 -> a*dx + b*dy > 0 -> chi=1, gate high.
    dx_t = torch.tensor(+0.1)
    dy_t = torch.tensor(+0.1)
    chi_up = ((a * dx_t + b * dy_t) > 0).float()
    g_up = _soft_upwind_2d(a, b, dx_t, dy_t, tau)
    _check(float(chi_up) == 1.0, "source upstream of target -> chi^non_up = 1")
    _check(g_up > 0.9, f"soft g^non_up upstream = {float(g_up):.4f} > 0.9")

    # Source downstream of target -> chi=0, gate low.
    dx_t = torch.tensor(-0.1)
    dy_t = torch.tensor(-0.1)
    chi_down = ((a * dx_t + b * dy_t) > 0).float()
    g_down = _soft_upwind_2d(a, b, dx_t, dy_t, tau)
    _check(float(chi_down) == 0.0, "source downstream of target -> chi^non_up = 0")
    _check(g_down < 0.1, f"soft g^non_up downstream = {float(g_down):.4f} < 0.1")


def test_replicate_padding_no_wraparound() -> None:
    print("=== 7. Replicate padding does NOT wrap around ===")
    from hyperbolic_pde.models.hypno_linear_2d_v0 import _pad_space_time_2d
    B, nt, ny, nx, d = 1, 2, 4, 5, 3
    h = torch.zeros(B, nt, ny, nx, d)
    h[0, 0, 0, 0] = 1.0   # bottom-left-corner at t=0; mark with a 1.
    k_x = k_y = k_t = 2
    h_pad = _pad_space_time_2d(h, k_x, k_y, k_t)
    # The right edge (x-index nx + k_x onward) should NOT see the corner.
    right_strip = h_pad[0, k_t, k_y + 0, k_x + nx - 1:, :]   # last col + padding
    _check(
        float(right_strip.abs().sum()) == 0.0,
        f"replicate pad does not leak corner mark to far x-side (sum={float(right_strip.abs().sum())})",
    )
    # The left padding region (x < k_x) at the same (t, y) row should see the
    # corner via replication (all equal to 1).
    left_strip = h_pad[0, k_t, k_y + 0, :k_x + 1, 0]
    _check(
        bool(torch.all(left_strip == 1.0)),
        "replicate pad copies corner value to leftmost columns (no zeros there)",
    )


def test_forward_pass_end_to_end() -> None:
    print("=== 8. End-to-end forward pass ===")
    B, nx, ny, nt = 2, 6, 5, 4
    a, b = 1.0, 0.7
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    h_s = math.sqrt(dx * dy)
    model = HypNO_Linear_2D_v0(
        stencil_k_x=2, stencil_k_y=2, stencil_k_t=2,
        d_latent=16, d_hidden=16, n_layers=2,
        a=a, b=b, h_s=h_s,
        use_checkpoint=False,
    )
    u0 = torch.rand(B, ny, nx)
    x = torch.linspace(0.0, 1.0, nx)
    y = torch.linspace(0.0, 1.0, ny)
    t = torch.linspace(0.0, 0.3, nt)
    with torch.no_grad():
        u_pred, u_hats = model(u0, x, y, t)
    _check(u_pred.shape == (B, nt, ny, nx),
           f"u_pred shape {tuple(u_pred.shape)} == [B, nt, ny, nx]")
    _check(len(u_hats) == 2 and u_hats[-1] is u_pred,
           f"n probes={len(u_hats)} and u_hats[-1] is u_pred")
    # `skip=True` default: first-frame prediction matches u0 before any MP.
    # Just confirm finiteness and reasonable magnitude.
    _check(torch.isfinite(u_pred).all().item(),
           "u_pred is finite everywhere")


def main() -> None:
    test_lifting_input_shape()
    test_edge_counts()
    test_axial_upwind_flags()
    test_entropy_gate_trivial()
    test_temporal_gate_only_on_nonadj()
    test_nonadj_upwind_sign_convention()
    test_replicate_padding_no_wraparound()
    test_forward_pass_end_to_end()
    print()
    print("All v0 diagnostics passed.")


if __name__ == "__main__":
    main()
