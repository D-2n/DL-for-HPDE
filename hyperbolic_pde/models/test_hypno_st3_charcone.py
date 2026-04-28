"""Smoke test for hypno_st3_charcone.py.

Checks:
1. Forward pass with B=2, nx=32, nt=16 returns expected output shapes.
2. Non-adjacent MLP input dimension matches the actual concatenated feature dim.
3. Cone offsets for k_x=5, k_t=4, dx=dt=1 are fewer than rectangular and
   include the two same-time adjacent edges.
"""
import sys
from pathlib import Path

# Allow running from repo root or from this file's directory.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from hyperbolic_pde.models.hypno_st3_charcone import (
    HypNO_ST3,
    HypNO_ST3_CharCone,
    _enumerate_hyperbolic_cone_offsets,
    _enumerate_ball_offsets,
    _PhysicsSpaceTimeMPLayer,
)


# ---- 1. Forward pass shapes ------------------------------------------------ #
B, nx, nt = 2, 32, 16
u0 = torch.rand(B, nx)
x  = torch.linspace(0, 1, nx)
t  = torch.linspace(0, 1, nt)

model = HypNO_ST3(
    stencil_k_x=3,
    stencil_k_t=2,
    d_latent=16,
    d_hidden=32,
    n_layers=2,
    activation="gelu",
    encoder_scaling="classic",
    encoder_type="gnn",
    skip=True,
    use_checkpoint=False,
)

u_pred, u_coarse, shock, u_hats = model(u0, x, t)

assert u_pred.shape == (B, nt, nx), f"u_pred shape mismatch: {u_pred.shape}"
assert u_coarse.shape == (B, nt, nx), f"u_coarse shape mismatch: {u_coarse.shape}"
assert shock.shape == (B, nt, nx), f"shock shape mismatch: {shock.shape}"
assert len(u_hats) == 2, f"expected 2 u_hat entries, got {len(u_hats)}"
assert u_hats[0].shape == (B, nt, nx), f"u_hats[0] shape mismatch: {u_hats[0].shape}"
print("PASS  forward pass shapes:", u_pred.shape, u_coarse.shape, shock.shape)


# ---- 2. Non-adjacent MLP input dimension ----------------------------------- #
# The nonadj_msg MLP was built with 2*d_latent+13.
# Verify this matches what _build_edge actually concatenates.
# We trace the dimension by hand: 2*d + 8 static node feats + 5 geometric feats = 2d+13.
#   h_i(d), h_j(d), u_i(1), u_j(1), f_i(1), f_j(1), a_i(1), a_j(1),   <- 2d+6
#   rel_x(1), rel_t(1), v_edge(1), char_res(1), cfl(1), sign(rel_x)(1), xi(1)  <- +7
# total = 2d + 13
d_latent = 16
expected_nonadj_in = 2 * d_latent + 13
# Read the actual first-layer weight shape.
layer = model.mp_layers[0]
actual_in = layer.nonadj_msg[0].in_features
assert actual_in == expected_nonadj_in, (
    f"nonadj_msg input dim mismatch: built={actual_in}, expected={expected_nonadj_in}"
)
print(f"PASS  nonadj_msg input dim = {actual_in} (= 2*{d_latent}+13)")


# ---- 3. Cone offset count vs rectangular ----------------------------------- #
k_x, k_t, dx, dt = 5, 4, 1.0, 1.0

cone_offsets = _enumerate_hyperbolic_cone_offsets(k_x, k_t, dx, dt, c_max=1.0, buffer_cells=1, causal=True)
rect_offsets = _enumerate_ball_offsets(k_x, k_t, causal=True)

cone_set = set(cone_offsets)
rect_set = set(rect_offsets)

assert len(cone_offsets) < len(rect_offsets), (
    f"cone ({len(cone_offsets)}) should be smaller than rect ({len(rect_offsets)})"
)
assert (-1, 0) in cone_set, "cone missing (-1, 0)"
assert (1, 0) in cone_set, "cone missing (1, 0)"

# No same-time non-adjacent spatial edges.
for di in range(-k_x, k_x + 1):
    if abs(di) > 1:
        assert (di, 0) not in cone_set, f"cone should not contain ({di}, 0)"

print(f"PASS  cone offsets = {len(cone_offsets)}, rect offsets = {len(rect_offsets)}")
print(f"      cone includes (-1,0)={((-1,0) in cone_set)}, (1,0)={(((1,0)) in cone_set)}")
print(f"      cone offsets: {sorted(cone_offsets)}")

# Alias sanity check.
assert HypNO_ST3_CharCone is HypNO_ST3
print("PASS  HypNO_ST3_CharCone alias correct")

print("\nAll tests passed.")
