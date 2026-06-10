"""Verify the ARZ characteristic projection isolates the two wave families.

The HypNO-ARZ redesign feeds an adjacent-edge state jump, decomposed into its
two characteristic coordinates, to two separate family MLPs (1-wave / GNL and
2-contact / LD). The decomposition is a *basis change* into the eigenbasis of
the primitive-variable Jacobian, NOT an orthogonal projection: the right
eigenvectors are not orthogonal, so the coordinates come from the inverse of
the eigenvector matrix, equivalently from dotting with the LEFT eigenvectors.

Eigenstructure in primitive variables U = (rho, v), with P := p'(rho_ij):

    r1 = (1, -P)^T   (1-wave, GNL; w = v + p is the Riemann invariant)
    r2 = (1,  0)^T   (2-contact, LD; v is the Riemann invariant)

    R = [r1 r2] = [[1, 1], [-P, 0]],   det R = P,
    R^{-1} = (1/P) [[0, -1], [P, 1]],

    => left eigenvectors (rows of R^{-1}):
       l1 = (0, -1/P),   l2 = (1, 1/P).

    Projection of a jump dU = (drho, dv):
       beta1 = l1 . dU = -dv / P                  (1-wave coordinate)
       beta2 = l2 . dU =  drho + dv / P           (2-contact coordinate)

Family-isolation properties this script asserts, using the EXACT homogeneous
Riemann solver (riemann_arz) to build genuine pure-family states:

  * Pure 2-contact (v_L == v_R, so dv == 0 at the interface):
        beta1 == 0,  beta2 != 0.      The 1-wave reads "absent".
  * Pure 1-wave (w_L == w_R, the 1-Riemann invariant preserved):
        beta2 == 0,  beta1 != 0.      The contact reads "absent".

It also checks biorthogonality l_m . r_n = delta_mn and that the two equivalent
projection routes (R^{-1} dU  vs  dotting with l1, l2) agree to machine eps.

Pure numpy, no torch, no dataset, no GPU. Run directly or under pytest.
"""
from __future__ import annotations

import argparse

import numpy as np

from hyperbolic_pde.arz import physics_arz as P


# --------------------------------------------------------------------------- #
# The projection under test (mirrors what the model layer will compute)
# --------------------------------------------------------------------------- #
def eigenvectors_at(rho_ij: float):
    """Right (r1, r2) and left (l1, l2) eigenvectors in (rho, v) at rho_ij."""
    Pp = float(P.dpressure(np.array(rho_ij)))   # p'(rho_ij); = 1 + 2*rho_ij for rho+rho^2
    r1 = np.array([1.0, -Pp])
    r2 = np.array([1.0, 0.0])
    l1 = np.array([0.0, -1.0 / Pp])
    l2 = np.array([1.0, 1.0 / Pp])
    return r1, r2, l1, l2, Pp


def project_jump(rho_i, w_i, rho_j, w_j):
    """Characteristic coordinates (beta1, beta2) of the interface state jump.

    Jump is taken in PRIMITIVE (rho, v); eigenvectors are evaluated at the
    arithmetic interface state rho_ij = 0.5*(rho_i + rho_j).
    Returns (beta1, beta2, drho, dv, P_ij).
    """
    v_i = w_i - float(P.pressure(np.array(rho_i)))
    v_j = w_j - float(P.pressure(np.array(rho_j)))
    drho = rho_j - rho_i
    dv = v_j - v_i
    rho_ij = 0.5 * (rho_i + rho_j)
    Pp = float(P.dpressure(np.array(rho_ij)))
    beta1 = -dv / Pp
    beta2 = drho + dv / Pp
    return beta1, beta2, drho, dv, Pp


def _interface_states_across_contact(rho_L, w_L, rho_R, w_R):
    """Return the two states straddling the 2-contact: (rho_*, w_*) | (rho_R, w_R).

    The intermediate state has w_* = w_L (1-wave preserves w) and v_* = v_R
    (contact preserves v); the contact is the jump (rho_*, w_*) -> (rho_R, w_R).
    """
    v_R = w_R - float(P.pressure(np.array(rho_R)))
    rho_star = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * max(w_L - v_R, 0.0)))
    if P.get_pressure_form() == "rho":
        rho_star = max(w_L - v_R, 0.0)
    w_star = w_L
    return (rho_star, w_star), (rho_R, w_R)


def _interface_states_across_1wave(rho_L, w_L, rho_R, w_R):
    """Return the two states straddling the 1-wave: (rho_L, w_L) | (rho_*, w_*).

    Across the 1-wave w is preserved (w_* = w_L); the jump is
    (rho_L, w_L) -> (rho_*, w_*).
    """
    v_R = w_R - float(P.pressure(np.array(rho_R)))
    rho_star = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * max(w_L - v_R, 0.0)))
    if P.get_pressure_form() == "rho":
        rho_star = max(w_L - v_R, 0.0)
    w_star = w_L
    return (rho_L, w_L), (rho_star, w_star)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_biorthogonality(rho_ij: float = 0.4, tol: float = 1e-12) -> None:
    r1, r2, l1, l2, _ = eigenvectors_at(rho_ij)
    M = np.array([[l1 @ r1, l1 @ r2],
                  [l2 @ r1, l2 @ r2]])
    err = np.max(np.abs(M - np.eye(2)))
    assert err < tol, f"biorthogonality l_m.r_n != delta_mn (err={err:.2e})\n{M}"


def check_inverse_route_agrees(rho_ij: float = 0.4, tol: float = 1e-12) -> None:
    """beta = R^{-1} dU must equal (l1.dU, l2.dU) for an arbitrary jump."""
    r1, r2, l1, l2, _ = eigenvectors_at(rho_ij)
    R = np.column_stack([r1, r2])
    dU = np.array([0.137, -0.291])  # arbitrary
    beta_inv = np.linalg.solve(R, dU)
    beta_dot = np.array([l1 @ dU, l2 @ dU])
    err = np.max(np.abs(beta_inv - beta_dot))
    assert err < tol, f"R^-1 route disagrees with left-eigenvector dot (err={err:.2e})"


def check_pure_contact(tol: float = 1e-9) -> dict:
    """v_L == v_R => the only wave is the 2-contact => beta1 == 0 at the jump."""
    # Pick L and R sharing the same velocity but different density.
    rho_L, rho_R = 0.25, 0.6
    v_common = 0.3
    w_L = v_common + float(P.pressure(np.array(rho_L)))
    w_R = v_common + float(P.pressure(np.array(rho_R)))
    (rs, ws), (rR, wR) = _interface_states_across_contact(rho_L, w_L, rho_R, w_R)
    beta1, beta2, drho, dv, Pp = project_jump(rs, ws, rR, wR)
    assert abs(beta1) < tol, (
        f"pure contact: beta1 should be ~0, got {beta1:.3e} (dv={dv:.3e})"
    )
    assert abs(beta2) > 1e-3, f"pure contact: beta2 should be nonzero, got {beta2:.3e}"
    return dict(case="pure_contact", beta1=beta1, beta2=beta2, drho=drho, dv=dv)


def check_pure_1wave(tol: float = 1e-9) -> dict:
    """w_L == w_R => the contact is trivial (w preserved across 1-wave AND the
    intermediate already matches R in w) => the only wave is the 1-wave =>
    beta2 == 0 at the 1-wave jump."""
    # w preserved across the 1-wave; choose w_R == w_L so the * state's jump to
    # R carries no w-jump, isolating the 1-wave on the (L -> *) interface.
    rho_L, rho_R = 0.55, 0.2
    w_common = float(P.w_eq(np.array(0.4)))
    w_L = w_common
    w_R = w_common
    (rL, wL), (rs, ws) = _interface_states_across_1wave(rho_L, w_L, rho_R, w_R)
    beta1, beta2, drho, dv, Pp = project_jump(rL, wL, rs, ws)
    # Across a 1-wave w is invariant, so dw = 0 => dv = -P*drho at the interface,
    # which forces beta2 = drho + dv/P = drho - drho = 0.
    assert abs(beta2) < 5e-3, (
        f"pure 1-wave: beta2 should be ~0, got {beta2:.3e} "
        f"(this is exact only at the linearization point rho_ij; small residual "
        f"is the finite-jump curvature of p)"
    )
    assert abs(beta1) > 1e-3, f"pure 1-wave: beta1 should be nonzero, got {beta1:.3e}"
    return dict(case="pure_1wave", beta1=beta1, beta2=beta2, drho=drho, dv=dv)


def check_pure_1wave_infinitesimal(tol: float = 1e-10) -> dict:
    """Exact version of the 1-wave isolation: in the infinitesimal-jump limit
    (where the arithmetic interface state IS the exact linearization point),
    a w-preserving jump must give beta2 == 0 to machine precision.

    Build a tiny jump along the 1-eigenvector direction r1 = (1, -P) and confirm
    beta2 vanishes exactly.
    """
    rho0, v0 = 0.4, 0.3
    Pp = float(P.dpressure(np.array(rho0)))
    eps = 1e-6
    drho = eps
    dv = -Pp * eps                       # move along r1 => w preserved
    rho_i, rho_j = rho0, rho0 + drho
    v_i, v_j = v0, v0 + dv
    w_i = v_i + float(P.pressure(np.array(rho_i)))
    w_j = v_j + float(P.pressure(np.array(rho_j)))
    beta1, beta2, _, _, _ = project_jump(rho_i, w_i, rho_j, w_j)
    # Interface rho_ij = rho0 + drho/2; p' there differs from p'(rho0) by O(drho),
    # so beta2 = O(drho^2). Assert it's tiny relative to beta1.
    assert abs(beta2) < 1e-9, f"infinitesimal 1-wave: beta2={beta2:.3e} not ~0"
    return dict(case="pure_1wave_infinitesimal", beta1=beta1, beta2=beta2)


def run_all() -> int:
    print(f"[verify_char_projection] pressure_form = {P.get_pressure_form()!r}\n")

    check_biorthogonality()
    print("  [ok] biorthogonality  l_m . r_n = delta_mn")

    check_inverse_route_agrees()
    print("  [ok] R^{-1} dU  ==  (l1.dU, l2.dU)")

    c = check_pure_contact()
    print(f"  [ok] pure 2-contact:  beta1={c['beta1']:+.3e} (~0)  "
          f"beta2={c['beta2']:+.3e}  | drho={c['drho']:+.3e} dv={c['dv']:+.3e}")

    w1 = check_pure_1wave()
    print(f"  [ok] pure 1-wave:     beta2={w1['beta2']:+.3e} (~0)  "
          f"beta1={w1['beta1']:+.3e}  | drho={w1['drho']:+.3e} dv={w1['dv']:+.3e}")

    wi = check_pure_1wave_infinitesimal()
    print(f"  [ok] 1-wave (eps lim): beta2={wi['beta2']:+.3e} (==0 to eps)  "
          f"beta1={wi['beta1']:+.3e}")

    print("\nAll characteristic-projection checks passed.")
    return 0


# --------------------------------------------------------------------------- #
# pytest entry points
# --------------------------------------------------------------------------- #
def test_biorthogonality():
    check_biorthogonality()


def test_inverse_route_agrees():
    check_inverse_route_agrees()


def test_pure_contact_isolates_family():
    check_pure_contact()


def test_pure_1wave_isolates_family():
    check_pure_1wave()


def test_pure_1wave_infinitesimal():
    check_pure_1wave_infinitesimal()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pressure-form", choices=("rho", "rho+rho2"), default=None,
        help="Override the global ARZ pressure closure for the run.",
    )
    args = ap.parse_args()
    if args.pressure_form is not None:
        P.set_pressure_form(args.pressure_form)
    raise SystemExit(run_all())
