"""ARZ (Aw-Rascle-Zhang, 1D, with relaxation) extension of HypNO.

Mirrors the scalar-LWR pipeline file-for-file. See ARZ_DEVELOPMENT_PLAN.md
at the repo root for the spec.

Frozen physics:
    p(rho)  = rho + rho^2
    p'(rho) = 1 + 2*rho
    V(rho)  = 1 - rho                  (Greenshields, inherited from LWR)
    v       = w - p(rho)
    lambda1 = v - rho * p'(rho)
    lambda2 = v
    U       = (rho, y), y = rho * w
    F1      = rho * v
    F2      = y   * v
    w_eq    = V(rho) + p(rho) = 1 + rho^2
    S       = (0, (y_eq - y) / tau)
"""
