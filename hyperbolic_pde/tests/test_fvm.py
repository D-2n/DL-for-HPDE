from __future__ import annotations

from hyperbolic_pde.data.fvm import generate_dataset, godunov_flux
import numpy as np


def test_godunov_flux_shapes() -> None:
    u_l = np.array([0.1, 0.6, 0.9], dtype=np.float32)
    u_r = np.array([0.2, 0.4, 0.7], dtype=np.float32)
    f = godunov_flux(u_l, u_r)
    assert f.shape == u_l.shape
    assert np.all(np.isfinite(f))


def test_generate_dataset_shapes() -> None:
    bundle = generate_dataset(
        num_samples=2,
        nx=16,
        nt=8,
        x_min=-1.0,
        x_max=1.0,
        t_max=0.5,
        cfl=0.4,
        num_segments=3,
        u_min=0.0,
        u_max=1.0,
        ic_points=4,
        boundary="periodic",
        seed=123,
    )
    assert bundle.u.shape == (2, 8, 16)
    assert bundle.u0.shape == (2, 16)
    assert bundle.ic.shape == (2, 4)
