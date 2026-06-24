from .pinn_methods.pinn import UniversalPINN, hyperbolic_residual
from .competitive_architectures.fno import FNO2d, SpectralConv2d
from .competitive_architectures.deeponet import DeepONet
from .pinn_methods.vpinn import VPINN
from .hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3

# PINN-based methods (UniversalPINN, VPINN, ShockDetector) now live under
# ``hyperbolic_pde.models.pinn_methods``; competitive baselines (FNO, DeepONet)
# under ``hyperbolic_pde.models.competitive_architectures``.
#
# Superseded lineages now live under ``hyperbolic_pde.models.legacy`` and are
# no longer re-exported here (HypNO, HypNO_ST, HypNO_ST2, HypNO_ST4, _charcone,
# _pinn, GridGNN, FluxGNN1D, FluxGNN2, HypFluxNO, HypGNO). Import them
# explicitly from there if needed.

__all__ = [
    "UniversalPINN",
    "hyperbolic_residual",
    "FNO2d",
    "SpectralConv2d",
    "FNO2dExperiment",
    "DeepONet",
    "VPINN",
    "HypNO_ST3",
    "precompute_lwr_edge_features_v3",
]
