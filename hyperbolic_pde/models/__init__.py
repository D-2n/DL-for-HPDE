from .pinn import UniversalPINN, hyperbolic_residual
from .fno import FNO2d, SpectralConv2d
from .fno_experiment import FNO2d as FNO2dExperiment
from .deeponet import DeepONet
from .vpinn import VPINN
from .fluxgnn import FluxGNN1D
from .hypgno import HypGNO
from .hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3

# Superseded LWR lineages (HypNO, HypNO_ST, HypNO_ST2, HypNO_ST4, _charcone,
# _pinn, GridGNN) now live under ``hyperbolic_pde.models.legacy`` and are no
# longer re-exported here. Import them explicitly from there if needed.

__all__ = [
    "UniversalPINN",
    "hyperbolic_residual",
    "FNO2d",
    "SpectralConv2d",
    "FNO2dExperiment",
    "DeepONet",
    "VPINN",
    "FluxGNN1D",
    "HypGNO",
    "HypNO_ST3",
    "precompute_lwr_edge_features_v3",
]
