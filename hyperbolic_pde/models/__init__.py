from .pinn import UniversalPINN, hyperbolic_residual
from .fno import FNO2d, SpectralConv2d
from .fno_experiment import FNO2d as FNO2dExperiment
from .deeponet import DeepONet
from .vpinn import VPINN
from .fluxgnn import FluxGNN1D
from .hypgno import HypGNO
from .hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3

# Superseded LWR lineages live under .legacy; re-exported here for back-compat.
from .legacy.hypno import HypNO
from .legacy.hypno_st import HypNO_ST
from .legacy.hypno_st2 import HypNO_ST2

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
    "HypNO",
    "HypNO_ST",
    "HypNO_ST2",
]
