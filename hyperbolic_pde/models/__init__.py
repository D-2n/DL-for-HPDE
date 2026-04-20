from .pinn import UniversalPINN, hyperbolic_residual
from .fno import FNO2d, SpectralConv2d
from .fno_experiment import FNO2d as FNO2dExperiment
from .deeponet import DeepONet
from .vpinn import VPINN
from .fluxgnn import FluxGNN1D
from .hypgno import HypGNO
from .hypno import HypNO
from .hypno_pinn import HypNO_PINN
from .hypno_st import HypNO_ST
from .hypno_st2 import HypNO_ST2
from .hypno_st3 import HypNO_ST3, precompute_lwr_edge_features_v3

#from .gnn import GridGNN

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
    "HypNO",
    "HypNO_PINN",
    "HypNO_ST",
    "HypNO_ST2",
    "HypNO_ST3",
    "precompute_lwr_edge_features_v3",
    "HypNO_ST4",
    #"GridGNN",
]
