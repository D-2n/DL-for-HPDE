from .pinn import UniversalPINN, hyperbolic_residual
from .fno import FNO2d, SpectralConv2d
from .fno_experiment import FNO2d as FNO2dExperiment
from .deeponet import DeepONet
from .vpinn import VPINN
from .fluxgnn import FluxGNN1D
from .hypgno import HypGNO
from .hypno import HypNO
from .hypno_pinn import HypNO_PINN
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
    #"GridGNN",
]
