from .pinn import UniversalPINN, hyperbolic_residual
from .fno import FNO2d, SpectralConv2d
from .deeponet import DeepONet

__all__ = [
    "UniversalPINN",
    "hyperbolic_residual",
    "FNO2d",
    "SpectralConv2d",
    "DeepONet",
]
