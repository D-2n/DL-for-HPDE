from .pinn import BurgersPINN, burgers_residual
from .vpinn import BurgersVPINN, train_vpinn

try:
    from .neural_operators import FNO
except ImportError:
    FNO = None

__all__ = [
    "BurgersPINN",
    "burgers_residual",
    "BurgersVPINN",
    "train_vpinn",
    "FNO",
]
