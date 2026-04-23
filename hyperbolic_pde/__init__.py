from __future__ import annotations

import socket
from pathlib import Path

from .models.pinn import UniversalPINN, hyperbolic_residual
from .models.fno import FNO2d, SpectralConv2d
from .data.fvm import generate_dataset, load_dataset, save_dataset, solve_conservation_fvm
from .data.lax_hopf import solve_lax_hopf

# Map of hostname substring -> remote data root.
# On machines whose hostname contains the key, "hyperbolic_pde/data" in paths
# is replaced with the corresponding value.
_DATA_ROOTS = {
    "dragon": "/work/work_dzdrale/lwr_1d",
}

def resolve_data_path(path: str | Path) -> Path:
    """Remap data paths based on hostname (dragon vs local)."""
    s = str(path)
    hostname = socket.gethostname().lower()
    for key, root in _DATA_ROOTS.items():
        if key in hostname:
            s = s.replace("hyperbolic_pde/data", root)
            break
    return Path(s)

__all__ = [
    "UniversalPINN",
    "hyperbolic_residual",
    "FNO2d",
    "SpectralConv2d",
    "generate_dataset",
    "load_dataset",
    "save_dataset",
    "solve_conservation_fvm",
    "solve_lax_hopf",
    "resolve_data_path",
]
