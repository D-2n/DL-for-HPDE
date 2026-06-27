"""Precomputed numerical-baseline cache for a FIXED ARZ eval set.

The FV schemes (godunov / hll / weno5) are deterministic functions of the IC, so
for a frozen eval set their solutions never change. Precompute them once
(`arz_precompute_baselines.py`), store to a sibling `.npz`, and load at eval time
so paper-eval / shock-export don't re-solve the schemes on every run.

Cache layout (one npz):
    <method>__rho : (N, nt, nx) float32   # failed samples stored as all-NaN
    <method>__w   : (N, nt, nx) float32
    __methods     : array of method names present
    __n_samples   : int
    __ic_checksum : float  # fingerprint of (rho0, w0) -> ties cache to its set
    __p_form      : str

Loading validates N and the IC checksum so a cache can never be silently paired
with a different eval set.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def ic_checksum(bundle) -> float:
    """Cheap content fingerprint of the IC fields (independent of method)."""
    rho0 = np.asarray(bundle.rho0, dtype=np.float64)
    w0 = np.asarray(bundle.w0, dtype=np.float64)
    return float(rho0.sum() + 1.000001 * w0.sum())


def save_baseline_cache(path, bundle, solutions: dict) -> None:
    """Write the cache. `solutions`: {method: {'rho': (N,nt,nx), 'w': (N,nt,nx)}}."""
    out: dict = {}
    for m, d in solutions.items():
        out[f"{m}__rho"] = np.asarray(d["rho"], dtype=np.float32)
        out[f"{m}__w"] = np.asarray(d["w"], dtype=np.float32)
    out["__methods"] = np.array(sorted(solutions.keys()))
    out["__n_samples"] = np.int64(bundle.rho.shape[0])
    out["__ic_checksum"] = np.float64(ic_checksum(bundle))
    out["__p_form"] = np.array(str(bundle.p_form))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)


def load_baseline_cache(path, bundle, strict: bool = True) -> dict:
    """Return {method: {'rho','w'}}, validated against `bundle`. Raises on mismatch."""
    z = np.load(path, allow_pickle=False)
    n = int(z["__n_samples"])
    if n != bundle.rho.shape[0]:
        raise ValueError(
            f"baseline cache N={n} != eval set N={bundle.rho.shape[0]} ({path})."
        )
    if strict:
        chk = float(z["__ic_checksum"])
        want = ic_checksum(bundle)
        if abs(chk - want) > 1e-3 * (abs(want) + 1.0):
            raise ValueError(
                f"baseline cache IC checksum {chk:.6g} != eval set {want:.6g} "
                f"({path}) -> cache was built for a DIFFERENT eval set. Regenerate."
            )
    methods = [str(m) for m in z["__methods"]]
    return {m: {"rho": z[f"{m}__rho"], "w": z[f"{m}__w"]} for m in methods}


def cached_sample(cache: dict | None, method: str, i: int):
    """Return (rho_i, w_i) for one cached sample, or None if absent / failed (NaN).

    None means "fall back to solving / treat as a baseline failure on this sample".
    """
    if cache is None or method not in cache:
        return None
    rho_i = cache[method]["rho"][i]
    w_i = cache[method]["w"][i]
    if not (np.isfinite(rho_i).all() and np.isfinite(w_i).all()):
        return None
    return rho_i.astype(np.float64), w_i.astype(np.float64)
