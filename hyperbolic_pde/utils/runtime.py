from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import torch


def _pick_existing_path(candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand and Path(cand).exists():
            return cand
    return None


def _resolve_path(data_cfg: dict, key: str) -> None:
    env_key = "HPDE_DATA_PATH" if key == "path" else "HPDE_TEST_PATH"
    env_val = os.getenv(env_key)
    if env_val:
        data_cfg[key] = env_val
        return

    current = data_cfg.get(key)
    if current and Path(current).exists():
        return

    local_key = f"{key}_local"
    local_path = data_cfg.get(local_key)
    if local_path and Path(local_path).exists():
        data_cfg[key] = local_path
        return

    remote_key = f"{key}_remote"
    remote_path = data_cfg.get(remote_key)
    if remote_path and Path(remote_path).exists():
        data_cfg[key] = remote_path
        return

    host = socket.gethostname().lower()
    host_map_key = f"{key}_by_host"
    host_map = data_cfg.get(host_map_key, {})
    if isinstance(host_map, dict):
        for host_key, path in host_map.items():
            if host_key.lower() in host and Path(path).exists():
                data_cfg[key] = path
                return

    candidates_key = f"{key}_candidates"
    candidates = data_cfg.get(candidates_key, [])
    if isinstance(candidates, str):
        candidates = [candidates]
    if isinstance(candidates, (list, tuple)):
        picked = _pick_existing_path([str(c) for c in candidates])
        if picked is not None:
            data_cfg[key] = picked
            return


def resolve_data_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        return cfg
    _resolve_path(data_cfg, "path")
    _resolve_path(data_cfg, "test_path")
    return cfg


def resolve_device(cfg: dict[str, Any]) -> str:
    dev = cfg.get("device", "auto")
    dev = str(dev).lower() if dev is not None else "auto"
    if dev in ("auto", "cuda", "gpu"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return dev


def apply_runtime_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    resolve_data_paths(cfg)
    cfg["device"] = resolve_device(cfg)
    return cfg
