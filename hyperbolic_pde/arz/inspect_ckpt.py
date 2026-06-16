"""Inspect an ARZ checkpoint's format + architecture from its state_dict.

For old/undocumented checkpoints (no sibling config.yaml): reports whether it is
a legacy dict ({model, args, tau} -> self-describing) or a bare state_dict, and
prints the architecture-defining tensor shapes so the right model class / config
section can be reconstructed.

    python -m hyperbolic_pde.arz.inspect_ckpt /path/to/best.pt
"""
from __future__ import annotations

import argparse
import re

import torch


# Param keys whose shapes pin down the architecture.
_ARCH_KEYS = (
    "node_mlp.0.weight",          # mark1 lifting node MLP in-dim (7 vs 9)
    "lifting.node_mlp.0.weight",
    "adj_edge_mlp.0.weight",      # mark1 lifting adj edge MLP
    "nonadj_edge_mlp.0.weight",
    "adj_msg.0.weight",           # MP-layer message MLPs (mark1 / mark2 / mark1_router)
    "nonadj_msg.0.weight",
    "msg_gnl.0.weight",           # mark2 two-family branches (presence => mark2/2.1)
    "msg_ld.0.weight",
    "decoder.0.weight",
    "combine.0.weight",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"top-level type: {type(raw).__name__}")

    sd = raw
    if isinstance(raw, dict) and "model" in raw:
        print("=> LEGACY dict format (self-describing, no config.yaml needed)")
        print("   non-model keys:", [k for k in raw if k != "model"])
        if "args" in raw:
            print("   args:", raw["args"])
        if "tau" in raw:
            print("   tau:", raw.get("tau"))
        sd = raw["model"]
    else:
        print("=> BARE state_dict (needs a matching config.yaml / --config to load)")

    # Strip torch.compile prefix if present.
    if isinstance(sd, dict) and any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        print("   (stripped _orig_mod. compile prefix)")

    ks = list(sd.keys())
    print(f"n params: {len(ks)}")

    print("architecture-defining shapes:")
    found = False
    for k in ks:
        if any(k.endswith(a) for a in _ARCH_KEYS):
            print(f"  {k}  {tuple(sd[k].shape)}")
            found = True
    if not found:
        print("  (none of the known arch keys matched -- dumping first 12 keys)")
        for k in ks[:12]:
            print(f"  {k}  {tuple(sd[k].shape)}")

    # Infer layer count.
    nl = max(
        [int(m.group(1)) for k in ks for m in [re.search(r"mp_layers\.(\d+)\.", k)] if m],
        default=-1,
    ) + 1
    print(f"inferred n_layers (mp_layers.*): {nl}")

    # Quick family hint.
    has_gnl = any(k.endswith("msg_gnl.0.weight") for k in ks)
    has_adjmsg = any(k.endswith("adj_msg.0.weight") for k in ks)
    has_adjedge = any(k.endswith("adj_edge_mlp.0.weight") for k in ks)
    if has_gnl:
        print("family hint: mark2 / mark2_1 (msg_gnl/msg_ld present)")
    elif has_adjmsg and has_adjedge:
        print("family hint: mark1 / mark1_router (adj_msg + adj_edge_mlp present)")
    else:
        print("family hint: indeterminate")


if __name__ == "__main__":
    main()
