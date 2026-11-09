# -*- coding: utf-8 -*-
"""Runtime configuration with layered defaults (code default < recipe yaml < cli)."""
from __future__ import annotations

import copy

from .utils.yamlish import yaml_loads
from .constants import (
    VOCAB, DIM, SEQ_LEN, LAYERS, FULL_CTX_EVERY, MLA_RANK,
    WINDOW, MAX_GLOBAL, N_EXPERTS, TOP_K, MAX_ROUTERS, K_STEPS, MAX_RANK,
)

# Defaults are deliberately small so a reference run stays interactive (<1s on a
# laptop) while still exercising the full HCA-attention + multi-router MoE +
# ECHO multi-step pipeline, the verifiable-reward audit and the migration
# metrics. The real ERNIE 5.5 Flash Base is a 280B-parameter MoE; the reference
# keeps the same architectural protocol shape on a laptop-friendly scale.
DEFAULTS = {
    "attention": {
        "vocab": VOCAB,          # vocabulary size
        "dim": DIM,              # embedding / hidden dimension
        "seq_len": SEQ_LEN,      # reference sequence length
        "layers": LAYERS,        # transformer layers
        "full_ctx_every": FULL_CTX_EVERY,  # 1 in N layers keeps full context (MLA)
        "mla_rank": MLA_RANK,    # MLA latent key/value rank
        "window": WINDOW,        # HCA sliding-window size
        "max_global": MAX_GLOBAL,  # HCA global summary capacity
    },
    "moe": {
        "n_experts": N_EXPERTS,  # experts per layer
        "top_k": TOP_K,          # experts selected per token
        "max_routers": MAX_ROUTERS,  # router-ensemble size ceiling
        "eps": 1.0,              # single-router noise magnitude
    },
    "echo": {
        "k_steps": K_STEPS,      # multi-step prediction horizon
        "max_rank": MAX_RANK,    # ECHO projection rank ceiling (= k_steps)
    },
    "model": {
        "skill": 0.5,            # alignment: 0.0 baseline .. 1.0 fully honest
    },
    "n_sequences": 8,            # deterministic corpus size
    "bench": {
        "skills": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
    "report": {
        "format": "markdown",    # markdown | json
        "out_dir": "reports",
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(recipe_path=None):
    """Return the merged config dict. A recipe (yaml subset) overrides defaults."""
    cfg = copy.deepcopy(DEFAULTS)
    if recipe_path:
        with open(recipe_path, "r", encoding="utf-8") as f:
            raw = yaml_loads(f.read())
        cfg = _merge(cfg, raw)
    return cfg
