# -*- coding: utf-8 -*-
"""Shared test helpers."""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def small_cfg(**overrides):
    """A minimal, fast config dict for tests."""
    from echoforge.config import DEFAULTS
    cfg = copy.deepcopy(DEFAULTS)
    cfg["n_sequences"] = 4
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg
