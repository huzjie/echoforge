# -*- coding: utf-8 -*-
"""Wire everything into a ready-to-use kit (corpus + world + audit)."""
from __future__ import annotations

from .world.corpus import build_corpus
from .world.oracle import EchoWorld
from .training.reward import EchoAudit


def build_kit(cfg):
    corpus = build_corpus(cfg)
    world = EchoWorld(cfg)
    audit = EchoAudit(cfg, world)
    return {
        "cfg": cfg,
        "corpus": corpus,
        "world": world,
        "audit": audit,
    }
