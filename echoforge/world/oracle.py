# -*- coding: utf-8 -*-
"""EchoWorld: wire the corpus + config into the three-axis metric runner."""
from __future__ import annotations

from .corpus import build_corpus
from ..core import metrics as M


class EchoWorld:
    """Bundle the deterministic corpus with the closed-form oracles.

    ``run(skill)`` drives the three honesty axes (HCA cache / router ensemble /
    ECHO rank) with ``skill`` and returns the four monotone metrics plus the
    migration metrics. See ``core/metrics.py`` for the exact construction.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.corpus = build_corpus(cfg)
        self.sequences = self.corpus.sequences
        self.needles = self.corpus.needles
        self.n_sequences = self.corpus.n_sequences
        self.seq_len = self.corpus.seq_len

    def run(self, skill: float) -> dict:
        return M.aggregate(self, skill)

    def params_at(self, skill: float):
        return M.skill_params(self.cfg, skill)
