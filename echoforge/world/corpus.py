# -*- coding: utf-8 -*-
"""Corpus: deterministic sequences + aligned audit rounds."""
from __future__ import annotations

from ..constants import N_ROUNDS
from .generator import generate_sequence, needle_positions


class Corpus:
    """A small deterministic long-context corpus used to audit the model."""

    def __init__(self, cfg):
        self.cfg = cfg
        attn = cfg["attention"]
        self.seq_len = attn["seq_len"]
        self.n_sequences = cfg.get("n_sequences", 8)
        self.sequences = [generate_sequence(i, self.seq_len)
                          for i in range(self.n_sequences)]
        # needles are sequence-relative positions of the earliest tokens
        self.needles = needle_positions(N_ROUNDS)
        self.n_rounds = N_ROUNDS


def build_corpus(cfg) -> Corpus:
    """Build the deterministic corpus from config."""
    return Corpus(cfg)
