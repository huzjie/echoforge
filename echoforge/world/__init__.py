# -*- coding: utf-8 -*-
"""World: deterministic long-context corpus + the three-axis metric runner."""
from __future__ import annotations

from .generator import generate_sequence, needle_positions
from .corpus import Corpus, build_corpus
from .oracle import EchoWorld

__all__ = ["generate_sequence", "needle_positions", "Corpus", "build_corpus", "EchoWorld"]
