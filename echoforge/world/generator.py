# -*- coding: utf-8 -*-
"""Deterministic long-context sequence + needle corpus generation."""
from __future__ import annotations

from ..utils.rng import stable_rand


def generate_sequence(seed, seq_len: int) -> list:
    """A deterministic token-id sequence of length ``seq_len`` (md5-seeded)."""
    r = stable_rand("seq:%d" % seed)
    return [r.randrange(256) for _ in range(seq_len)]


def needle_positions(n_needles: int) -> list:
    """The needle positions: the earliest ``n_needles`` positions of a sequence.

    These are the long-context tokens the HCA global summary must decide to keep
    in order for a later query to retrieve them.
    """
    return list(range(n_needles))
