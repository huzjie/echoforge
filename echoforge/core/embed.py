# -*- coding: utf-8 -*-
"""Deterministic token embeddings and expert specialty vectors.

Everything here is derived from md5(key) seeds, so the same token id always maps
to the same unit vector on every platform and process — the reference is fully
reproducible with no learned weights and no external assets.
"""
from __future__ import annotations

from ..utils.rng import stable_vector
from ..utils.linalg import normalize
from ..constants import VOCAB, DIM


def token_embed(token_id: int, dim: int = DIM) -> list:
    """Unit-norm deterministic embedding for a token id."""
    return normalize(stable_vector("tok:%d" % (token_id % VOCAB), dim))


def expert_vector(expert_id: int, dim: int = DIM) -> list:
    """Unit-norm deterministic 'specialty' vector for a MoE expert id."""
    return normalize(stable_vector("exp:%d" % expert_id, dim))


def embeds(vocab: int = VOCAB, dim: int = DIM) -> list:
    """Precompute the full embedding table (list of rows)."""
    return [token_embed(t, dim) for t in range(vocab)]
