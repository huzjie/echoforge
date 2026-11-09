# -*- coding: utf-8 -*-
"""HCA hybrid attention + MLA latent compression (long-context memory budget).

This module captures the *memory* side of a long-context model:

* **Full attention** stores a ``d``-dim key and a ``d``-dim value for every token
  of every layer — ``2 * seq_len * d`` floats per layer.
* **MLA** (Multi-head Latent Attention) compresses key/value into a low-rank
  latent of rank ``r``, so a full-context layer costs ``2 * seq_len * r``.
* **HCA** (Hybrid Cache Attention) keeps only a sliding window plus a fixed-size
  global summary, so a compressed-context layer costs ``2 * (window+global) * r``.

In a real stack only one layer in every ``full_ctx_every`` keeps full context
(MLA); the rest run HCA. The measured ``memory_ratio`` below is the real outcome
of that design, and ``needle_recall`` is the honest price you pay: tokens you do
not keep in cache simply cannot be retrieved later.
"""
from __future__ import annotations


def retained_positions(seq_len: int, window: int, n_global: int) -> set:
    """HCA cache = last ``window`` tokens + first ``n_global`` tokens (summary)."""
    positions = set()
    for p in range(max(0, seq_len - window), seq_len):
        positions.add(p)
    for p in range(min(n_global, seq_len)):
        positions.add(p)
    return positions


def needle_recall(needle_positions, retained) -> float:
    """Fraction of needle positions present in the cache (0 .. 1)."""
    if not needle_positions:
        return 0.0
    return sum(1 for p in needle_positions if p in retained) / len(needle_positions)


def full_attention_kv(seq_len: int, dim: int) -> float:
    return seq_len * 2 * dim


def mla_kv(seq_len: int, latent_rank: int) -> float:
    return seq_len * 2 * latent_rank


def hca_kv(window: int, n_global: int, latent_rank: int) -> float:
    return (window + n_global) * 2 * latent_rank


def memory_ratio(seq_len: int, dim: int, latent_rank: int,
                 window: int, n_global: int, full_ctx_every: int) -> float:
    """Average per-layer KV size relative to full attention (lower is better)."""
    full = full_attention_kv(seq_len, dim)
    if full == 0:
        return 1.0
    mla_full = mla_kv(seq_len, latent_rank)
    hca = hca_kv(window, n_global, latent_rank)
    avg = (1.0 / full_ctx_every) * mla_full + ((full_ctx_every - 1.0) / full_ctx_every) * hca
    return avg / full
