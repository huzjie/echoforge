# -*- coding: utf-8 -*-
"""Multi-router latent MoE: single router vs router ensemble (routing utility).

A single router scores experts with a deterministic signal plus per-(token,
expert) noise. The canonical specialist expert for token ``t`` is ``t % E``; the
router is *correct* when that expert survives into the top-k selection.

A multi-router ensemble averages the router scores, which shrinks the noise by
``1 / sqrt(R)`` — the standard ensemble variance reduction. The reference models
this explicitly as ``noise_scale(R) = eps / sqrt(R)`` over a fixed per-(token,
expert) noise value, so *routing utility is monotonically non-decreasing in the
number of routers*: more routers -> less noise -> the true expert is selected
more reliably.
"""
from __future__ import annotations

import math

from ..utils.rng import stable_float
from ..constants import N_EXPERTS, TOP_K, MAX_ROUTERS


def noise_scale(n_routers: int, eps: float = 1.0) -> float:
    """Ensemble noise magnitude, decreasing as ``eps / sqrt(R)``."""
    return eps / math.sqrt(max(1, int(n_routers)))


def expert_scores(token_id: int, n_experts: int, n_routers: int, eps: float = 1.0) -> list:
    """Deterministic router scores: true expert = 1.0 + noise, others = noise."""
    g = token_id % n_experts
    scale = noise_scale(n_routers, eps)
    scores = []
    for e in range(n_experts):
        base = 1.0 if e == g else 0.0
        z = stable_float("moe:%d:%d" % (token_id, e), -1.0, 1.0)
        scores.append(base + scale * z)
    return scores


def select_topk(scores: list, top_k: int) -> set:
    """Indices of the ``top_k`` highest-scoring experts."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return set(order[:top_k])


def router_hits_true(token_id: int, n_experts: int, top_k: int,
                     n_routers: int, eps: float = 1.0) -> bool:
    """True if the canonical expert for ``token_id`` survives top-k selection."""
    g = token_id % n_experts
    return g in select_topk(expert_scores(token_id, n_experts, n_routers, eps), top_k)


def routing_utility(token_ids, n_experts: int, top_k: int,
                    n_routers: int, eps: float = 1.0) -> float:
    """Fraction of tokens whose true expert is selected (0 .. 1)."""
    if not token_ids:
        return 0.0
    hits = sum(1 for t in token_ids if router_hits_true(t, n_experts, top_k, n_routers, eps))
    return hits / len(token_ids)
