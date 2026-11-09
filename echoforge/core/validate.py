# -*- coding: utf-8 -*-
"""Verifiable checks: needle retained / true-expert hit / echo step correct.

These are the *only* places a generated output is judged — pure deterministic
functions over the closed-form oracles, with no learned reward model and no
human labelling. The benchmark aggregates them into the monotone metrics.
"""
from __future__ import annotations

from .attention import retained_positions
from .moe import router_hits_true
from .echo import echo_step_correct


def check_recall(needle_pos: int, retained: set) -> bool:
    """True if a long-context needle position is still present in the cache."""
    return needle_pos in retained


def check_routing(token_id: int, n_experts: int, top_k: int,
                  n_routers: int, eps: float = 1.0) -> bool:
    """True if the true expert for ``token_id`` survives top-k selection."""
    return router_hits_true(token_id, n_experts, top_k, n_routers, eps)


def check_echo(token_id: int, step: int, rank: int) -> bool:
    """True if the rank-``r`` ECHO head predicts future step ``step+1``."""
    return echo_step_correct(token_id, step, rank)
