# -*- coding: utf-8 -*-
"""Core science: deterministic embeddings, oracles, HCA attention, MoE, ECHO."""
from __future__ import annotations

from .embed import token_embed, expert_vector, embeds
from .oracle import next_token, next_k, true_expert, Transition
from .attention import retained_positions, needle_recall, memory_ratio
from .moe import router_hits_true, routing_utility, select_topk
from .echo import hidden_state, echo_predict, echo_step_correct, echo_accuracy
from .validate import check_recall, check_routing, check_echo
from . import metrics as M

__all__ = [
    "token_embed", "expert_vector", "embeds",
    "next_token", "next_k", "true_expert", "Transition",
    "retained_positions", "needle_recall", "memory_ratio",
    "router_hits_true", "routing_utility", "select_topk",
    "hidden_state", "echo_predict", "echo_step_correct", "echo_accuracy",
    "check_recall", "check_routing", "check_echo",
    "M",
]
