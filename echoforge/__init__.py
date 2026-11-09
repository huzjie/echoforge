# -*- coding: utf-8 -*-
"""echoforge — long-context HCA attention + multi-router MoE + ECHO multi-step
prediction (ERNIE 5.5 Flash Base style) reference implementation.

A zero-dependency reference implementation of three reusable pretraining
techniques from a 280B sparse MoE base model: hybrid cache attention (long
context on a fixed KV budget), a multi-router latent MoE (routing quality via
ensemble variance reduction), and ECHO multi-step prediction (decode several
tokens per forward pass by reusing the backbone's last hidden). See README.md
and docs/ for the full technical walkthrough.
"""
from __future__ import annotations

from .version import __version__
from .core.embed import token_embed, expert_vector
from .core.oracle import next_token, next_k, true_expert, Transition
from .core.attention import retained_positions, needle_recall, memory_ratio
from .core.moe import router_hits_true, routing_utility
from .core.echo import hidden_state, echo_predict, echo_accuracy

__all__ = [
    "__version__",
    "token_embed", "expert_vector",
    "next_token", "next_k", "true_expert", "Transition",
    "retained_positions", "needle_recall", "memory_ratio",
    "router_hits_true", "routing_utility",
    "hidden_state", "echo_predict", "echo_accuracy",
]
