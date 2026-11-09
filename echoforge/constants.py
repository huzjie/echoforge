# -*- coding: utf-8 -*-
"""Shared constants for the echoforge reference implementation.

These mirror the architectural knobs of a long-context sparse MoE language model
(ERNIE 5.5 Flash Base style) but are scaled down so the whole thing stays a
deterministic, laptop-friendly reference that runs in well under a second.
"""
from __future__ import annotations

VOCAB = 256            # vocabulary size
DIM = 32               # embedding / hidden dimension
SEQ_LEN = 64           # reference sequence length
LAYERS = 8             # transformer layers
FULL_CTX_EVERY = 8     # every N layers keep full context (MLA); the rest use HCA
MLA_RANK = 8           # latent rank of MLA key/value compression

WINDOW = 8             # HCA sliding-window size (most recent tokens)
MAX_GLOBAL = 16        # HCA global summary capacity (earliest tokens retained)

N_EXPERTS = 64         # MoE experts per layer
TOP_K = 8              # experts selected per token
MAX_ROUTERS = 8        # maximum number of routers in the multi-router ensemble

K_STEPS = 4            # ECHO multi-step prediction horizon
MAX_RANK = 4           # ECHO low-rank projection capacity (= K_STEPS)

N_ROUNDS = 16          # audit rounds (aligns needles / tokens / echo steps)
