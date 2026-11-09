# -*- coding: utf-8 -*-
"""Closed-form ground-truth oracles (zero annotation, zero learned reward).

The reference model is validated against *closed-form* truth rather than human
labels:

* **next-token** — an invertible affine map ``(t*7 + 13) mod 256`` (7 is coprime
  to 256, so it is a permutation of the vocabulary);
* **true expert** — ``t mod E`` assigns every token a canonical specialist;
* **next-k tokens** — iterating the affine map gives the multi-step future the
  ECHO head is trained to predict.

Because every answer is a pure function of the input, correctness can be judged
exactly, which is what makes the verifiable-reward story possible.
"""
from __future__ import annotations

from ..constants import VOCAB

_A = 7
_B = 13


def next_token(t: int) -> int:
    """Closed-form successor of token ``t`` under the affine permutation."""
    return (_A * t + _B) % VOCAB


def next_k(t: int, k: int) -> list:
    """The ``k`` future tokens after ``t`` (step 1 .. step k)."""
    out = []
    for _ in range(k):
        t = next_token(t)
        out.append(t)
    return out


def true_expert(t: int, n_experts: int) -> int:
    """Canonical specialist expert for token ``t`` (closed form)."""
    return t % n_experts


class Transition:
    """A tiny bundle of the closed-form oracle for convenience."""

    def __init__(self, n_experts: int = 64, k_steps: int = 4):
        self.n_experts = n_experts
        self.k_steps = k_steps

    def next_token(self, t: int) -> int:
        return next_token(t)

    def next_k(self, t: int, k: int = None) -> list:
        return next_k(t, self.k_steps if k is None else k)

    def true_expert(self, t: int) -> int:
        return true_expert(t, self.n_experts)
