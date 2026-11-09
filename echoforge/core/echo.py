# -*- coding: utf-8 -*-
"""ECHO multi-step prediction: one hidden, many future tokens.

The backbone's last layer already encodes the future: coordinate ``j`` of the
hidden state carries the token ``next^j(t)``. An ECHO head reuses that single
hidden with a rank-``r`` projection and predicts steps 1..K from it, instead of
attaching a full ``V x d`` head per extra step.

The rank ``r`` is the honest capacity knob: the head can only recover the
coordinates it keeps (steps ``j < r``). Steps beyond the rank return ``None``
("cannot predict"), so ``echo_accuracy`` is exactly ``min(r, K) / K`` — monotone
0 -> 1 in rank, with a parameter overhead of ``r*d`` versus the ``K*V*d`` that K
independent heads would cost.
"""
from __future__ import annotations

from ..utils.rng import stable_float
from ..constants import VOCAB, DIM, K_STEPS, MAX_RANK
from .oracle import next_k


def hidden_state(t: int, dim: int = DIM, k_steps: int = K_STEPS) -> list:
    """Backbone last-layer hidden: future tokens in the first ``k_steps`` coords."""
    h = [0.0] * dim
    futs = next_k(t, k_steps)
    for j, v in enumerate(futs):
        if j < dim:
            h[j] = (v + 0.5) / VOCAB
    for j in range(k_steps, dim):
        h[j] = stable_float("hid:%d:%d" % (t, j), -1.0, 1.0)
    return h


def echo_predict(h: list, rank: int, k_steps: int = K_STEPS) -> list:
    """Predict steps 1..K via the rank-``r`` projection of ``h``.

    Steps ``j < rank`` decode the preserved coordinate; steps ``>= rank`` are
    ``None`` (the low-rank head has no capacity left for them).
    """
    preds = []
    for j in range(k_steps):
        if j < rank:
            preds.append(int(round(h[j] * VOCAB - 0.5)) % VOCAB)
        else:
            preds.append(None)
    return preds


def echo_step_correct(t: int, step: int, rank: int, k_steps: int = K_STEPS) -> bool:
    """True if the rank-``r`` head predicts future step ``step+1`` of ``t``."""
    preds = echo_predict(hidden_state(t, k_steps=k_steps), rank, k_steps=k_steps)
    if preds[step] is None:
        return False
    return preds[step] == next_k(t, k_steps)[step]


def echo_accuracy(token_ids, rank: int, k_steps: int = K_STEPS) -> float:
    """Fraction of (token, step) predictions the rank-``r`` head gets right."""
    if not token_ids:
        return 0.0
    steps = [i % k_steps for i in range(len(token_ids))]
    hits = sum(1 for i, t in enumerate(token_ids)
               if echo_step_correct(t, steps[i], rank, k_steps))
    return hits / len(token_ids)


def echo_param_overhead(rank: int, dim: int = DIM, vocab: int = VOCAB,
                        k_steps: int = K_STEPS) -> float:
    """ECHO extra params (``rank*dim``) over K independent heads (``K*vocab*dim``)."""
    denom = k_steps * vocab * dim
    return (rank * dim) / denom if denom else 1.0
