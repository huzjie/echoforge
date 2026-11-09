# -*- coding: utf-8 -*-
"""Demo: ECHO multi-step prediction at increasing projection rank."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from echoforge.core.echo import hidden_state, echo_predict
from echoforge.core.oracle import next_k
from echoforge.constants import K_STEPS


def main():
    token = 42
    truth = next_k(token, K_STEPS)
    print("token=%d  true future=%s" % (token, truth))
    for rank in range(K_STEPS + 1):
        preds = echo_predict(hidden_state(token), rank)
        hits = sum(1 for p, t in zip(preds, truth) if p == t)
        print("rank=%d pred=%s  hits=%d/%d" % (rank, preds, hits, K_STEPS))


if __name__ == "__main__":
    main()
