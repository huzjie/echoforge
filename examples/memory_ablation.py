# -*- coding: utf-8 -*-
"""Ablation: KV memory ratio vs HCA cache budget (window / global)."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from echoforge.core.attention import memory_ratio


def main():
    print("seq_len=64 dim=32 mla_rank=8 full_ctx_every=8")
    print("%-8s %-8s %-10s" % ("window", "global", "memory_ratio"))
    for window in (0, 4, 8, 16):
        for g in (0, 8, 16, 32):
            r = memory_ratio(64, 32, 8, window, g, 8)
            print("%-8d %-8d %-10.4f" % (window, g, r))


if __name__ == "__main__":
    main()
