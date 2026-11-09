# -*- coding: utf-8 -*-
"""Print the four monotone metrics vs skill (text table)."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from echoforge.config import load_config
from echoforge.kit import build_kit


def main():
    cfg = load_config()
    audit = build_kit(cfg)["audit"]
    print("%-8s %-9s %-9s %-9s %-9s %-8s"
          % ("skill", "recall", "routing", "echo", "quality", "speedup"))
    for s in cfg["bench"]["skills"]:
        r = audit.audit_skill(s)
        print("%-8.2f %-9.4f %-9.4f %-9.4f %-9.4f %-8.2f"
              % (s, r["longctx_recall"], r["routing_utility"],
                 r["echo_acc"], r["quality"], r["decode_speedup"]))


if __name__ == "__main__":
    main()
