# -*- coding: utf-8 -*-
"""Tests for the verifiable-reward audit."""
from __future__ import annotations

import unittest

from tests.util import small_cfg
from echoforge.kit import build_kit


class TestEchoAudit(unittest.TestCase):
    def setUp(self):
        self.audit = build_kit(small_cfg())["audit"]

    def test_audit_skill_keys(self):
        r = self.audit.audit_skill(0.5)
        for key in ("longctx_recall", "routing_utility", "echo_acc", "quality",
                    "memory_ratio", "decode_speedup", "param_overhead", "skill"):
            self.assertIn(key, r)

    def test_mean_reward_monotone(self):
        prev = -1.0
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            r = self.audit.mean_reward(s)
            self.assertGreaterEqual(r + 1e-9, prev)
            prev = r

    def test_mean_reward_range(self):
        self.assertAlmostEqual(self.audit.mean_reward(1.0), 1.0)
        self.assertAlmostEqual(self.audit.mean_reward(0.0), 0.0)

    def test_speedup(self):
        self.assertEqual(self.audit.speedup(1.0), 4.0)

    def test_memory_ratio(self):
        self.assertLess(self.audit.memory_ratio(), 1.0)


if __name__ == "__main__":
    unittest.main()
