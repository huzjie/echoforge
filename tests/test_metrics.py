# -*- coding: utf-8 -*-
"""Tests for metric aggregation."""
from __future__ import annotations

import unittest

from tests.util import small_cfg
from echoforge.config import load_config
from echoforge.kit import build_kit
from echoforge.core import metrics as M


class TestSkillParams(unittest.TestCase):
    def test_endpoints(self):
        cfg = small_cfg()
        self.assertEqual(M.skill_params(cfg, 0.0), (0, 1, 0))
        self.assertEqual(M.skill_params(cfg, 1.0), (16, 8, 4))

    def test_clamped(self):
        cfg = small_cfg()
        self.assertEqual(M.skill_params(cfg, -1.0)[0], 0)
        self.assertEqual(M.skill_params(cfg, 2.0)[0], 16)


class TestAggregate(unittest.TestCase):
    def setUp(self):
        self.cfg = small_cfg()
        self.world = build_kit(self.cfg)["world"]

    def test_skill_one_perfect(self):
        m = M.aggregate(self.world, 1.0)
        self.assertAlmostEqual(m["longctx_recall"], 1.0)
        self.assertAlmostEqual(m["routing_utility"], 1.0)
        self.assertAlmostEqual(m["echo_acc"], 1.0)
        self.assertAlmostEqual(m["quality"], 1.0)

    def test_skill_zero_baseline(self):
        m = M.aggregate(self.world, 0.0)
        self.assertAlmostEqual(m["longctx_recall"], 0.0)
        self.assertAlmostEqual(m["echo_acc"], 0.0)
        self.assertAlmostEqual(m["quality"], 0.0)

    def test_monotone(self):
        prev = None
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            m = M.aggregate(self.world, s)
            for key in ("longctx_recall", "routing_utility", "echo_acc", "quality"):
                if prev is not None:
                    self.assertGreaterEqual(m[key] + 1e-9, prev[key])
            prev = m

    def test_memory_ratio_constant(self):
        a = M.aggregate(self.world, 0.5)["memory_ratio"]
        b = M.aggregate(self.world, 0.5)["memory_ratio"]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
