# -*- coding: utf-8 -*-
"""Tests for the benchmark suite."""
from __future__ import annotations

import unittest

from tests.util import small_cfg
from echoforge import bench as B


class TestBench(unittest.TestCase):
    def setUp(self):
        self.cfg = small_cfg()

    def test_skill_table_shape(self):
        table = B.run_skill_table(self.cfg)
        self.assertEqual(len(table), 5)
        for r in table:
            for key in ("longctx_recall", "routing_utility", "echo_acc", "quality"):
                self.assertIn(key, r)

    def test_summarize(self):
        table = B.run_skill_table(self.cfg)
        s = B.summarize(table)
        self.assertEqual(s["longctx_recall"][1], 1.0)

    def test_monotone(self):
        table = B.run_skill_table(self.cfg)
        mono = B.check_monotone(table)
        self.assertTrue(all(mono.values()))

    def test_migration_bench(self):
        mig = B.run_migration_bench(self.cfg)
        self.assertLess(mig["memory_ratio"], 1.0)
        self.assertGreater(mig["decode_speedup_at_skill_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
