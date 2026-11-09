# -*- coding: utf-8 -*-
"""Tests for HCA hybrid attention + MLA memory budget."""
from __future__ import annotations

import unittest

from echoforge.core.attention import (
    retained_positions, needle_recall, full_attention_kv, mla_kv, hca_kv, memory_ratio,
)


class TestRetainedPositions(unittest.TestCase):
    def test_window_included(self):
        retained = retained_positions(64, 8, 0)
        self.assertIn(63, retained)
        self.assertIn(56, retained)
        self.assertNotIn(0, retained)

    def test_global_included(self):
        retained = retained_positions(64, 0, 4)
        self.assertIn(0, retained)
        self.assertIn(3, retained)
        self.assertNotIn(4, retained)

    def test_union(self):
        retained = retained_positions(64, 8, 4)
        self.assertEqual(len(retained), 12)  # 8 window + 4 global, disjoint


class TestNeedleRecall(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(needle_recall([0, 1, 2], set()), 0.0)

    def test_full(self):
        self.assertEqual(needle_recall([0, 1, 2], {0, 1, 2}), 1.0)

    def test_partial(self):
        self.assertAlmostEqual(needle_recall([0, 1, 2, 3], {0, 1}), 0.5)

    def test_empty_needles(self):
        self.assertEqual(needle_recall([], {0, 1}), 0.0)


class TestMemoryBudget(unittest.TestCase):
    def test_full_attention(self):
        self.assertEqual(full_attention_kv(64, 32), 64 * 2 * 32)

    def test_mla_smaller(self):
        self.assertLess(mla_kv(64, 8), full_attention_kv(64, 32))

    def test_hca_smaller(self):
        self.assertLess(hca_kv(8, 16, 8), mla_kv(64, 8))

    def test_memory_ratio_range(self):
        r = memory_ratio(64, 32, 8, 8, 16, 8)
        self.assertGreater(r, 0.0)
        self.assertLess(r, 1.0)

    def test_memory_ratio_monotone_in_global(self):
        lo = memory_ratio(64, 32, 8, 8, 0, 8)
        hi = memory_ratio(64, 32, 8, 8, 16, 8)
        self.assertLess(lo, hi)


if __name__ == "__main__":
    unittest.main()
