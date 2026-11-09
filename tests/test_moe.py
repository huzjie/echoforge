# -*- coding: utf-8 -*-
"""Tests for the multi-router latent MoE."""
from __future__ import annotations

import unittest

from echoforge.core.moe import (
    noise_scale, expert_scores, select_topk, router_hits_true, routing_utility,
)
from echoforge.core.oracle import true_expert


class TestNoiseScale(unittest.TestCase):
    def test_decreasing(self):
        self.assertGreater(noise_scale(1), noise_scale(4))
        self.assertGreater(noise_scale(4), noise_scale(16))


class TestExpertScores(unittest.TestCase):
    def test_shape(self):
        scores = expert_scores(0, 64, 1)
        self.assertEqual(len(scores), 64)

    def test_true_expert_has_base_advantage(self):
        t = 17
        scores = expert_scores(t, 64, 1, eps=0.0)
        g = true_expert(t, 64)
        self.assertAlmostEqual(scores[g], 1.0)
        for e in range(64):
            if e != g:
                self.assertAlmostEqual(scores[e], 0.0)


class TestSelectTopK(unittest.TestCase):
    def test_size(self):
        self.assertEqual(len(select_topk([0.1, 0.9, 0.5, 0.3], 2)), 2)

    def test_picks_best(self):
        self.assertIn(1, select_topk([0.1, 0.9, 0.5, 0.3], 2))


class TestRoutingUtility(unittest.TestCase):
    def test_zero_noise_perfect(self):
        tokens = list(range(128))
        u = routing_utility(tokens, 64, 8, 1, eps=0.0)
        self.assertAlmostEqual(u, 1.0)

    def test_monotone_in_routers(self):
        tokens = list(range(256))
        prev = -1.0
        for r in (1, 2, 4, 8):
            u = routing_utility(tokens, 64, 8, r, eps=1.0)
            self.assertGreaterEqual(u + 1e-9, prev)
            prev = u

    def test_range(self):
        u = routing_utility(list(range(64)), 64, 8, 4, eps=1.0)
        self.assertGreaterEqual(u, 0.0)
        self.assertLessEqual(u, 1.0)


class TestRouterHitsTrue(unittest.TestCase):
    def test_hits_with_many_routers(self):
        t = 123
        self.assertTrue(router_hits_true(t, 64, 8, 8, eps=0.0))

    def test_boolean(self):
        self.assertIsInstance(router_hits_true(0, 64, 8, 1, eps=1.0), bool)


if __name__ == "__main__":
    unittest.main()
