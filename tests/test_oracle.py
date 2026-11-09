# -*- coding: utf-8 -*-
"""Tests for the closed-form oracles."""
from __future__ import annotations

import unittest

from echoforge.core.oracle import next_token, next_k, true_expert, Transition
from echoforge.constants import VOCAB


class TestNextToken(unittest.TestCase):
    def test_is_permutation(self):
        seen = {next_token(t) for t in range(VOCAB)}
        self.assertEqual(len(seen), VOCAB)

    def test_deterministic(self):
        self.assertEqual(next_token(0), next_token(0))
        self.assertNotEqual(next_token(0), next_token(1))

    def test_range(self):
        for t in range(100):
            self.assertGreaterEqual(next_token(t), 0)
            self.assertLess(next_token(t), VOCAB)


class TestNextK(unittest.TestCase):
    def test_length(self):
        self.assertEqual(len(next_k(5, 4)), 4)

    def test_chaining(self):
        t = 5
        futs = next_k(t, 3)
        self.assertEqual(futs[0], next_token(t))
        self.assertEqual(futs[1], next_token(next_token(t)))

    def test_k_zero(self):
        self.assertEqual(next_k(5, 0), [])


class TestTrueExpert(unittest.TestCase):
    def test_closed_form(self):
        for t in range(50):
            self.assertEqual(true_expert(t, 64), t % 64)

    def test_range(self):
        for t in range(200):
            self.assertGreaterEqual(true_expert(t, 8), 0)
            self.assertLess(true_expert(t, 8), 8)


class TestTransition(unittest.TestCase):
    def test_bundle(self):
        tr = Transition(n_experts=16, k_steps=2)
        self.assertEqual(tr.true_expert(20), 4)
        self.assertEqual(len(tr.next_k(7)), 2)


if __name__ == "__main__":
    unittest.main()
