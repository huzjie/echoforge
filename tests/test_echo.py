# -*- coding: utf-8 -*-
"""Tests for the ECHO multi-step prediction head."""
from __future__ import annotations

import unittest

from echoforge.core.echo import (
    hidden_state, echo_predict, echo_step_correct, echo_accuracy, echo_param_overhead,
)
from echoforge.core.oracle import next_k


class TestHiddenState(unittest.TestCase):
    def test_dim(self):
        self.assertEqual(len(hidden_state(0)), 32)

    def test_encodes_future(self):
        t = 5
        h = hidden_state(t)
        futs = next_k(t, 4)
        for j, v in enumerate(futs):
            self.assertAlmostEqual(h[j], (v + 0.5) / 256, places=9)


class TestEchoPredict(unittest.TestCase):
    def test_full_rank_exact(self):
        t = 7
        h = hidden_state(t)
        preds = echo_predict(h, 4)
        self.assertEqual(preds, next_k(t, 4))

    def test_partial_rank_none(self):
        h = hidden_state(9)
        preds = echo_predict(h, 2)
        self.assertEqual(preds[2], None)
        self.assertEqual(preds[3], None)

    def test_rank_zero_all_none(self):
        preds = echo_predict(hidden_state(1), 0)
        self.assertTrue(all(p is None for p in preds))


class TestEchoStepCorrect(unittest.TestCase):
    def test_monotone_in_rank(self):
        for t in (3, 11, 42, 255):
            prev = -1
            for rank in range(5):
                ok = echo_step_correct(t, 2, rank)
                self.assertGreaterEqual(int(ok) + 1e-9, prev)
                prev = int(ok)


class TestEchoAccuracy(unittest.TestCase):
    def test_full_rank(self):
        self.assertAlmostEqual(echo_accuracy(list(range(64)), 4), 1.0)

    def test_zero_rank(self):
        self.assertAlmostEqual(echo_accuracy(list(range(64)), 0), 0.0)

    def test_monotone_in_rank(self):
        tokens = list(range(128))
        prev = -1.0
        for rank in range(5):
            a = echo_accuracy(tokens, rank)
            self.assertGreaterEqual(a + 1e-9, prev)
            prev = a


class TestParamOverhead(unittest.TestCase):
    def test_tiny(self):
        self.assertLess(echo_param_overhead(4), 0.01)

    def test_monotone_in_rank(self):
        self.assertLess(echo_param_overhead(1), echo_param_overhead(4))


if __name__ == "__main__":
    unittest.main()
