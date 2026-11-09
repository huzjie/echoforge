# -*- coding: utf-8 -*-
"""Tests for the verifiable check functions."""
from __future__ import annotations

import unittest

from echoforge.core.validate import check_recall, check_routing, check_echo
from echoforge.core.attention import retained_positions


class TestCheckRecall(unittest.TestCase):
    def test_present(self):
        self.assertTrue(check_recall(0, {0, 1}))

    def test_absent(self):
        self.assertFalse(check_recall(5, {0, 1}))


class TestCheckRouting(unittest.TestCase):
    def test_perfect_zero_noise(self):
        self.assertTrue(check_routing(0, 64, 8, 8, eps=0.0))

    def test_boolean(self):
        self.assertIsInstance(check_routing(0, 64, 8, 1), bool)


class TestCheckEcho(unittest.TestCase):
    def test_full_rank(self):
        self.assertTrue(check_echo(0, 0, 4))

    def test_zero_rank(self):
        self.assertFalse(check_echo(0, 0, 0))


class TestIntegration(unittest.TestCase):
    def test_recall_wiring(self):
        retained = retained_positions(64, 8, 16)
        self.assertTrue(check_recall(0, retained))
        self.assertFalse(check_recall(40, retained))


if __name__ == "__main__":
    unittest.main()
