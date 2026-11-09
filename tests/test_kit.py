# -*- coding: utf-8 -*-
"""Tests for the kit and world wiring."""
from __future__ import annotations

import unittest

from tests.util import small_cfg
from echoforge.kit import build_kit
from echoforge.world.corpus import build_corpus
from echoforge.world.generator import generate_sequence, needle_positions


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate_sequence(0, 32), generate_sequence(0, 32))

    def test_length(self):
        self.assertEqual(len(generate_sequence(1, 50)), 50)

    def test_distinct_seed(self):
        self.assertNotEqual(generate_sequence(0, 16), generate_sequence(1, 16))

    def test_needles(self):
        self.assertEqual(needle_positions(4), [0, 1, 2, 3])


class TestCorpus(unittest.TestCase):
    def test_build(self):
        c = build_corpus(small_cfg())
        self.assertEqual(len(c.sequences), 4)
        self.assertEqual(c.seq_len, 64)
        self.assertEqual(c.needles, list(range(16)))


class TestKit(unittest.TestCase):
    def test_build(self):
        kit = build_kit(small_cfg())
        self.assertIn("world", kit)
        self.assertIn("audit", kit)
        self.assertIn("corpus", kit)

    def test_world_run(self):
        kit = build_kit(small_cfg())
        m = kit["world"].run(1.0)
        self.assertAlmostEqual(m["quality"], 1.0)


if __name__ == "__main__":
    unittest.main()
