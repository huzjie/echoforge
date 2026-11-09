# -*- coding: utf-8 -*-
"""Tests for deterministic embeddings and expert vectors."""
from __future__ import annotations

import unittest
import math

from echoforge.core.embed import token_embed, expert_vector, embeds
from echoforge.constants import VOCAB, DIM


class TestEmbed(unittest.TestCase):
    def test_unit_norm(self):
        for t in range(20):
            v = token_embed(t)
            n = math.sqrt(sum(x * x for x in v))
            self.assertAlmostEqual(n, 1.0, places=6)

    def test_deterministic(self):
        self.assertEqual(token_embed(3), token_embed(3))

    def test_dim(self):
        self.assertEqual(len(token_embed(0)), DIM)

    def test_expert_unit_norm(self):
        for e in range(20):
            v = expert_vector(e)
            n = math.sqrt(sum(x * x for x in v))
            self.assertAlmostEqual(n, 1.0, places=6)

    def test_embeds_table(self):
        table = embeds(vocab=32)
        self.assertEqual(len(table), 32)
        self.assertEqual(len(table[0]), DIM)


if __name__ == "__main__":
    unittest.main()
