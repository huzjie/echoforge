# -*- coding: utf-8 -*-
"""Tests for the monotone trainer."""
from __future__ import annotations

import unittest

from tests.util import small_cfg
from echoforge.kit import build_kit
from echoforge.training.trainer import SkillTrainer, RewardTrainer


class TestSkillTrainer(unittest.TestCase):
    def test_monotone(self):
        t = SkillTrainer(skill=0.0, lr=0.05)
        prev = 0.0
        for _ in range(20):
            rec = t.train_step()
            self.assertGreater(rec["skill"], prev)
            prev = rec["skill"]
        self.assertLessEqual(prev, 1.0 + 1e-9)

    def test_delta_positive(self):
        t = SkillTrainer(skill=0.3, lr=0.1)
        self.assertGreater(t.train_step()["delta"], 0.0)


class TestRewardTrainer(unittest.TestCase):
    def test_monotone_reward(self):
        audit = build_kit(small_cfg())["audit"]
        t = RewardTrainer(audit, skill=0.0, lr=0.1)
        for _ in range(12):
            t.step()
        self.assertTrue(t.is_monotone())

    def test_reaches_one(self):
        audit = build_kit(small_cfg())["audit"]
        t = RewardTrainer(audit, skill=0.0, lr=0.2)
        for _ in range(40):
            t.step()
        self.assertAlmostEqual(t.history[-1]["reward"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
